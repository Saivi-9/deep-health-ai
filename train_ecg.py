"""
ECG Multi-Architecture Training & Evaluation CLI
------------------------------------------------
Trains & benchmarks:
1. Conv1dAutoencoder (Unsupervised NSR reconstruction anomaly detector) -> `saved_models/best_ecg_autoencoder.pt`
2. ECGResNet1D (Supervised 5-class Lead-II ResNet1D classifier) -> `saved_models/ecg_resnet1d.pt`
3. ECGMorphologyLightGBM & ECGMorphologyHistGB (18 ECG biomarkers) -> `saved_models/ecg_lgbm.joblib`
4. ECGResNet1D_MITBIH (Supervised 5-class MIT-BIH real beats classifier) -> `saved_models/ecg_mitbih_resnet1d.pt`

Usage:
    python train_ecg.py [--epochs 15] [--lr 0.001] [--batch-size 32]
"""

import os
import copy
import argparse
import json
from typing import Dict, Any, Tuple
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier

from src.ecg.dataset import load_ecg_dataloaders, load_supervised_ecg_dataloaders, load_mitbih_dataloaders
from src.ecg.model import Conv1dAutoencoder, ECGResNet1D
from src.ecg.trainer import ECGAutoencoderTrainer
from src.ecg.anomaly_detector import ECGAnomalyDetector
from src.ecg.evaluate import evaluate_ecg_anomaly_detection
from src.ecg.telemetry import ECG_FEATURE_NAMES, extract_ecg_feature_matrix
from src.utils.metrics import compute_classification_metrics
from src.utils.plotting import plot_training_history
from src.utils.logger import get_logger

logger = get_logger("TrainECGCLI")


def train_supervised_resnet1d(
    train_loader,
    val_loader,
    test_loader,
    num_classes: int,
    save_path: str,
    epochs: int = 12,
    lr: float = 0.0015,
    device: str = "cpu",
    label: str = "ECGResNet1D",
) -> Tuple[ECGResNet1D, Dict[str, Any], np.ndarray]:
    """Trains a 5-class ECGResNet1D classifier and returns model, metrics, and test probs."""
    model = ECGResNet1D(in_channels=1, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-5)

    best_weights = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")

    for ep in range(1, epochs + 1):
        model.train()
        for sigs, lbls in train_loader:
            sigs, lbls = sigs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(sigs), lbls)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_total = 0
        with torch.no_grad():
            for sigs, lbls in val_loader:
                sigs, lbls = sigs.to(device), lbls.to(device)
                out = model(sigs)
                val_loss += criterion(out, lbls).item() * sigs.size(0)
                val_total += sigs.size(0)
        val_loss /= max(1, val_total)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(best_weights, save_path)
    model.load_state_dict(best_weights)
    model.eval()

    y_true, y_pred, probs_all = [], [], []
    with torch.no_grad():
        for sigs, lbls in test_loader:
            logits = model(sigs.to(device))
            probs = F.softmax(logits, dim=1).cpu().numpy()
            y_pred.extend(np.argmax(probs, axis=1).tolist())
            y_true.extend(lbls.numpy().tolist())
            probs_all.append(probs)

    probs_mat = np.concatenate(probs_all, axis=0)
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    metrics = {
        "multiclass_accuracy": round(float(accuracy_score(y_true_arr, y_pred_arr)), 4),
        "macro_f1": round(float(f1_score(y_true_arr, y_pred_arr, average="macro")), 4),
        "multiclass_roc_auc": round(float(roc_auc_score(y_true_arr, probs_mat, multi_class="ovr")), 4),
    }
    logger.info(
        f"[{label}] 5-Class Acc: {metrics['multiclass_accuracy']*100:.2f}% | "
        f"Macro-F1: {metrics['macro_f1']:.4f} | OVR-AUC: {metrics['multiclass_roc_auc']:.4f}"
    )
    return model, metrics, probs_mat


def train_and_benchmark_ecg(
    epochs: int = 16,
    lr: float = 0.001,
    batch_size: int = 32,
    data_dir: str = "data/ecg",
    save_model: str = "saved_models/best_ecg_autoencoder.pt",
    figures_dir: str = "reports/figures",
) -> Dict[str, Any]:
    if not os.path.exists(os.path.join(data_dir, "ecg_train_normal.npy")):
        logger.info("ECG Dataset not found. Generating datasets...")
        from data.generate_datasets import generate_all_datasets
        generate_all_datasets()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using compute device: {device}")

    # 1. Conv1dAutoencoder (Unsupervised NSR baseline)
    train_loader, val_loader, test_loader, test_signals, test_labels = load_ecg_dataloaders(
        data_dir=data_dir, batch_size=batch_size
    )
    ae_model = Conv1dAutoencoder(in_channels=1)
    ae_params = sum(p.numel() for p in ae_model.parameters() if p.requires_grad)
    logger.info(f"1/4 Training Conv1dAutoencoder ({ae_params:,} params)...")
    ae_trainer = ECGAutoencoderTrainer(
        model=ae_model, train_loader=train_loader, val_loader=val_loader,
        learning_rate=lr, num_epochs=epochs, patience=5, device=device,
    )
    history = ae_trainer.fit(save_path=save_model)
    plot_training_history(history, os.path.join(figures_dir, "ecg_training_history.png"), title="1D ECG Autoencoder")

    detector = ECGAnomalyDetector(model=ae_model, device=device)
    threshold = detector.calibrate_threshold(val_loader, percentile=97.5)
    ae_metrics = evaluate_ecg_anomaly_detection(
        detector=detector, test_loader=test_loader, test_labels=test_labels, figures_dir=figures_dir
    )

    models_comparison: Dict[str, Any] = {
        "Conv1dAutoencoder": {
            "architecture": "Unsupervised 1D CNN Autoencoder (NSR Reconstruction)",
            "trainable_parameters": ae_params,
            "checkpoint": save_model,
            "binary_anomaly_metrics": ae_metrics,
        }
    }

    # 2. Supervised 5-Class Lead-II Models
    classes = ["nsr", "pvc", "afib", "st_elevation", "lbbb"]
    if os.path.exists(os.path.join(data_dir, "ecg_supervised_train.npz")):
        (
            sup_tr_loader, sup_val_loader, sup_te_loader,
            classes, X_sup, y_sup, X_test, y_test_multi, y_test_bin
        ) = load_supervised_ecg_dataloaders(data_dir=data_dir, batch_size=batch_size)

        logger.info("2/4 Training Supervised 5-Class ECGResNet1D (Lead-II)...")
        resnet1d, resnet_multi_m, resnet_probs = train_supervised_resnet1d(
            sup_tr_loader, sup_val_loader, sup_te_loader,
            num_classes=len(classes), save_path="saved_models/ecg_resnet1d.pt",
            epochs=12, lr=0.0015, device=device, label="ECGResNet1D-LeadII",
        )
        # Convert 5-class probabilities to binary anomaly probability (1 - P(nsr))
        anom_prob_resnet = 1.0 - resnet_probs[:, 0]
        resnet_bin_m = compute_classification_metrics(
            y_test_bin, (anom_prob_resnet >= 0.5).astype(int), anom_prob_resnet
        )
        models_comparison["ECGResNet1D"] = {
            "architecture": "Supervised 4-Stage 1D Residual CNN (5-Class Differential)",
            "trainable_parameters": sum(p.numel() for p in resnet1d.parameters() if p.requires_grad),
            "checkpoint": "saved_models/ecg_resnet1d.pt",
            "binary_anomaly_metrics": resnet_bin_m,
            "multiclass_metrics": resnet_multi_m,
        }

        # 3 & 4. 18-Biomarker Tabular Models (LightGBM & HistGradientBoosting)
        logger.info("3/4 Extracting 18 Morphological & Spectral ECG Features for GBDT tournament...")
        X_feat_train = extract_ecg_feature_matrix(X_sup, sampling_rate=250)
        X_feat_test = extract_ecg_feature_matrix(X_test, sampling_rate=250)

        lgbm_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LGBMClassifier(n_estimators=220, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)),
        ])
        lgbm_pipe.fit(X_feat_train, y_sup)
        lgbm_probs = lgbm_pipe.predict_proba(X_feat_test)
        lgbm_preds = np.argmax(lgbm_probs, axis=1)
        lgbm_anom = 1.0 - lgbm_probs[:, 0]
        lgbm_bin_m = compute_classification_metrics(y_test_bin, (lgbm_anom >= 0.5).astype(int), lgbm_anom)
        lgbm_multi_m = {
            "multiclass_accuracy": round(float(accuracy_score(y_test_multi, lgbm_preds)), 4),
            "macro_f1": round(float(f1_score(y_test_multi, lgbm_preds, average="macro")), 4),
            "multiclass_roc_auc": round(float(roc_auc_score(y_test_multi, lgbm_probs, multi_class="ovr")), 4),
        }
        os.makedirs("saved_models", exist_ok=True)
        joblib.dump(
            {"pipeline": lgbm_pipe, "feature_names": ECG_FEATURE_NAMES, "classes": classes},
            "saved_models/ecg_lgbm.joblib",
        )
        models_comparison["ECGMorphologyLightGBM"] = {
            "architecture": "18-Feature Electrophysiology + FFT LightGBM Classifier",
            "trainable_parameters": len(ECG_FEATURE_NAMES),
            "checkpoint": "saved_models/ecg_lgbm.joblib",
            "binary_anomaly_metrics": lgbm_bin_m,
            "multiclass_metrics": lgbm_multi_m,
        }

        hgb_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, random_state=42)),
        ])
        hgb_pipe.fit(X_feat_train, y_sup)
        hgb_probs = hgb_pipe.predict_proba(X_feat_test)
        hgb_preds = np.argmax(hgb_probs, axis=1)
        hgb_anom = 1.0 - hgb_probs[:, 0]
        models_comparison["ECGMorphologyHistGB"] = {
            "architecture": "18-Feature Electrophysiology + FFT HistGradientBoosting",
            "trainable_parameters": len(ECG_FEATURE_NAMES),
            "checkpoint": "saved_models/ecg_lgbm.joblib",
            "binary_anomaly_metrics": compute_classification_metrics(y_test_bin, (hgb_anom >= 0.5).astype(int), hgb_anom),
            "multiclass_metrics": {
                "multiclass_accuracy": round(float(accuracy_score(y_test_multi, hgb_preds)), 4),
                "macro_f1": round(float(f1_score(y_test_multi, hgb_preds, average="macro")), 4),
                "multiclass_roc_auc": round(float(roc_auc_score(y_test_multi, hgb_probs, multi_class="ovr")), 4),
            },
        }

    # 5. Real PhysioNet MIT-BIH 10k Training & Benchmark
    mitbih_benchmark = None
    mitbih_loaders = load_mitbih_dataloaders(batch_size=64)
    if mitbih_loaders is not None:
        m_tr, m_val, m_te, m_classes = mitbih_loaders
        logger.info("4/4 Training Supervised ECGResNet1D on Real 10,000-Beat PhysioNet MIT-BIH Cohort...")
        _, mitbih_m, _ = train_supervised_resnet1d(
            m_tr, m_val, m_te,
            num_classes=len(m_classes), save_path="saved_models/ecg_mitbih_resnet1d.pt",
            epochs=8, lr=0.0015, device=device, label="ECGResNet1D-MITBIH-10k",
        )
        mitbih_benchmark = {
            "dataset": "dheerajthuvara/ecg-arrhythmia-data (10k real stratified beats)",
            "classes": m_classes,
            "checkpoint": "saved_models/ecg_mitbih_resnet1d.pt",
            **mitbih_m,
        }

    metadata = {
        "threshold": float(threshold),
        "metrics": ae_metrics,
        "class_names": classes,
        "models_comparison": models_comparison,
        "mitbih_benchmark": mitbih_benchmark,
    }
    metrics_save_path = "reports/ecg_metrics.json"
    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
    with open(metrics_save_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved multi-model ECG metrics to {metrics_save_path}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train Multi-Architecture ECG Anomaly & Arrhythmia Classifiers")
    parser.add_argument("--epochs", type=int, default=16, help="Epochs for Conv1dAutoencoder")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--data-dir", type=str, default="data/ecg", help="Data directory")
    parser.add_argument("--save-model", type=str, default="saved_models/best_ecg_autoencoder.pt", help="Autoencoder checkpoint")
    parser.add_argument("--figures-dir", type=str, default="reports/figures", help="Figures output directory")
    args = parser.parse_args()

    train_and_benchmark_ecg(
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        data_dir=args.data_dir, save_model=args.save_model, figures_dir=args.figures_dir,
    )


if __name__ == "__main__":
    main()
