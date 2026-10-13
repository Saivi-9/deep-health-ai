"""
Deep Health AI - End-to-End Multi-Modal Practice Benchmark & Demo
-----------------------------------------------------------------
Orchestrates:
1. Synthetic dataset synthesis (Chest X-Rays + Lead-II ECGs)
2. Chest X-Ray Residual CNN Training, Validation, and Grad-CAM generation
3. ECG 1D Convolutional Autoencoder Training and Anomaly Detection
4. Generation of all clinical evaluation metrics and publication-ready figures
"""

import os
import json
import torch

from data.generate_datasets import generate_all_datasets
from src.vision.dataset import get_vision_dataloaders
from src.vision.model import ResidualChestCNN
from src.vision.trainer import VisionTrainer
from src.vision.evaluate import evaluate_vision_model

from src.ecg.dataset import load_ecg_dataloaders
from src.ecg.model import Conv1dAutoencoder
from src.ecg.trainer import ECGAutoencoderTrainer
from src.ecg.anomaly_detector import ECGAnomalyDetector
from src.ecg.evaluate import evaluate_ecg_anomaly_detection

from src.utils.plotting import plot_training_history
from src.utils.logger import get_logger

logger = get_logger("DeepHealthDemo")


def run_full_pipeline():
    logger.info("================================================================")
    logger.info("STARTING DEEP HEALTH AI MULTI-MODAL PIPELINE")
    logger.info("================================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Target compute hardware: {device}")

    figures_dir = "reports/figures"
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Generate datasets
    logger.info("[Step 1/4] Synthesizing Medical Datasets...")
    generate_all_datasets(base_dir=".")

    # 2. Chest Radiograph Pipeline
    logger.info("[Step 2/4] Executing Chest X-Ray Vision Pipeline...")
    train_loader_v, val_loader_v, test_loader_v, class_names = get_vision_dataloaders(
        data_dir="data/vision",
        batch_size=32
    )
    vision_model = ResidualChestCNN(in_channels=1, num_classes=2)
    vision_trainer = VisionTrainer(
        model=vision_model,
        train_loader=train_loader_v,
        val_loader=val_loader_v,
        learning_rate=0.001,
        num_epochs=10,
        patience=4,
        device=device
    )
    v_history = vision_trainer.fit(save_path="saved_models/best_chest_cnn.pt")
    plot_training_history(v_history, os.path.join(figures_dir, "vision_training_history.png"), title="Chest X-Ray Residual CNN")

    vision_metrics = evaluate_vision_model(
        model=vision_model,
        test_loader=test_loader_v,
        class_names=class_names,
        figures_dir=figures_dir,
        device=device
    )

    # 3. ECG Bio-Signal Pipeline
    logger.info("[Step 3/4] Executing ECG 1D Autoencoder Pipeline...")
    train_loader_e, val_loader_e, test_loader_e, test_signals, test_labels = load_ecg_dataloaders(
        data_dir="data/ecg",
        batch_size=32
    )
    ecg_model = Conv1dAutoencoder(in_channels=1)
    ecg_trainer = ECGAutoencoderTrainer(
        model=ecg_model,
        train_loader=train_loader_e,
        val_loader=val_loader_e,
        learning_rate=0.001,
        num_epochs=15,
        patience=4,
        device=device
    )
    e_history = ecg_trainer.fit(save_path="saved_models/best_ecg_autoencoder.pt")
    plot_training_history(e_history, os.path.join(figures_dir, "ecg_training_history.png"), title="1D ECG Autoencoder")

    detector = ECGAnomalyDetector(model=ecg_model, device=device)
    threshold = detector.calibrate_threshold(val_loader_e, percentile=97.5)

    ecg_metrics = evaluate_ecg_anomaly_detection(
        detector=detector,
        test_loader=test_loader_e,
        test_labels=test_labels,
        figures_dir=figures_dir
    )

    # 4. Save consolidated metrics summary
    logger.info("[Step 4/4] Compiling Diagnostic Benchmark Report...")
    summary = {
        "chest_xray_vision": {
            "model": "ResidualChestCNN + SpatialAttention + GradCAM",
            "accuracy": vision_metrics["accuracy"],
            "sensitivity": vision_metrics["sensitivity_recall"],
            "specificity": vision_metrics["specificity"],
            "precision": vision_metrics["precision"],
            "f1_score": vision_metrics["f1_score"],
            "roc_auc": vision_metrics.get("roc_auc"),
            "pr_auc": vision_metrics.get("pr_auc"),
            "confusion_matrix": vision_metrics["confusion_matrix"]
        },
        "ecg_cardiac_anomaly": {
            "model": "1D-ConvAutoencoder (Temporal Latent Bottleneck)",
            "calibrated_threshold": threshold,
            "accuracy": ecg_metrics["accuracy"],
            "arrhythmia_sensitivity": ecg_metrics["sensitivity_recall"],
            "sinus_specificity": ecg_metrics["specificity"],
            "precision": ecg_metrics["precision"],
            "f1_score": ecg_metrics["f1_score"],
            "reconstruction_roc_auc": ecg_metrics.get("roc_auc"),
            "confusion_matrix": ecg_metrics["confusion_matrix"]
        }
    }

    summary_file = "reports/metrics_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("================================================================")
    logger.info(f"DEMO COMPLETED SUCCESSFULLY! Results saved to {summary_file}")
    logger.info("Figures generated in reports/figures/:")
    for f in sorted(os.listdir(figures_dir)):
        logger.info(f"  - {f}")
    logger.info("================================================================")
    return summary


if __name__ == "__main__":
    run_full_pipeline()
