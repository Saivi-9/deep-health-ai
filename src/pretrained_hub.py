"""
Hugging Face Medical Pre-Trained Model Hub Integration
------------------------------------------------------
Provides live Hugging Face Hub search, verified multi-modal medical model
catalog, lazy cached model loading, single-patient inference (including Grad-CAM
for TorchVision backbones), and reproducible empirical benchmarking across:
- Chest X-Ray Pneumonia Classifiers (MobileNetV2-256-Gray, MobileNetV2-320, DenseNet121-PneumoniaMNIST)
- Lead-II / MIT-BIH 5-Class ECG Arrhythmia Detector (CNN-BiLSTM-Attention)
- Cardiology EHR Risk Classifier (RandomForest Z-Scored Cohort)
"""

import os
import json
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms as T
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from .utils.logger import get_logger
from .utils.metrics import compute_classification_metrics
from .vision.gradcam import GradCAM, encode_gradcam_artifacts
from .vision.radiology_tools import compute_consolidation_severity

logger = get_logger("PretrainedMedicalHub")

CURATED_PRETRAINED_CATALOG: List[Dict[str, Any]] = [
    {
        "repo_id": "Purino/mobilenetv2-chest-xray-pneumonia-256-fullframe-gray",
        "domain": "vision",
        "display_name": "MobileNetV2-256 Full-Frame Grayscale CXR",
        "architecture": "MobileNetV2 (TorchVision)",
        "input_spec": "256x256 Grayscale->RGB (ImageNet Norm)",
        "classes": ["Normal", "Pneumonia"],
        "license": "apache-2.0",
        "summary": "Trained on full-frame chest radiographs with grayscale normalization to avoid border-padding shortcuts.",
    },
    {
        "repo_id": "sdoerrich97/colorist_densenet121_pneumoniamnist_s71397589",
        "domain": "vision",
        "display_name": "DenseNet121 PneumoniaMNIST-224",
        "architecture": "DenseNet121 (TorchVision)",
        "input_spec": "224x224 Grayscale->RGB (Mean 0.5717, Std 0.177)",
        "classes": ["Normal", "Pneumonia"],
        "license": "mit",
        "summary": "DenseNet121 benchmark checkpoint trained directly on MedMNIST PneumoniaMNIST-224 pediatric radiographs.",
    },
    {
        "repo_id": "Purino/mobilenetv2-chest-xray-pneumonia",
        "domain": "vision",
        "display_name": "MobileNetV2-320 High-Res CXR",
        "architecture": "MobileNetV2 (TorchVision)",
        "input_spec": "320x320 RGB (ImageNet Norm)",
        "classes": ["Normal", "Pneumonia"],
        "license": "apache-2.0",
        "summary": "High-resolution MobileNetV2 chest X-ray classifier tuned for AP/PA pneumonia screening.",
    },
    {
        "repo_id": "dheerajthuvara/ecg-arrhythmia-detection",
        "domain": "ecg",
        "display_name": "CNN-BiLSTM-Attention MIT-BIH 5-Class ECG",
        "architecture": "3-Stage Conv1D + 2-Layer BiLSTM + Temporal Attention",
        "input_spec": "180-sample Z-score normalized cardiac beat",
        "classes": ["A (Atrial Premature)", "L (LBBB)", "N (Normal Sinus)", "R (RBBB)", "V (PVC)"],
        "license": "mit",
        "summary": "Hybrid spatiotemporal CNN-BiLSTM with attention trained on PhysioNet MIT-BIH 5-class arrhythmia beats.",
    },
    {
        "repo_id": "nexusbert/heart-disease-random-forest",
        "domain": "clinical",
        "display_name": "Heart Disease 300-Tree Random Forest",
        "architecture": "scikit-learn RandomForestClassifier (300 estimators)",
        "input_spec": "7 Z-scored Clinical Cardiology Features (Age, BP, Chol, FBS, Oldpeak, ExAng)",
        "classes": ["Low Cardiac Risk", "Coronary / Heart Disease Risk"],
        "license": "mit",
        "summary": "Public clinical heart disease Random Forest baseline over standardized hemodynamic & stress biomarkers.",
    },
]


class ECGCNNBiLSTM(nn.Module):
    """Exact PyTorch architecture for `dheerajthuvara/ecg-arrhythmia-detection`."""

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.3),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.3),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.3),
        )
        self.bilstm = nn.LSTM(128, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.attention = nn.Linear(256, 1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        h = self.cnn(x).transpose(1, 2)
        h, _ = self.bilstm(h)
        weights = torch.softmax(self.attention(h), dim=1)
        ctx = torch.sum(h * weights, dim=1)
        return self.classifier(ctx)


_MODEL_CACHE: Dict[str, Any] = {}


def get_curated_catalog() -> List[Dict[str, Any]]:
    return list(CURATED_PRETRAINED_CATALOG)


def search_hf_medical_models(query: str = "chest xray pneumonia", domain: Optional[str] = None, limit: int = 12) -> List[Dict[str, Any]]:
    """Queries live Hugging Face Hub API for public medical AI models matching `query`."""
    curated_ids = {item["repo_id"] for item in CURATED_PRETRAINED_CATALOG}
    api = HfApi()
    results: List[Dict[str, Any]] = []
    try:
        models_iter = api.list_models(search=query, sort="downloads", limit=limit)
        for m in models_iter:
            results.append({
                "repo_id": m.id,
                "downloads": int(getattr(m, "downloads", 0) or 0),
                "likes": int(getattr(m, "likes", 0) or 0),
                "pipeline_tag": getattr(m, "pipeline_tag", None) or "tabular/custom",
                "library_name": getattr(m, "library_name", None) or "pytorch/sklearn",
                "verified_in_studio": m.id in curated_ids,
                "domain_filter": domain or "all",
                "url": f"https://huggingface.co/{m.id}",
            })
    except Exception as e:
        logger.warning(f"Live HF search fallback ({e}); returning curated matches.")
        for item in CURATED_PRETRAINED_CATALOG:
            if query.lower() in item["repo_id"].lower() or query.lower() in item["summary"].lower():
                results.append({
                    "repo_id": item["repo_id"],
                    "downloads": 0,
                    "likes": 0,
                    "pipeline_tag": "image-classification" if item["domain"] == "vision" else "tabular-classification",
                    "library_name": item["architecture"],
                    "verified_in_studio": True,
                    "domain_filter": item["domain"],
                    "url": f"https://huggingface.co/{item['repo_id']}",
                })
    return results


def _load_hf_vision_model(repo_id: str, device: str = "cpu"):
    if repo_id in _MODEL_CACHE:
        return _MODEL_CACHE[repo_id]

    weights_file = hf_hub_download(repo_id, "model.safetensors")
    state = load_file(weights_file, device="cpu")

    if "densenet121" in repo_id.lower():
        net = models.densenet121(weights=None)
        net.classifier = nn.Linear(net.classifier.in_features, 2)
        net.load_state_dict(state)
        tf = T.Compose([
            T.Resize((224, 224)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize([0.5717, 0.5717, 0.5717], [0.177, 0.177, 0.177]),
        ])
        target_layer = net.features[-1]
    else:
        img_sz = 256 if "256" in repo_id else 320
        net = models.mobilenet_v2(weights=None)
        net.classifier[1] = nn.Linear(net.last_channel, 2)
        net.load_state_dict(state)
        tf = T.Compose([
            T.Resize((img_sz, img_sz)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        target_layer = net.features[-1]

    net.to(device)
    net.eval()
    bundle = {"model": net, "transform": tf, "target_layer": target_layer}
    _MODEL_CACHE[repo_id] = bundle
    return bundle


def _load_hf_ecg_model(repo_id: str = "dheerajthuvara/ecg-arrhythmia-detection", device: str = "cpu"):
    if repo_id in _MODEL_CACHE:
        return _MODEL_CACHE[repo_id]

    pt_path = hf_hub_download(repo_id, "models/best_model.pt")
    net = ECGCNNBiLSTM(num_classes=5)
    net.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
    net.to(device)
    net.eval()
    bundle = {"model": net, "classes": ["A", "L", "N", "R", "V"]}
    _MODEL_CACHE[repo_id] = bundle
    return bundle


def _load_hf_clinical_rf(repo_id: str = "nexusbert/heart-disease-random-forest"):
    if repo_id in _MODEL_CACHE:
        return _MODEL_CACHE[repo_id]
    rf_path = hf_hub_download(repo_id, "model.joblib")
    rf = joblib.load(rf_path)
    _MODEL_CACHE[repo_id] = rf
    return rf


def infer_pretrained_vision(
    pil_img: Image.Image,
    repo_id: str = "Purino/mobilenetv2-chest-xray-pneumonia-256-fullframe-gray",
    device: str = "cpu",
    colormap_name: str = "turbo",
) -> Dict[str, Any]:
    """Runs zero-shot pre-trained HF Chest X-Ray classification + Grad-CAM overlay."""
    bundle = _load_hf_vision_model(repo_id, device=device)
    net = bundle["model"]
    tensor = bundle["transform"](pil_img.convert("L")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = net(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))

    gradcam = GradCAM(net, bundle["target_layer"])
    heatmap, _, _ = gradcam.generate_heatmap(tensor, target_class=pred_idx)
    gradcam.remove_hooks()
    artifacts = encode_gradcam_artifacts(pil_img, heatmap, alpha=0.55, colormap_name=colormap_name)
    severity = compute_consolidation_severity(heatmap, threshold=0.40)

    classes = ["Normal Healthy Parenchyma", "Alveolar Consolidation / Pneumonia"]
    return {
        "repo_id": repo_id,
        "diagnosis": classes[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "normal_probability": round(float(probs[0]), 4),
        "pneumonia_probability": round(float(probs[1]), 4),
        "severity": severity,
        "gradcam_heatmap_base64": artifacts["heatmap_base64"],
        "gradcam_overlay_base64": artifacts["overlay_base64"],
        "hotspot": artifacts["hotspot"],
    }


def _resample_1d(sig: np.ndarray, target_len: int) -> np.ndarray:
    if len(sig) == target_len:
        return sig.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, len(sig))
    x_new = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_new, x_old, sig).astype(np.float32)


def infer_pretrained_ecg(
    signal: np.ndarray,
    repo_id: str = "dheerajthuvara/ecg-arrhythmia-detection",
    device: str = "cpu",
) -> Dict[str, Any]:
    """Runs pre-trained HF MIT-BIH 5-class ECG CNN-BiLSTM inference."""
    bundle = _load_hf_ecg_model(repo_id, device=device)
    net = bundle["model"]
    classes = bundle["classes"]

    s = np.asarray(signal, dtype=np.float32).ravel()
    s180 = _resample_1d(s, 180)
    s180 = (s180 - np.mean(s180)) / (np.std(s180) + 1e-8)

    with torch.no_grad():
        logits = net(torch.from_numpy(s180).unsqueeze(0).to(device))
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_code = classes[pred_idx]
    label_map = {
        "N": "Normal Sinus Beat (MIT-BIH N)",
        "A": "Atrial Premature Beat (MIT-BIH A)",
        "L": "Left Bundle Branch Block (MIT-BIH L)",
        "R": "Right Bundle Branch Block (MIT-BIH R)",
        "V": "Premature Ventricular Contraction (MIT-BIH V)",
    }
    return {
        "repo_id": repo_id,
        "predicted_code": pred_code,
        "diagnosis": label_map.get(pred_code, pred_code),
        "is_anomaly": bool(pred_code != "N"),
        "confidence": round(float(probs[pred_idx]), 4),
        "class_probabilities": {classes[i]: round(float(probs[i]), 4) for i in range(len(classes))},
    }


def _load_hf_raw_stats(metrics_path: str = "reports/clinical_metrics.json") -> Dict[str, Dict[str, float]]:
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            data = json.load(f)
            if "hf_raw_stats" in data:
                return data["hf_raw_stats"]
    return {
        "age": {"mean": 53.9, "std": 9.3, "median": 55.0},
        "resting_bp": {"mean": 132.4, "std": 17.5, "median": 130.0},
        "cholesterol": {"mean": 244.8, "std": 53.5, "median": 239.0},
        "fasting_bs": {"mean": 0.16, "std": 0.37, "median": 0.0},
        "oldpeak": {"mean": 0.90, "std": 1.07, "median": 0.6},
        "exercise_angina": {"mean": 0.38, "std": 0.48, "median": 0.0},
    }


def _prepare_nexusbert_features(df_or_dict, stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    if isinstance(df_or_dict, dict):
        df = pd.DataFrame([df_or_dict])
    else:
        df = df_or_dict.copy()

    def z(col: str) -> np.ndarray:
        s = stats[col]
        vals = pd.to_numeric(df[col], errors="coerce").fillna(s["median"]).to_numpy(dtype=np.float64)
        return (vals - s["mean"]) / max(1e-6, s["std"])

    ex = pd.to_numeric(df["exercise_angina"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    ex_no = (ex == 0).astype(np.float64)
    ex_yes = (ex == 1).astype(np.float64)
    # Standardize booleans relative to training proportion
    p_yes = stats["exercise_angina"]["mean"]
    std_yes = max(1e-6, stats["exercise_angina"]["std"])
    z_ex_yes = (ex_yes - p_yes) / std_yes
    z_ex_no = (ex_no - (1.0 - p_yes)) / std_yes

    return pd.DataFrame({
        "Age": z("age"),
        "Resting Blood Pressure": z("resting_bp"),
        "Cholesterol": z("cholesterol"),
        "Fasting Blood Sugar": z("fasting_bs"),
        "Oldpeak": z("oldpeak"),
        "Exercise Angina: No": z_ex_no,
        "Exercise Angina: Yes": z_ex_yes,
    })


def infer_pretrained_clinical(
    patient: Dict[str, Any],
    repo_id: str = "nexusbert/heart-disease-random-forest",
) -> Dict[str, Any]:
    """Runs pre-trained HF `nexusbert/heart-disease-random-forest` using leak-free training Z-score stats."""
    rf = _load_hf_clinical_rf(repo_id)
    stats = _load_hf_raw_stats()
    X_in = _prepare_nexusbert_features(patient, stats)
    probs = rf.predict_proba(X_in)[0]
    p_disease = float(probs[1])
    return {
        "repo_id": repo_id,
        "predicted_label": "Coronary / Cardiac Risk" if p_disease >= 0.5 else "Low Cardiac Risk",
        "risk_probability": round(p_disease, 4),
        "healthy_probability": round(float(probs[0]), 4),
    }


def benchmark_all_pretrained(
    base_dir: str = ".",
    device: str = "cpu",
    save_path: str = "reports/pretrained_benchmarks.json",
) -> Dict[str, Any]:
    """
    Empirically evaluates all 5 curated Hugging Face models against real held-out
    clinical test cohorts (no cherry-picking, full metrics recorded to JSON).
    """
    results: Dict[str, Any] = {"vision": {}, "ecg": {}, "clinical": {}}

    # 1. Vision Pre-Trained Models on real PneumoniaMNIST test split (624 images)
    manifest_csv = os.path.join(base_dir, "data", "external", "pneumoniamnist_manifest.csv")
    if os.path.exists(manifest_csv):
        df_m = pd.read_csv(manifest_csv)
        df_test = df_m[df_m["source_split"] == "test_real"]
        pils = [Image.open(os.path.join(base_dir, p)).convert("L") for p in df_test["relative_path"]]
        y_true = df_test["label"].astype(int).to_numpy()

        for repo_id in [
            "Purino/mobilenetv2-chest-xray-pneumonia-256-fullframe-gray",
            "sdoerrich97/colorist_densenet121_pneumoniamnist_s71397589",
            "Purino/mobilenetv2-chest-xray-pneumonia",
        ]:
            bundle = _load_hf_vision_model(repo_id, device=device)
            net = bundle["model"]
            tf = bundle["transform"]
            batch_probs = []
            with torch.no_grad():
                for i in range(0, len(pils), 64):
                    t_batch = torch.stack([tf(p) for p in pils[i:i + 64]]).to(device)
                    probs = F.softmax(net(t_batch), dim=1)[:, 1].cpu().numpy()
                    batch_probs.append(probs)
            y_prob = np.concatenate(batch_probs, axis=0)
            y_pred = (y_prob >= 0.5).astype(int)
            m = compute_classification_metrics(y_true, y_pred, y_prob)
            results["vision"][repo_id] = {
                "cohort": f"Real PneumoniaMNIST Test Split (n={len(y_true)})",
                **m,
            }
            logger.info(f"[HF Vision Benchmark] {repo_id}: Acc={m['accuracy']*100:.2f}%, ROC-AUC={m['roc_auc']:.4f}")

    # 2. ECG Pre-Trained Model on MIT-BIH 10k Holdout Test Split
    mitbih_npz = os.path.join(base_dir, "data", "external", "mitbih_10k.npz")
    if os.path.exists(mitbih_npz):
        from sklearn.model_selection import train_test_split
        loaded = np.load(mitbih_npz, allow_pickle=True)
        X, y = loaded["X"], loaded["y"]
        _, X_temp, _, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
        _, X_te, _, y_te = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

        ecg_bundle = _load_hf_ecg_model("dheerajthuvara/ecg-arrhythmia-detection", device=device)
        net = ecg_bundle["model"]
        probs_all = []
        with torch.no_grad():
            for i in range(0, len(X_te), 128):
                xb = torch.from_numpy(X_te[i:i + 128]).to(device)
                probs_all.append(F.softmax(net(xb), dim=1).cpu().numpy())
        probs_mat = np.concatenate(probs_all, axis=0)
        y_pred = np.argmax(probs_mat, axis=1)
        results["ecg"]["dheerajthuvara/ecg-arrhythmia-detection"] = {
            "cohort": f"PhysioNet MIT-BIH Stratified Holdout (n={len(y_te)})",
            "multiclass_accuracy": round(float(accuracy_score(y_te, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_te, y_pred, average="macro")), 4),
            "multiclass_roc_auc": round(float(roc_auc_score(y_te, probs_mat, multi_class="ovr")), 4),
        }
        logger.info(
            f"[HF ECG Benchmark] dheerajthuvara/ecg-arrhythmia-detection: "
            f"Acc={results['ecg']['dheerajthuvara/ecg-arrhythmia-detection']['multiclass_accuracy']*100:.2f}%"
        )

    # 3. Clinical Pre-Trained Model on Cardiology EHR Holdout Test Split (leak-free!)
    cardio_csv = os.path.join(base_dir, "data", "external", "cardiology_triage.csv")
    if os.path.exists(cardio_csv):
        from sklearn.model_selection import train_test_split
        from .clinical.features import RAW_CLINICAL_FEATURES
        df_c = pd.read_csv(cardio_csv)
        _, X_te_raw, _, y_te_c = train_test_split(
            df_c[RAW_CLINICAL_FEATURES], df_c["target"].astype(int).to_numpy(),
            test_size=0.20, stratify=df_c["target"].astype(int).to_numpy(), random_state=42
        )
        rf = _load_hf_clinical_rf("nexusbert/heart-disease-random-forest")
        stats = _load_hf_raw_stats(os.path.join(base_dir, "reports", "clinical_metrics.json"))
        X_hf = _prepare_nexusbert_features(X_te_raw, stats)
        y_prob_c = rf.predict_proba(X_hf)[:, 1]
        y_pred_c = (y_prob_c >= 0.5).astype(int)
        m_c = compute_classification_metrics(y_te_c, y_pred_c, y_prob_c)
        results["clinical"]["nexusbert/heart-disease-random-forest"] = {
            "cohort": f"Real Deduplicated Cardiology Test Split (n={len(y_te_c)})",
            **m_c,
        }
        logger.info(
            f"[HF Clinical Benchmark] nexusbert/heart-disease-random-forest: "
            f"Acc={m_c['accuracy']*100:.2f}%, ROC-AUC={m_c['roc_auc']:.4f}"
        )

    out_full = os.path.join(base_dir, save_path)
    os.makedirs(os.path.dirname(out_full), exist_ok=True)
    with open(out_full, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved empirical Hugging Face pre-trained benchmarks to {out_full}")
    return results
