"""
Deep Health AI - Batch Cohort Processor & Statistical Epidemiological Analyzer
-------------------------------------------------------------------------------
Evaluates hospital patient batches across Chest Radiographs, Lead-II ECGs, and
Cardiology EHR encounters with multi-model confidence, uncertainty & referral triage.

Usage:
    python batch_processor.py --modality vision --input-dir data/vision/pneumonia --output-csv reports/batch_vision_cohort.csv
    python batch_processor.py --modality ecg --input-file data/ecg/ecg_test.npy --output-csv reports/batch_ecg_cohort.csv
    python batch_processor.py --modality clinical --input-file data/external/cardiology_triage.csv --output-csv reports/batch_clinical_cohort.csv
"""

import os
import glob
import json
import argparse
from typing import List, Dict, Any

import numpy as np
from PIL import Image
import pandas as pd
import joblib
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.vision.model import ResidualChestCNN
from src.vision.gradcam import GradCAM
from src.vision.uncertainty import MonteCarloDropoutEstimator
from src.vision.radiology_tools import compute_consolidation_severity, extract_radiomic_features
from src.ecg.model import Conv1dAutoencoder, ECGResNet1D
from src.ecg.anomaly_detector import ECGAnomalyDetector
from src.ecg.telemetry import analyze_ecg_telemetry
from src.clinical.features import engineer_clinical_features


def process_vision_cohort(
    image_paths: List[str],
    weights_path: str = "saved_models/best_chest_cnn.pt",
    device: str = "cpu"
) -> Dict[str, Any]:
    """Processes a batch of radiograph files and computes population metrics."""
    model = ResidualChestCNN(in_channels=1, num_classes=2)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=torch.device(device), weights_only=True))
    model.to(device)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    estimator = MonteCarloDropoutEstimator(model, n_samples=12, device=device)
    records = []

    for path in image_paths:
        try:
            img = Image.open(path).convert("L")
            inp = tf(img).unsqueeze(0).to(device)
            rad = extract_radiomic_features(np.asarray(img.resize((128, 128)), dtype=np.float32))

            with torch.no_grad():
                logits = model(inp)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))

            uncert = estimator.estimate_uncertainty(inp)

            gradcam = GradCAM(model, model.layer4)
            hm, _, _ = gradcam.generate_heatmap(inp, target_class=pred_idx)
            gradcam.remove_hooks()
            sev = compute_consolidation_severity(hm, threshold=0.40)

            records.append({
                "file_path": path,
                "filename": os.path.basename(path),
                "predicted_diagnosis": "Pneumonia" if pred_idx == 1 else "Normal",
                "confidence": round(float(probs[pred_idx]), 4),
                "normal_prob": round(float(probs[0]), 4),
                "pneumonia_prob": round(float(probs[1]), 4),
                "predictive_variance": round(uncert["predictive_variance"], 6),
                "normalized_entropy": round(uncert["normalized_entropy"], 4),
                "referral_mandatory": uncert["human_referral_mandatory"],
                "radiological_grade": sev["radiological_grade"],
                "infiltrate_percentage": sev["infiltrate_burden_percentage"],
                "triage_urgency": sev["clinical_triage_urgency"],
                "consolidation_opacity_fraction": rad["consolidation_opacity_fraction"],
                "left_right_asymmetry": rad["left_right_asymmetry"],
            })
        except Exception as e:
            records.append({"file_path": path, "error": str(e)})

    df = pd.DataFrame(records)
    valid_df = df[df["predicted_diagnosis"].notnull()]

    total_scans = len(valid_df)
    pneumonia_count = int((valid_df["predicted_diagnosis"] == "Pneumonia").sum())
    referral_count = int(valid_df["referral_mandatory"].sum())

    return {
        "modality": "Thoracic Radiography",
        "total_scans_evaluated": total_scans,
        "pathology_prevalence_pct": round((pneumonia_count / max(1, total_scans)) * 100, 2),
        "mean_confidence": round(float(valid_df["confidence"].mean()), 4),
        "mean_epistemic_variance": round(float(valid_df["predictive_variance"].mean()), 6),
        "secondary_radiologist_referral_rate_pct": round((referral_count / max(1, total_scans)) * 100, 2),
        "results_dataframe": df,
    }


def process_ecg_cohort(
    signals: np.ndarray,
    weights_path: str = "saved_models/best_ecg_autoencoder.pt",
    metrics_path: str = "reports/ecg_metrics.json",
    device: str = "cpu"
) -> Dict[str, Any]:
    """Processes a batch of 1D ECG waveforms and computes population + 5-class rhythm breakdown."""
    model = Conv1dAutoencoder(in_channels=1)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=torch.device(device), weights_only=True))
    model.to(device)
    model.eval()

    resnet_path = "saved_models/ecg_resnet1d.pt"
    resnet1d = None
    rhythm_classes = ["NSR", "PVC", "AFib", "ST-Elevation", "LBBB"]
    if os.path.exists(resnet_path):
        resnet1d = ECGResNet1D(in_channels=1, num_classes=5)
        resnet1d.load_state_dict(torch.load(resnet_path, map_location=torch.device(device), weights_only=True))
        resnet1d.to(device)
        resnet1d.eval()

    threshold = 0.01037
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                data = json.load(f)
                threshold = float(data.get("threshold", 0.01037))
        except Exception:
            pass

    detector = ECGAnomalyDetector(model=model, threshold=threshold, device=device)
    records = []

    for idx, signal in enumerate(signals):
        sig_256 = signal[:256]
        res = detector.predict(sig_256)
        tele = analyze_ecg_telemetry(sig_256, sampling_rate=250)

        subtype = "Normal Sinus Rhythm" if not res["is_anomaly"] else "Unspecified Arrhythmia"
        if resnet1d is not None:
            with torch.no_grad():
                t_sig = torch.from_numpy(sig_256.astype(np.float32)).view(1, 1, -1).to(device)
                p_sub = F.softmax(resnet1d(t_sig), dim=1).cpu().numpy()[0]
                subtype = rhythm_classes[int(np.argmax(p_sub))]

        records.append({
            "beat_index": idx,
            "diagnosis": res["diagnosis"],
            "rhythm_subtype": subtype,
            "is_anomaly": res["is_anomaly"],
            "reconstruction_mse": round(res["reconstruction_error"], 6),
            "threshold": threshold,
            "heart_rate_bpm": tele["heart_rate_bpm"],
            "qrs_duration_ms": tele["qrs_duration_ms"],
            "qtc_ms": tele["qtc_interval_ms"],
            "st_segment_mv": tele["st_segment_mv"],
            "wide_qrs_flag": tele["clinical_flags"]["wide_qrs_ventricular_conduction_delay"],
        })

    df = pd.DataFrame(records)
    total_beats = len(df)
    anomaly_count = int(df["is_anomaly"].sum())

    return {
        "modality": "Lead-II Electrocardiogram",
        "total_beats_evaluated": total_beats,
        "arrhythmia_burden_pct": round((anomaly_count / max(1, total_beats)) * 100, 2),
        "mean_reconstruction_mse": round(float(df["reconstruction_mse"].mean()), 6),
        "mean_heart_rate_bpm": round(float(df["heart_rate_bpm"].mean()), 1),
        "mean_qrs_duration_ms": round(float(df["qrs_duration_ms"].mean()), 1),
        "mean_qtc_ms": round(float(df["qtc_ms"].mean()), 1),
        "results_dataframe": df,
    }


def process_clinical_cohort(
    df_patients: pd.DataFrame,
    model_path: str = "saved_models/best_clinical_triage.joblib",
) -> Dict[str, Any]:
    """Runs batch Cardiology EHR risk stratification across patient rows."""
    bundle = joblib.load(model_path)
    pipe = bundle["pipeline"]
    X_feat = engineer_clinical_features(df_patients)
    probs = pipe.predict_proba(X_feat)[:, 1]
    preds = (probs >= 0.5).astype(int)

    df_out = df_patients.copy().reset_index(drop=True)
    df_out["cad_risk_probability"] = np.round(probs, 4)
    df_out["predicted_risk"] = np.where(preds == 1, "High Cardiac Risk", "Low Cardiac Risk")
    df_out["triage_tier"] = np.where(
        probs >= 0.78, "CRITICAL",
        np.where(probs >= 0.55, "HIGH", np.where(probs >= 0.35, "MODERATE", "LOW"))
    )

    total = len(df_out)
    high_cnt = int((probs >= 0.55).sum())
    return {
        "modality": "Cardiology EHR Risk Stratification",
        "total_patients_evaluated": total,
        "high_risk_prevalence_pct": round((high_cnt / max(1, total)) * 100, 2),
        "mean_cad_probability": round(float(np.mean(probs)), 4),
        "best_model": bundle.get("best_model_name", "StackingEnsemble"),
        "results_dataframe": df_out,
    }


def main():
    parser = argparse.ArgumentParser(description="Deep Health AI Batch Cohort Processing Tool")
    parser.add_argument("--modality", choices=["vision", "ecg", "clinical"], required=True, help="Modality to process")
    parser.add_argument("--input-dir", type=str, help="Input directory containing PNG radiographs (for vision)")
    parser.add_argument("--input-file", type=str, help="Input .npy array (ecg) or .csv file (clinical)")
    parser.add_argument("--output-csv", type=str, default="reports/batch_cohort_results.csv", help="Output CSV path")
    parser.add_argument("--device", type=str, default="cpu", help="Compute hardware (cpu/cuda)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    if args.modality == "vision":
        if not args.input_dir:
            raise ValueError("--input-dir is required when --modality vision is selected.")
        image_files = sorted(glob.glob(os.path.join(args.input_dir, "*.png")) + glob.glob(os.path.join(args.input_dir, "*.jpg")))
        if not image_files:
            raise FileNotFoundError(f"No PNG or JPG images found in {args.input_dir}")
        summary = process_vision_cohort(image_files, device=args.device)
    elif args.modality == "ecg":
        if not args.input_file:
            raise ValueError("--input-file is required when --modality ecg is selected.")
        signals = np.load(args.input_file)
        summary = process_ecg_cohort(signals, device=args.device)
    else:
        if not args.input_file:
            raise ValueError("--input-file is required when --modality clinical is selected.")
        df_c = pd.read_csv(args.input_file)
        summary = process_clinical_cohort(df_c)

    df = summary.pop("results_dataframe")
    df.to_csv(args.output_csv, index=False)
    print("\n" + "=" * 55)
    print(f"COHORT SUMMARY ({summary['modality']})")
    print("=" * 55)
    for k, v in summary.items():
        print(f"{k.replace('_', ' ').title():<40}: {v}")
    print(f"Results saved to: {args.output_csv}\n")


if __name__ == "__main__":
    main()
