"""
Deep Health AI - Batch Cohort Processor & Statistical Epidemiological Analyzer
-------------------------------------------------------------------------------
Evaluates hospital patient batches across chest radiographs or ECG signals:
- Multi-threaded cohort inference
- Aggregate epidemiology & pathology prevalence metrics
- Mean epistemic uncertainty and secondary referral triage rates
- Automated CSV output export

Usage:
    python batch_processor.py --modality vision --input-dir data/vision/pneumonia --output-csv reports/batch_vision_cohort.csv
    python batch_processor.py --modality ecg --input-file data/ecg/ecg_test.npy --output-csv reports/batch_ecg_cohort.csv
"""

import os
import glob
import json
import argparse
from typing import List, Dict, Any

import numpy as np
from PIL import Image
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.vision.model import ResidualChestCNN
from src.vision.gradcam import GradCAM
from src.vision.uncertainty import MonteCarloDropoutEstimator
from src.vision.radiology_tools import compute_consolidation_severity
from src.ecg.model import Conv1dAutoencoder
from src.ecg.anomaly_detector import ECGAnomalyDetector
from src.ecg.telemetry import analyze_ecg_telemetry


def process_vision_cohort(
    image_paths: List[str],
    weights_path: str = "saved_models/best_chest_cnn.pt",
    device: str = "cpu"
) -> Dict[str, Any]:
    """Processes a batch of radiograph files and computes population metrics."""
    model = ResidualChestCNN(in_channels=1, num_classes=2)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    model.to(device)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    estimator = MonteCarloDropoutEstimator(model, n_samples=15, device=device)
    records = []

    for path in image_paths:
        try:
            img = Image.open(path).convert("L")
            inp = tf(img).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(inp)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))

            uncert = estimator.estimate_uncertainty(inp)

            # Grad-CAM severity
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
                "triage_urgency": sev["clinical_triage_urgency"]
            })
        except Exception as e:
            records.append({"file_path": path, "error": str(e)})

    df = pd.DataFrame(records)
    valid_df = df[df["predicted_diagnosis"].notnull()]

    total_scans = len(valid_df)
    pneumonia_count = int((valid_df["predicted_diagnosis"] == "Pneumonia").sum())
    referral_count = int(valid_df["referral_mandatory"].sum())

    summary = {
        "modality": "Thoracic Radiography",
        "total_scans_evaluated": total_scans,
        "pathology_prevalence_pct": round((pneumonia_count / max(1, total_scans)) * 100, 2),
        "mean_confidence": round(float(valid_df["confidence"].mean()), 4),
        "mean_epistemic_variance": round(float(valid_df["predictive_variance"].mean()), 6),
        "secondary_radiologist_referral_rate_pct": round((referral_count / max(1, total_scans)) * 100, 2),
        "results_dataframe": df
    }
    return summary


def process_ecg_cohort(
    signals: np.ndarray,
    weights_path: str = "saved_models/best_ecg_autoencoder.pt",
    metrics_path: str = "reports/ecg_metrics.json",
    device: str = "cpu"
) -> Dict[str, Any]:
    """Processes a batch of 1D ECG waveforms and computes population metrics."""
    model = Conv1dAutoencoder(in_channels=1)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    model.to(device)
    model.eval()

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

        records.append({
            "beat_index": idx,
            "diagnosis": res["diagnosis"],
            "is_anomaly": res["is_anomaly"],
            "reconstruction_mse": round(res["reconstruction_error"], 6),
            "threshold": threshold,
            "heart_rate_bpm": tele["heart_rate_bpm"],
            "qrs_duration_ms": tele["qrs_duration_ms"],
            "qtc_ms": tele["qtc_interval_ms"],
            "st_segment_mv": tele["st_segment_mv"],
            "wide_qrs_flag": tele["clinical_flags"]["wide_qrs_ventricular_conduction_delay"]
        })

    df = pd.DataFrame(records)
    total_beats = len(df)
    anomaly_count = int(df["is_anomaly"].sum())

    summary = {
        "modality": "Lead-II Electrocardiogram",
        "total_beats_evaluated": total_beats,
        "arrhythmia_burden_pct": round((anomaly_count / max(1, total_beats)) * 100, 2),
        "mean_reconstruction_mse": round(float(df["reconstruction_mse"].mean()), 6),
        "mean_heart_rate_bpm": round(float(df["heart_rate_bpm"].mean()), 1),
        "mean_qrs_duration_ms": round(float(df["qrs_duration_ms"].mean()), 1),
        "mean_qtc_ms": round(float(df["qtc_ms"].mean()), 1),
        "results_dataframe": df
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Deep Health AI Batch Cohort Processing Tool")
    parser.add_argument("--modality", choices=["vision", "ecg"], required=True, help="Modality to process")
    parser.add_argument("--input-dir", type=str, help="Input directory containing PNG radiographs (for vision)")
    parser.add_argument("--input-file", type=str, help="Input .npy array file (for ecg)")
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

        print(f"🔬 Processing {len(image_files)} chest radiographs in cohort...")
        summary = process_vision_cohort(image_files, device=args.device)
        df = summary.pop("results_dataframe")
        df.to_csv(args.output_csv, index=False)

        print("\n" + "=" * 55)
        print("COHORT EPIDEMIOLOGICAL SUMMARY")
        print("=" * 55)
        for k, v in summary.items():
            print(f"{k.replace('_', ' ').title():<40}: {v}")
        print(f"Results saved to: {args.output_csv}\n")

    else:
        if not args.input_file:
            raise ValueError("--input-file is required when --modality ecg is selected.")
        signals = np.load(args.input_file)
        print(f"💓 Processing {len(signals)} ECG cardiac cycles in cohort...")
        summary = process_ecg_cohort(signals, device=args.device)
        df = summary.pop("results_dataframe")
        df.to_csv(args.output_csv, index=False)

        print("\n" + "=" * 55)
        print("ELECTROPHYSIOLOGICAL COHORT SUMMARY")
        print("=" * 55)
        for k, v in summary.items():
            print(f"{k.replace('_', ' ').title():<40}: {v}")
        print(f"Results saved to: {args.output_csv}\n")


if __name__ == "__main__":
    main()
