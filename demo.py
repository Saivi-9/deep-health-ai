"""
Deep Health AI v2.0 - Full Multi-Modal Clinical Benchmark & Pipeline Runner
-------------------------------------------------------------------------
Orchestrates all 5 stages end-to-end:
1. Real external HF cohorts + scaled multi-regime dataset generation
2. Multi-Architecture Vision Suite (ResidualChestCNN, ResNet18, MobileNetV2, RadiomicLightGBM)
3. Multi-Architecture ECG Suite (Conv1dAutoencoder, ECGResNet1D, LightGBM, HistGB + MIT-BIH 10k)
4. Clinical Cardiology EHR Risk Suite (LightGBM, HistGB, RF, ExtraTrees, StackingEnsemble)
5. Hugging Face Pre-Trained Medical Hub Benchmarks + Unified `reports/metrics_summary.json`
"""

import os
import json
import torch

from data.fetch_external import fetch_all_external_datasets
from data.generate_datasets import generate_all_datasets
from train_vision import train_and_benchmark_vision
from train_ecg import train_and_benchmark_ecg
from src.clinical.trainer import train_clinical_triage_suite
from src.pretrained_hub import benchmark_all_pretrained
from src.utils.logger import get_logger

logger = get_logger("DeepHealthDemo")


def run_full_pipeline():
    logger.info("================================================================")
    logger.info("STARTING DEEP HEALTH AI v2.0 MULTI-MODAL PIPELINE")
    logger.info("================================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("reports/figures", exist_ok=True)

    logger.info("[Step 1/5] Verifying External & Synthetic Clinical Cohorts...")
    fetch_all_external_datasets(base_dir=".")
    generate_all_datasets(base_dir=".")

    logger.info("[Step 2/5] Running Multi-Architecture Chest X-Ray Suite...")
    vision_report = train_and_benchmark_vision(epochs=5, lr=0.0018, transfer_epochs=4, batch_size=64)

    logger.info("[Step 3/5] Running Multi-Architecture 1D ECG Suite (Lead-II + MIT-BIH)...")
    ecg_report = train_and_benchmark_ecg(epochs=24, lr=0.002, batch_size=32)

    logger.info("[Step 4/5] Running Real Cardiology EHR 5-Model Tournament...")
    clinical_report = train_clinical_triage_suite()

    logger.info("[Step 5/5] Benchmarking Hugging Face Pre-Trained Medical Models...")
    pretrained_report = benchmark_all_pretrained(base_dir=".", device=device)

    summary = {
        "version": "2.0.0",
        "chest_xray_vision": {
            "model": "ResidualChestCNN + SpatialAttention + GradCAM (plus ResNet18, MobileNetV2, Radiomic-LGBM)",
            "accuracy": vision_report["accuracy"],
            "sensitivity": vision_report["sensitivity_recall"],
            "specificity": vision_report["specificity"],
            "precision": vision_report["precision"],
            "f1_score": vision_report["f1_score"],
            "roc_auc": vision_report.get("roc_auc"),
            "pr_auc": vision_report.get("pr_auc"),
            "confusion_matrix": vision_report["confusion_matrix"],
            "dataset_summary": vision_report.get("dataset_summary"),
            "models_comparison": vision_report.get("models_comparison"),
        },
        "ecg_cardiac_anomaly": {
            "model": "1D-ConvAutoencoder + Supervised 5-Class ECGResNet1D + ECG-LGBM + MIT-BIH ResNet1D",
            "calibrated_threshold": ecg_report["threshold"],
            "accuracy": ecg_report["metrics"]["accuracy"],
            "arrhythmia_sensitivity": ecg_report["metrics"]["sensitivity_recall"],
            "sinus_specificity": ecg_report["metrics"]["specificity"],
            "precision": ecg_report["metrics"]["precision"],
            "f1_score": ecg_report["metrics"]["f1_score"],
            "reconstruction_roc_auc": ecg_report["metrics"].get("roc_auc"),
            "confusion_matrix": ecg_report["metrics"]["confusion_matrix"],
            "models_comparison": ecg_report.get("models_comparison"),
            "mitbih_benchmark": ecg_report.get("mitbih_benchmark"),
        },
        "clinical_cardiology_ehr": clinical_report,
        "pretrained_huggingface_hub": pretrained_report,
    }

    summary_file = "reports/metrics_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("================================================================")
    logger.info(f"DEEP HEALTH AI v2.0 COMPLETE! Consolidated report: {summary_file}")
    logger.info("================================================================")
    return summary


if __name__ == "__main__":
    run_full_pipeline()
