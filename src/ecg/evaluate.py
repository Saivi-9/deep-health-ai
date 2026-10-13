"""
ECG Anomaly Detection Evaluation Pipeline
-----------------------------------------
Evaluates autoencoder reconstruction performance on mixed test sets:
- Computes Arrhythmia Sensitivity (Recall), Specificity, AUROC, F1-Score
- Generates side-by-side normal vs ectopic beat reconstruction plots
- Visualizes error distributions & decision boundary
"""

import os
from typing import Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..utils.metrics import compute_classification_metrics
from ..utils.plotting import (
    plot_ecg_reconstructions,
    plot_anomaly_score_distribution,
    plot_roc_pr_curves
)
from ..utils.logger import get_logger
from .anomaly_detector import ECGAnomalyDetector

logger = get_logger("ECGEvaluator")


def evaluate_ecg_anomaly_detection(
    detector: ECGAnomalyDetector,
    test_loader: DataLoader,
    test_labels: np.ndarray,
    figures_dir: str = "reports/figures"
) -> Dict[str, Any]:
    """
    Evaluates autoencoder-based anomaly detection on test set.
    """
    os.makedirs(figures_dir, exist_ok=True)
    errors, originals, reconstructed = detector.compute_reconstruction_errors(test_loader)

    if detector.threshold is None:
        raise ValueError("Detector threshold must be calibrated before evaluation.")

    threshold = detector.threshold
    predictions = (errors > threshold).astype(int)

    # Note: errors act as the continuous anomaly score for ROC-AUC
    metrics = compute_classification_metrics(test_labels, predictions, errors)

    logger.info("=== ECG Anomaly Detection Test Results ===")
    logger.info(f"Anomaly Detection Accuracy: {metrics['accuracy']*100:.2f}%")
    logger.info(f"Arrhythmia Sensitivity:     {metrics['sensitivity_recall']*100:.2f}%")
    logger.info(f"Sinus Rhythm Specificity:   {metrics['specificity']*100:.2f}%")
    logger.info(f"Precision (PPV):            {metrics['precision']*100:.2f}%")
    logger.info(f"F1-Score:                   {metrics['f1_score']:.4f}")
    if metrics.get("roc_auc") is not None:
        logger.info(f"Reconstruction ROC-AUC:     {metrics['roc_auc']:.4f}")

    # Generate figures
    recon_fig_path = os.path.join(figures_dir, "ecg_reconstruction_examples.png")
    plot_ecg_reconstructions(originals, reconstructed, test_labels, errors, threshold, recon_fig_path)

    dist_fig_path = os.path.join(figures_dir, "ecg_anomaly_histogram_and_threshold.png")
    plot_anomaly_score_distribution(errors, test_labels, threshold, dist_fig_path)

    if metrics.get("roc_auc") is not None:
        roc_fig_path = os.path.join(figures_dir, "ecg_roc_curve.png")
        plot_roc_pr_curves(test_labels, errors, roc_fig_path)

    return metrics
