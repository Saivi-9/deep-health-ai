"""
Vision Model Evaluation Pipeline & Interpretability Reporting
-------------------------------------------------------------
Evaluates the trained Residual Chest CNN on an isolated test set:
- Computes sensitivity, specificity, AUROC, AUPRC, F1
- Generates confusion matrix and ROC/PR curve plots
- Produces Grad-CAM explainability overlays on positive/negative test cases
"""

import os
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..utils.metrics import compute_classification_metrics
from ..utils.plotting import plot_confusion_matrix, plot_roc_pr_curves, plot_gradcam_overlay
from ..utils.logger import get_logger
from .gradcam import GradCAM

logger = get_logger("VisionEvaluator")


def evaluate_vision_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    figures_dir: str = "reports/figures",
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Evaluates vision model on test set and generates diagnostic plots.
    """
    device_obj = torch.device(device)
    model.to(device_obj)
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []
    all_images = []
    all_paths = []

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device_obj)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())  # Probability of positive class (Pneumonia)
            all_images.append(images.cpu())
            all_paths.extend(paths)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    metrics = compute_classification_metrics(all_labels, all_preds, all_probs)
    logger.info("=== Vision Test Set Diagnostic Results ===")
    logger.info(f"Accuracy:            {metrics['accuracy']*100:.2f}%")
    logger.info(f"Sensitivity (Recall): {metrics['sensitivity_recall']*100:.2f}%")
    logger.info(f"Specificity:         {metrics['specificity']*100:.2f}%")
    logger.info(f"Precision:           {metrics['precision']*100:.2f}%")
    logger.info(f"F1-Score:            {metrics['f1_score']:.4f}")
    if metrics.get("roc_auc") is not None:
        logger.info(f"ROC-AUC:             {metrics['roc_auc']:.4f}")
        logger.info(f"PR-AUC:              {metrics['pr_auc']:.4f}")

    # Generate evaluation figures
    os.makedirs(figures_dir, exist_ok=True)
    cm_path = os.path.join(figures_dir, "vision_confusion_matrix.png")
    plot_confusion_matrix(all_labels, all_preds, class_names, cm_path)

    if metrics.get("roc_auc") is not None:
        roc_path = os.path.join(figures_dir, "vision_roc_pr_curves.png")
        plot_roc_pr_curves(all_labels, all_probs, roc_path)

    # Generate Grad-CAM Explainability overlays on test samples
    try:
        gradcam = GradCAM(model, model.layer4)
        # Select one true positive and one true negative
        tp_indices = np.where((all_labels == 1) & (all_preds == 1))[0]
        tn_indices = np.where((all_labels == 0) & (all_preds == 0))[0]

        target_idx = tp_indices[0] if len(tp_indices) > 0 else 0
        img_tensor = torch.cat(all_images, dim=0)[target_idx:target_idx+1].to(device_obj)

        heatmap, pred_c, conf = gradcam.generate_heatmap(img_tensor, target_class=1)
        raw_img = img_tensor.squeeze().cpu().numpy()

        gradcam_path = os.path.join(figures_dir, "vision_gradcam_explanations.png")
        plot_gradcam_overlay(
            image=raw_img,
            heatmap=heatmap,
            pred_class=class_names[pred_c],
            true_class=class_names[all_labels[target_idx]],
            confidence=conf,
            save_path=gradcam_path
        )
        gradcam.remove_hooks()
        logger.info(f"Grad-CAM visual explainability saved to {gradcam_path}")
    except Exception as e:
        logger.warning(f"Grad-CAM generation encountered an issue: {e}")

    return metrics
