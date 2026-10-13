"""
High-Resolution Visualization & Diagnostic Plotting Utilities
-------------------------------------------------------------
Generates publication-ready figures for model evaluation:
- Training & Validation loss/accuracy dynamics
- Confusion matrices
- ROC & Precision-Recall curves
- Grad-CAM heatmap overlays on medical imaging
- ECG autoencoder reconstruction comparisons & error histograms
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# Style defaults
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


def plot_training_history(history: Dict[str, List[float]], save_path: str, title: str = "Training History"):
    """Plots training and validation loss and metric curves over epochs."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss plot
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], "o-", color="#1f77b4", label="Train Loss", linewidth=2)
    if "val_loss" in history and len(history["val_loss"]) > 0:
        axes[0].plot(epochs, history["val_loss"], "s--", color="#ff7f0e", label="Val Loss", linewidth=2)
    axes[0].set_title(f"{title} - Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Metric plot (Accuracy or Score)
    metric_key = "val_acc" if "val_acc" in history else ("train_acc" if "train_acc" in history else None)
    if metric_key:
        if "train_acc" in history:
            axes[1].plot(epochs, history["train_acc"], "o-", color="#2ca02c", label="Train Accuracy", linewidth=2)
        if "val_acc" in history:
            axes[1].plot(epochs, history["val_acc"], "s--", color="#d62728", label="Val Accuracy", linewidth=2)
        axes[1].set_title(f"{title} - Accuracy", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].legend(loc="lower right")
        axes[1].grid(True, linestyle="--", alpha=0.6)
    else:
        # If autoencoder without accuracy, plot learning rate or secondary loss
        axes[1].plot(epochs, history.get("lr", [1e-3] * len(epochs)), "d-", color="#9467bd", label="Learning Rate")
        axes[1].set_title(f"{title} - Optimizer Dynamics", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("LR")
        axes[1].set_yscale("log")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], save_path: str):
    """Generates an annotated confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel="True Clinical Diagnosis",
           xlabel="Model Prediction",
           title="Diagnostic Confusion Matrix")

    # Annotate counts and percentages
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_pr_curves(y_true: np.ndarray, y_prob: np.ndarray, save_path: str):
    """Plots Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC Curve
    axes[0].plot(fpr, tpr, color="#2b5c8f", lw=2.5, label=f"ROC (AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Sensitivity)")
    axes[0].set_title("ROC Curve", fontsize=12, fontweight="bold")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # PR Curve
    axes[1].plot(rec, prec, color="#d95f02", lw=2.5, label=f"PR (AUC = {pr_auc:.3f})")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall (Sensitivity)")
    axes[1].set_ylabel("Precision (PPV)")
    axes[1].set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_gradcam_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    pred_class: str,
    true_class: str,
    confidence: float,
    save_path: str
):
    """
    Renders side-by-side: Original radiograph, Grad-CAM attention heatmap,
    and Alpha-blended clinical diagnostic overlay.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Normalize image to [0, 1]
    img_disp = (image - image.min()) / (image.max() - image.min() + 1e-8)

    # 1. Original
    axes[0].imshow(img_disp, cmap="gray")
    axes[0].set_title("Original Chest Radiograph", fontweight="bold")
    axes[0].axis("off")

    # 2. Grad-CAM Activation Heatmap
    im_hm = axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Class Activation Map", fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im_hm, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Superimposed Overlay
    axes[2].imshow(img_disp, cmap="gray")
    axes[2].imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    color = "darkgreen" if pred_class == true_class else "red"
    axes[2].set_title(
        f"Explainable Diagnostic Overlay\nPred: {pred_class} ({confidence*100:.1f}%) | True: {true_class}",
        fontweight="bold", color=color
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_ecg_reconstructions(
    originals: np.ndarray,
    reconstructed: np.ndarray,
    labels: np.ndarray,
    errors: np.ndarray,
    threshold: float,
    save_path: str
):
    """Visualizes normal vs anomalous ECG beat reconstructions and residual error peaks."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    norm_idx = np.where(labels == 0)[0][0]
    anom_idx = np.where(labels == 1)[0][0]

    # Row 1: Normal beat
    axes[0, 0].plot(originals[norm_idx], color="#1f77b4", label="Original Normal Beat", lw=2)
    axes[0, 0].plot(reconstructed[norm_idx], color="#2ca02c", linestyle="--", label="Autoencoder Reconstruction", lw=2)
    axes[0, 0].set_title(f"Normal Sinus Rhythm (Reconstruction MSE: {errors[norm_idx]:.4f} <= Threshold {threshold:.4f})", fontweight="bold")
    axes[0, 0].set_ylabel("Normalized Voltage (mV)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)

    residual_norm = (originals[norm_idx] - reconstructed[norm_idx]) ** 2
    axes[0, 1].plot(residual_norm, color="#7f7f7f", label="Squared Error Residual", lw=1.5)
    axes[0, 1].axhline(y=threshold, color="red", linestyle=":", label=f"Detection Threshold ({threshold:.4f})")
    axes[0, 1].set_title("Normal Beat Residual Error (Low)", fontweight="bold")
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)

    # Row 2: Anomalous beat
    axes[1, 0].plot(originals[anom_idx], color="#d62728", label="Original Arrhythmic Beat", lw=2)
    axes[1, 0].plot(reconstructed[anom_idx], color="#9467bd", linestyle="--", label="Autoencoder Reconstruction", lw=2)
    axes[1, 0].set_title(f"Cardiac Arrhythmia / Ectopic Beat (Reconstruction MSE: {errors[anom_idx]:.4f} > Threshold {threshold:.4f})", fontweight="bold", color="darkred")
    axes[1, 0].set_xlabel("Sample Point (t)")
    axes[1, 0].set_ylabel("Normalized Voltage (mV)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)

    residual_anom = (originals[anom_idx] - reconstructed[anom_idx]) ** 2
    axes[1, 1].plot(residual_anom, color="#d62728", label="Squared Error Residual Peak", lw=1.5)
    axes[1, 1].axhline(y=threshold, color="red", linestyle=":", label=f"Detection Threshold ({threshold:.4f})")
    axes[1, 1].set_title("Anomalous Beat Residual Error (High Diagnostic Peak)", fontweight="bold", color="darkred")
    axes[1, 1].set_xlabel("Sample Point (t)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_anomaly_score_distribution(
    errors: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    save_path: str
):
    """Plots the reconstruction error histograms separating normal from anomalous beats."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    normal_errs = errors[labels == 0]
    anom_errs = errors[labels == 1]

    ax.hist(normal_errs, bins=30, alpha=0.65, color="#1f77b4", label="Normal Sinus Rhythm", density=True)
    ax.hist(anom_errs, bins=30, alpha=0.65, color="#d62728", label="Cardiac Arrhythmia (Anomalies)", density=True)
    ax.axvline(threshold, color="black", linestyle="--", lw=2, label=f"Optimal Decision Boundary ({threshold:.4f})")

    ax.set_title("ECG Reconstruction Error Distribution & Anomaly Threshold", fontsize=12, fontweight="bold")
    ax.set_xlabel("Reconstruction Mean Squared Error (MSE)")
    ax.set_ylabel("Probability Density")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
