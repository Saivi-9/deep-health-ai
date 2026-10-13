"""
Clinical Diagnostic & Deep Learning Evaluation Metrics
------------------------------------------------------
Calculates comprehensive clinical and statistical performance metrics:
- Sensitivity (Recall)
- Specificity (True Negative Rate)
- Precision (Positive Predictive Value)
- F1-Score
- ROC-AUC & PR-AUC
- Confusion Matrix (TP, FP, TN, FN)
"""

from typing import Dict, Any, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

def compute_classification_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    y_prob: Union[np.ndarray, list] = None
) -> Dict[str, Any]:
    """
    Computes rigorous clinical classification evaluation metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        false_alarm_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    else:
        tn = fp = fn = tp = 0
        specificity = 0.0
        false_alarm_rate = 0.0

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "sensitivity_recall": rec,
        "specificity": specificity,
        "f1_score": f1,
        "false_alarm_rate": false_alarm_rate,
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        }
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None

    return metrics
