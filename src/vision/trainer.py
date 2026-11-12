"""
Vision Model Trainers with Regularization, Early Stopping & Radiomics Gradient Boosting
----------------------------------------------------------------------------------------
1. VisionTrainer: End-to-end PyTorch trainer (AdamW + CosineAnnealingLR + Early Stopping)
   supporting ResidualChestCNN and TransferChestClassifier backbones.
2. train_radiomic_lgbm: Extracts 14 quantitative radiomic biomarkers strictly inside each
   split and fits a leak-free StandardScaler + LightGBM classifier pipeline.
"""

import os
import copy
from typing import Dict, List, Any, Tuple
import numpy as np
from PIL import Image
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

from ..utils.logger import get_logger
from ..utils.metrics import compute_classification_metrics
from .radiology_tools import RADIOMIC_FEATURE_NAMES, extract_radiomic_features

logger = get_logger("VisionTrainer")


class VisionTrainer:
    """End-to-end training manager for PyTorch Chest Radiograph classifiers."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        num_epochs: int = 15,
        patience: int = 5,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.patience = patience

        self.criterion = nn.CrossEntropyLoss()
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params or self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(1, num_epochs), eta_min=1e-6)

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }
        self.best_model_weights = None
        self.best_val_loss = float("inf")

    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / max(1, total)
        epoch_acc = (correct / max(1, total)) * 100.0
        return epoch_loss, epoch_acc

    def evaluate_val(self) -> Tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels, _ in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / max(1, total)
        val_acc = (correct / max(1, total)) * 100.0
        return val_loss, val_acc

    def fit(self, save_path: str = "saved_models/best_chest_cnn.pt") -> Dict[str, List[float]]:
        logger.info(f"Starting training for {self.num_epochs} epochs on device: {self.device}")
        epochs_without_improvement = 0

        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.evaluate_val()
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            logger.info(
                f"Epoch [{epoch:02d}/{self.num_epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_weights = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.best_model_weights, save_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs (Patience = {self.patience}).")
                    break

        if self.best_model_weights is not None:
            self.model.load_state_dict(self.best_model_weights)
            logger.info(f"Loaded best checkpoint (Val Loss: {self.best_val_loss:.4f})")

        return self.history


def extract_radiomic_matrix(file_paths: List[str]) -> np.ndarray:
    """Computes the (N, 14) radiomic feature matrix for a list of image paths."""
    rows = []
    for path in file_paths:
        img = Image.open(path).convert("L").resize((128, 128), Image.Resampling.BILINEAR)
        feats = extract_radiomic_features(np.asarray(img, dtype=np.float32))
        rows.append([feats[k] for k in RADIOMIC_FEATURE_NAMES])
    return np.asarray(rows, dtype=np.float32)


def train_radiomic_lgbm(
    train_files: List[str],
    train_labels: List[int],
    test_files: List[str],
    test_labels: List[int],
    real_test_files: List[str] | None = None,
    real_test_labels: List[int] | None = None,
    save_path: str = "saved_models/vision_radiomic_lgbm.joblib",
) -> Tuple[Pipeline, Dict[str, Any], Dict[str, Any] | None]:
    """
    Trains a leak-free Radiomics-14 StandardScaler + LightGBM classifier strictly on `train_files`
    and evaluates on held-out `test_files` and optional isolated real PneumoniaMNIST test images.
    """
    logger.info(f"Extracting 14 Radiomic biomarkers on train (n={len(train_files)}) & test (n={len(test_files)})...")
    X_train = extract_radiomic_matrix(train_files)
    y_train = np.asarray(train_labels, dtype=np.int64)
    X_test = extract_radiomic_matrix(test_files)
    y_test = np.asarray(test_labels, dtype=np.int64)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lgbm", LGBMClassifier(
            n_estimators=220,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            verbose=-1,
        )),
    ])
    pipeline.fit(X_train, y_train)

    test_probs = pipeline.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)
    test_metrics = compute_classification_metrics(y_test, test_preds, test_probs)
    logger.info(
        f"=== Vision Evaluation [Radiomic-14 LightGBM] === Accuracy: {test_metrics['accuracy']*100:.2f}% | "
        f"ROC-AUC: {test_metrics.get('roc_auc', 0):.4f}"
    )

    real_metrics = None
    if real_test_files and real_test_labels:
        X_real = extract_radiomic_matrix(real_test_files)
        y_real = np.asarray(real_test_labels, dtype=np.int64)
        real_probs = pipeline.predict_proba(X_real)[:, 1]
        real_preds = (real_probs >= 0.5).astype(int)
        real_metrics = compute_classification_metrics(y_real, real_preds, real_probs)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump({"pipeline": pipeline, "feature_names": RADIOMIC_FEATURE_NAMES}, save_path)
    logger.info(f"Saved Radiomic LightGBM artifact to {save_path}")
    return pipeline, test_metrics, real_metrics
