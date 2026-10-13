"""
Vision Model Trainer with Regularization & Early Stopping
---------------------------------------------------------
Coordinates PyTorch training lifecycle:
- Optimizer: AdamW with L2 weight decay
- Learning Rate Scheduler: Cosine Annealing
- Early Stopping on validation loss
- Checkpoint persistence
"""

import os
import copy
from typing import Dict, List, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..utils.logger import get_logger

logger = get_logger("VisionTrainer")


class VisionTrainer:
    """End-to-end training manager for the Chest Radiograph CNN."""

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
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }
        self.best_model_weights = None
        self.best_val_loss = float("inf")

    def train_epoch(self) -> (float, float):
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

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100.0
        return epoch_loss, epoch_acc

    def evaluate_val(self) -> (float, float):
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

        val_loss = running_loss / total
        val_acc = (correct / total) * 100.0
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

            # Checkpoint tracking
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
