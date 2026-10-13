"""
ECG Autoencoder Trainer
-----------------------
Supervises unsupervised reconstruction training on normal cardiac signals:
- Mean Squared Error (MSE) loss minimization
- Learning rate decay
- Early stopping on validation loss
"""

import os
import copy
from typing import Dict, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..utils.logger import get_logger

logger = get_logger("ECGTrainer")


class ECGAutoencoderTrainer:
    """Trainer for 1D ECG Convolutional Autoencoder."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        num_epochs: int = 20,
        patience: int = 5,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.patience = patience

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=2)

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "lr": []
        }
        self.best_model_weights = None
        self.best_val_loss = float("inf")

    def train_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        total = 0

        for signals, _ in self.train_loader:
            signals = signals.to(self.device)
            self.optimizer.zero_grad()
            reconstructed = self.model(signals)
            loss = self.criterion(reconstructed, signals)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * signals.size(0)
            total += signals.size(0)

        return running_loss / total

    def evaluate_val(self) -> float:
        self.model.eval()
        running_loss = 0.0
        total = 0

        with torch.no_grad():
            for signals, _ in self.val_loader:
                signals = signals.to(self.device)
                reconstructed = self.model(signals)
                loss = self.criterion(reconstructed, signals)

                running_loss += loss.item() * signals.size(0)
                total += signals.size(0)

        return running_loss / total

    def fit(self, save_path: str = "saved_models/best_ecg_autoencoder.pt") -> Dict[str, List[float]]:
        logger.info(f"Starting ECG autoencoder training for {self.num_epochs} epochs on device: {self.device}")
        epochs_without_improvement = 0

        for epoch in range(1, self.num_epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.evaluate_val()
            self.scheduler.step(val_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(current_lr)

            logger.info(
                f"Epoch [{epoch:02d}/{self.num_epochs:02d}] "
                f"Train Recon Loss: {train_loss:.6f} | Val Recon Loss: {val_loss:.6f} | LR: {current_lr:.6f}"
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
                    logger.info(f"Early stopping triggered after {epoch} epochs.")
                    break

        if self.best_model_weights is not None:
            self.model.load_state_dict(self.best_model_weights)
            logger.info(f"Loaded best checkpoint (Val MSE Loss: {self.best_val_loss:.6f})")

        return self.history
