"""
ECG Anomaly Detection & Diagnostic Threshold Calibration
--------------------------------------------------------
Infers reconstruction errors and identifies cardiac arrhythmias via
unsupervised residual error thresholding.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..utils.logger import get_logger

logger = get_logger("ECGAnomalyDetector")


class ECGAnomalyDetector:
    """Detects ECG arrhythmias using autoencoder reconstruction loss."""

    def __init__(self, model: nn.Module, threshold: Optional[float] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.threshold = threshold

    def compute_reconstruction_errors(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes MSE reconstruction error for all samples in DataLoader.
        Returns (errors, original_signals, reconstructed_signals).
        """
        self.model.eval()
        errors = []
        originals = []
        reconstructed = []

        with torch.no_grad():
            for signals, _ in data_loader:
                signals = signals.to(self.device)
                recon = self.model(signals)

                # MSE along signal length (dim 2)
                mse = torch.mean((signals - recon) ** 2, dim=[1, 2]).cpu().numpy()
                errors.extend(mse.tolist())
                originals.append(signals.squeeze(1).cpu().numpy())
                reconstructed.append(recon.squeeze(1).cpu().numpy())

        return (
            np.array(errors),
            np.concatenate(originals, axis=0),
            np.concatenate(reconstructed, axis=0)
        )

    def calibrate_threshold(self, val_loader: DataLoader, percentile: float = 97.5) -> float:
        """
        Calibrates the anomaly decision boundary based on the distribution
        of reconstruction errors on healthy validation beats.
        """
        val_errors, _, _ = self.compute_reconstruction_errors(val_loader)
        threshold = float(np.percentile(val_errors, percentile))
        self.threshold = threshold
        logger.info(f"Calibrated anomaly threshold to {threshold:.6f} ({percentile}th percentile of normal beats)")
        return threshold

    def predict(self, signal: np.ndarray) -> Dict[str, Any]:
        """
        Predicts anomaly on a single signal (length 256).
        """
        if self.threshold is None:
            raise ValueError("Threshold is not calibrated. Run calibrate_threshold first.")

        self.model.eval()
        inp = torch.from_numpy(signal.astype(np.float32)).view(1, 1, -1).to(self.device)
        with torch.no_grad():
            recon = self.model(inp)
            mse = float(torch.mean((inp - recon) ** 2).item())

        is_anomaly = bool(mse > self.threshold)
        return {
            "reconstruction_error": mse,
            "threshold": self.threshold,
            "is_anomaly": is_anomaly,
            "diagnosis": "Cardiac Arrhythmia / Ventricular Ectopy" if is_anomaly else "Normal Sinus Rhythm",
            "reconstructed_signal": recon.squeeze().cpu().numpy()
        }
