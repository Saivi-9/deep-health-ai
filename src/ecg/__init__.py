"""ECG Bio-Signal Anomaly Detection & 1D Deep Autoencoder."""

try:
    from .model import Conv1dAutoencoder
    from .dataset import ECGDataset, load_ecg_dataloaders
    from .anomaly_detector import ECGAnomalyDetector
    from .trainer import ECGAutoencoderTrainer
except ImportError:  # pragma: no cover - allow pure NumPy/SciPy telemetry imports without torch
    pass

__all__ = [
    "Conv1dAutoencoder",
    "ECGDataset",
    "load_ecg_dataloaders",
    "ECGAnomalyDetector",
    "ECGAutoencoderTrainer",
]
