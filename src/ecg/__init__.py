"""ECG Bio-Signal Anomaly Detection & 1D Deep Autoencoder."""

from .telemetry import (
    ECG_FEATURE_NAMES,
    analyze_ecg_telemetry,
    extract_ecg_feature_vector,
    extract_ecg_feature_matrix,
    summarize_ecg_clinical_risk,
)

try:
    from .model import Conv1dAutoencoder
    from .dataset import ECGDataset, load_ecg_dataloaders
    from .anomaly_detector import ECGAnomalyDetector
    from .trainer import ECGAutoencoderTrainer
except ImportError:  # pragma: no cover - allow pure NumPy/SciPy telemetry imports without torch
    pass

__all__ = [
    "ECG_FEATURE_NAMES",
    "analyze_ecg_telemetry",
    "extract_ecg_feature_vector",
    "extract_ecg_feature_matrix",
    "summarize_ecg_clinical_risk",
    "Conv1dAutoencoder",
    "ECGDataset",
    "load_ecg_dataloaders",
    "ECGAnomalyDetector",
    "ECGAutoencoderTrainer",
]
