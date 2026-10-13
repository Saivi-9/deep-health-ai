"""Unit tests for ECG 1D Autoencoder Pipeline."""

import pytest
import numpy as np
import torch
from src.ecg.model import Conv1dAutoencoder
from src.ecg.dataset import ECGDataset
from src.ecg.anomaly_detector import ECGAnomalyDetector


def test_conv1d_autoencoder_shape():
    model = Conv1dAutoencoder(in_channels=1)
    model.eval()
    x = torch.randn(4, 1, 256)
    out = model(x)
    assert out.shape == (4, 1, 256), f"Expected shape (4, 1, 256), got {out.shape}"


def test_ecg_dataset_loading():
    raw_signals = np.random.randn(10, 256).astype(np.float32)
    labels = np.array([0, 1] * 5, dtype=np.int64)

    dataset = ECGDataset(raw_signals, labels)
    assert len(dataset) == 10

    sample, lbl = dataset[0]
    assert sample.shape == (1, 256)
    assert lbl == 0


def test_ecg_anomaly_detector_single_prediction():
    model = Conv1dAutoencoder(in_channels=1)
    detector = ECGAnomalyDetector(model=model, threshold=0.05)

    test_beat = np.random.randn(256).astype(np.float32)
    res = detector.predict(test_beat)

    assert "reconstruction_error" in res
    assert "is_anomaly" in res
    assert "diagnosis" in res
    assert res["reconstructed_signal"].shape == (256,)
    assert isinstance(res["is_anomaly"], bool)
