"""Unit tests for Clinical Telemetry and Radiology Tools."""

import pytest
import numpy as np
from src.ecg.telemetry import analyze_ecg_telemetry
from src.vision.radiology_tools import apply_radiology_window, compute_consolidation_severity


def test_ecg_telemetry_fields():
    dummy_signal = np.sin(np.linspace(0, 10, 256)).astype(np.float32)
    telemetry = analyze_ecg_telemetry(dummy_signal, sampling_rate=250)

    assert "heart_rate_bpm" in telemetry
    assert "qrs_duration_ms" in telemetry
    assert "pr_interval_ms" in telemetry
    assert "qtc_interval_ms" in telemetry
    assert "st_segment_mv" in telemetry
    assert "clinical_flags" in telemetry
    assert 40 <= telemetry["heart_rate_bpm"] <= 160


def test_radiology_window_presets():
    dummy_img = (np.random.rand(100, 100) * 255).astype(np.uint8)
    lung_win = apply_radiology_window(dummy_img, "Lung Window")
    bone_win = apply_radiology_window(dummy_img, "Bone Window")
    inv_win = apply_radiology_window(dummy_img, "Inverted Radiograph (Fluoroscopy)")

    assert lung_win.shape == (100, 100)
    assert bone_win.shape == (100, 100)
    assert inv_win.shape == (100, 100)
    assert lung_win.dtype == np.uint8


def test_compute_consolidation_severity():
    blank_heatmap = np.zeros((128, 128), dtype=np.float32)
    sev_normal = compute_consolidation_severity(blank_heatmap)
    assert "Normal" in sev_normal["radiological_severity"]

    hot_heatmap = np.ones((128, 128), dtype=np.float32)
    sev_severe = compute_consolidation_severity(hot_heatmap)
    assert "Severe" in sev_severe["radiological_severity"]
    assert sev_severe["infiltrate_burden_percentage"] == 100.0
