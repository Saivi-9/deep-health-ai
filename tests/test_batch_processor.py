"""
Unit Tests for Batch Cohort Processor
-------------------------------------
Verifies population metrics, CSV dataframe compilation, and multi-sample inference.
"""

import os
import pytest
import numpy as np
from PIL import Image

from batch_processor import process_vision_cohort, process_ecg_cohort


def test_process_vision_cohort(tmp_path):
    # Create 3 small mock images
    paths = []
    for i in range(3):
        p = tmp_path / f"scan_{i}.png"
        img = Image.fromarray((np.random.rand(128, 128) * 255).astype(np.uint8))
        img.save(p)
        paths.append(str(p))

    summary = process_vision_cohort(paths, device="cpu")
    assert summary["total_scans_evaluated"] == 3
    assert 0.0 <= summary["pathology_prevalence_pct"] <= 100.0
    assert "results_dataframe" in summary
    assert len(summary["results_dataframe"]) == 3
    assert "predictive_variance" in summary["results_dataframe"].columns


def test_process_ecg_cohort():
    # 5 synthetic beats
    signals = np.sin(np.linspace(0, 10, 256 * 5)).reshape(5, 256).astype(np.float32)
    summary = process_ecg_cohort(signals, device="cpu")

    assert summary["total_beats_evaluated"] == 5
    assert 0.0 <= summary["arrhythmia_burden_pct"] <= 100.0
    assert "results_dataframe" in summary
    assert len(summary["results_dataframe"]) == 5
    assert "heart_rate_bpm" in summary["results_dataframe"].columns
