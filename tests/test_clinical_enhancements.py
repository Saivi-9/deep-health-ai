"""Unit tests for Clinical Telemetry and Radiology Tools."""

import pytest
import numpy as np
from src.ecg.telemetry import analyze_ecg_telemetry, summarize_ecg_clinical_risk
from src.vision.radiology_tools import apply_radiology_window, compute_consolidation_severity
from src.clinical.features import audit_patient_biomarkers


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


def test_audit_patient_biomarkers():
    healthy = {
        "age": 38,
        "sex": 0,
        "chest_pain_type": 1,
        "resting_bp": 116,
        "cholesterol": 182,
        "fasting_bs": 0,
        "resting_ecg": 0,
        "max_hr": 172,
        "exercise_angina": 0,
        "oldpeak": 0.1,
        "st_slope": 1,
        "ca": 0,
        "thal": 2,
    }
    res_ok = audit_patient_biomarkers(healthy)
    assert res_ok["overall_audit_status"] == "NORMAL"
    assert res_ok["abnormal_biomarker_count"] == 0
    assert len(res_ok["engineered_biomarkers"]) == 9

    critical = {
        "age": 64,
        "sex": 1,
        "chest_pain_type": 3,
        "resting_bp": 165,
        "cholesterol": 265,
        "fasting_bs": 1,
        "resting_ecg": 1,
        "max_hr": 112,
        "exercise_angina": 1,
        "oldpeak": 2.6,
        "st_slope": 2,
        "ca": 2,
        "thal": 3,
    }
    res_crit = audit_patient_biomarkers(critical)
    assert res_crit["overall_audit_status"] == "CRITICAL_FINDINGS"
    assert res_crit["abnormal_biomarker_count"] >= 5


def test_summarize_ecg_clinical_risk_tiers():
    normal_tele = {
        "qrs_duration_ms": 88.0,
        "qtc_interval_ms": 405.0,
        "st_segment_mv": 0.04,
        "clinical_flags": {
            "wide_qrs_ventricular_conduction_delay": False,
            "prolonged_repolarization_qtc": False,
            "st_segment_abnormality_ischemia": False,
            "atrial_fibrillation_absent_p_wave": False,
        },
    }
    res_norm = summarize_ecg_clinical_risk(normal_tele, is_anomaly=False, reconstruction_mse=0.004)
    assert res_norm["urgency_tier"] == "ROUTINE_SINUS_MONITORING"
    assert res_norm["active_flag_count"] == 0

    crit_tele = {
        "qrs_duration_ms": 144.0,
        "qtc_interval_ms": 475.0,
        "st_segment_mv": 0.35,
        "clinical_flags": {
            "wide_qrs_ventricular_conduction_delay": True,
            "prolonged_repolarization_qtc": True,
            "st_segment_abnormality_ischemia": True,
            "atrial_fibrillation_absent_p_wave": False,
        },
    }
    res_crit = summarize_ecg_clinical_risk(crit_tele, is_anomaly=True, reconstruction_mse=0.052)
    assert res_crit["urgency_tier"] == "CRITICAL_ELECTROPHYSIOLOGY"
    assert res_crit["active_flag_count"] == 3
    assert len(res_crit["clinical_findings"]) >= 3


