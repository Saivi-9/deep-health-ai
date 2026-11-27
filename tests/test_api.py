"""
Unit Tests for Deep Health AI FastAPI REST Endpoints
-----------------------------------------------------
Verifies API routes, request validation, and response schemas using Starlette/HTTPX TestClient.
"""

from io import BytesIO
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert "service" in data
    assert "models_loaded" in data


def test_models_metadata_endpoint():
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "vision" in data
    assert "ecg" in data
    assert data["vision"]["trainable_parameters"] > 0
    assert data["ecg"]["trainable_parameters"] > 0


def test_vision_diagnose_valid_image():
    # Create a small in-memory 128x128 grayscale test image
    img = Image.fromarray((np.random.rand(128, 128) * 255).astype(np.uint8))
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/v1/vision/diagnose?mc_samples=10",
        files={"file": ("test_xray.png", buf, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "diagnosis" in data
    assert "confidence" in data
    assert "uncertainty" in data
    assert "severity" in data
    assert data["gradcam_available"] is True
    assert 0.0 <= data["confidence"] <= 1.0


def test_vision_diagnose_invalid_file_type():
    buf = BytesIO(b"not an image file content")
    response = client.post(
        "/v1/vision/diagnose",
        files={"file": ("notes.txt", buf, "text/plain")}
    )
    assert response.status_code == 400


def test_ecg_screen_valid_signal():
    # Generate 256-sample synthetic sine signal
    signal = (np.sin(np.linspace(0, 4 * np.pi, 256))).tolist()
    payload = {
        "signal": signal,
        "sampling_rate_hz": 250,
        "threshold_override": 0.015
    }
    response = client.post("/v1/ecg/screen", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "diagnosis" in data
    assert "is_anomaly" in data
    assert "reconstruction_mse" in data
    assert "telemetry" in data
    assert data["decision_threshold"] == 0.015
    assert data["telemetry"]["heart_rate_bpm"] > 0


def test_ecg_screen_too_short():
    payload = {"signal": [0.1, 0.2]}
    response = client.post("/v1/ecg/screen", json=payload)
    assert response.status_code == 400


def test_system_diagnostics_endpoint():
    response = client.get("/v1/system/diagnostics")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("READY", "DEGRADED")
    assert "device" in data
    assert "torch_version" in data
    assert data["checkpoints_total"] == 9
    assert isinstance(data["checkpoints"], list)
    assert "datasets_available" in data


def test_clinical_biomarker_reference_endpoint():
    response = client.get("/v1/clinical/biomarker-reference")
    assert response.status_code == 200
    data = response.json()
    assert data["total_pipeline_biomarkers"] == 20
    assert data["raw_feature_count"] == 11
    assert data["engineered_feature_count"] == 9
    assert len(data["reference_ranges"]) >= 7


def test_clinical_biomarker_audit_endpoint():
    payload = {
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
    response = client.post("/v1/clinical/biomarker-audit", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["overall_audit_status"] == "CRITICAL_FINDINGS"
    assert data["abnormal_biomarker_count"] >= 5
    flagged_names = {f["feature"] for f in data["flagged_biomarkers"]}
    assert "resting_bp" in flagged_names
    assert "oldpeak" in flagged_names
    assert len(data["engineered_biomarkers"]) == 9


def test_ecg_clinical_summary_endpoint():
    signal = (np.sin(np.linspace(0, 4 * np.pi, 256))).tolist()
    response = client.post(
        "/v1/ecg/clinical-summary",
        json={"signal": signal, "sampling_rate_hz": 250, "threshold_override": 0.015},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["urgency_tier"] in (
        "CRITICAL_ELECTROPHYSIOLOGY",
        "ELEVATED_CONDUCTION_RISK",
        "ROUTINE_SINUS_MONITORING",
    )
    assert isinstance(data["clinical_findings"], list)
    assert len(data["clinical_findings"]) >= 1
    assert "telemetry" in data


def test_clinical_cardiovascular_profile_endpoint():
    payload = {
        "age": 58,
        "sex": 1,
        "chest_pain_type": 2,
        "resting_bp": 158,
        "cholesterol": 255,
        "fasting_bs": 1,
        "resting_ecg": 1,
        "max_hr": 124,
        "exercise_angina": 1,
        "oldpeak": 2.2,
        "st_slope": 2,
        "ca": 2,
        "thal": 3,
    }
    response = client.post("/v1/clinical/cardiovascular-profile", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["chronological_age"] == 58
    assert data["estimated_vascular_age"] > 58.0
    assert data["vascular_age_acceleration_years"] >= 15.0
    assert data["risk_category"] == "VERY_HIGH_CARDIOVASCULAR_RISK"
    assert len(data["preventive_recommendations"]) >= 3


def test_clinical_ischemic_burden_endpoint():
    payload = {
        "age": 62,
        "sex": 1,
        "chest_pain_type": 3,
        "resting_bp": 152,
        "cholesterol": 248,
        "fasting_bs": 1,
        "resting_ecg": 1,
        "max_hr": 118,
        "exercise_angina": 1,
        "oldpeak": 2.4,
        "st_slope": 2,
        "ca": 2,
        "thal": 3,
    }
    response = client.post("/v1/clinical/ischemic-burden", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["duke_treadmill_score"] <= -11.0
    assert data["ischemic_burden_index"] >= 10.0
    assert data["fluoroscopic_vessels_involved"] == 2
    assert data["ischemia_severity_tier"] == "SEVERE_MYOCARDIAL_ISCHEMIA"
    assert "coronary angiography" in data["recommended_workup"]


def test_clinical_cardiometabolic_index_endpoint():
    payload = {
        "age": 60,
        "sex": 1,
        "chest_pain_type": 2,
        "resting_bp": 156,
        "cholesterol": 262,
        "fasting_bs": 1,
        "resting_ecg": 1,
        "max_hr": 122,
        "exercise_angina": 1,
        "oldpeak": 1.8,
        "st_slope": 2,
        "ca": 1,
        "thal": 3,
    }
    response = client.post("/v1/clinical/cardiometabolic-index", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["metabolic_risk_factors_count"] == 4
    assert data["atherogenic_lipid_ratio_est"] > 5.0
    assert data["cardiometabolic_index_score"] >= 7.5
    assert data["syndrome_tier"] == "HIGH_CARDIOMETABOLIC_RISK"
    assert "lipid-lowering" in data["clinical_guidance"]


def test_clinical_chronotropic_reserve_endpoint():
    payload = {
        "age": 60,
        "sex": 1,
        "chest_pain_type": 3,
        "resting_bp": 148,
        "cholesterol": 240,
        "fasting_bs": 0,
        "resting_ecg": 1,
        "max_hr": 112,
        "exercise_angina": 1,
        "oldpeak": 2.0,
        "st_slope": 2,
        "ca": 1,
        "thal": 3,
    }
    response = client.post("/v1/clinical/chronotropic-reserve", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["age_predicted_max_hr_bpm"] == 160.0
    assert data["pct_predicted_max_hr"] == 70.0
    assert data["chronotropic_incompetence"] is True
    assert data["ischemic_co_limitation"] is True
    assert data["chronotropic_status"] == "SEVERE_CHRONOTROPIC_INCOMPETENCE"


def test_clinical_vascular_stiffness_endpoint():
    payload = {
        "age": 66,
        "sex": 1,
        "chest_pain_type": 3,
        "resting_bp": 168,
        "cholesterol": 255,
        "fasting_bs": 1,
        "resting_ecg": 1,
        "max_hr": 118,
        "exercise_angina": 1,
        "oldpeak": 1.8,
        "st_slope": 2,
        "ca": 1,
        "thal": 3,
    }
    response = client.post("/v1/clinical/vascular-stiffness", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["resting_systolic_bp_mmhg"] == 168.0
    assert data["estimated_pulse_pressure_mmhg"] >= 60.0
    assert data["mean_arterial_pressure_mmhg"] > 90.0
    assert data["arterial_compliance_tier"] == "HIGH_ARTERIAL_STIFFNESS"









