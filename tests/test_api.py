"""
Unit Tests for Deep Health AI FastAPI REST Endpoints
-----------------------------------------------------
Verifies API routes, request validation, and response schemas using Starlette/HTTPX TestClient.
"""

from io import BytesIO
import pytest
import numpy as np
from PIL import Image
from starlette.testclient import TestClient

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
