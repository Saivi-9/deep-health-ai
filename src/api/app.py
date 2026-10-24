"""
FastAPI Clinical Decision Support REST API Application
-------------------------------------------------------
Provides high-performance, asynchronous REST endpoints for clinical deep learning inference:
- Thoracic Radiograph interpretation with Grad-CAM & MC Dropout uncertainty
- Lead-II ECG anomaly screening with electrophysiological telemetry
- Health status and model metadata inspection
"""

import os
import json
from io import BytesIO
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.vision.model import ResidualChestCNN
from src.vision.gradcam import GradCAM
from src.vision.uncertainty import MonteCarloDropoutEstimator
from src.vision.radiology_tools import compute_consolidation_severity
from src.ecg.model import Conv1dAutoencoder
from src.ecg.anomaly_detector import ECGAnomalyDetector
from src.ecg.telemetry import analyze_ecg_telemetry
from src.api.schemas import (
    HealthResponse,
    ModelsMetadataResponse,
    ModelInfo,
    VisionDiagnosisResponse,
    UncertaintyDetails,
    SeverityDetails,
    ECGScreenRequest,
    ECGScreenResponse,
    ECGTelemetry
)

app = FastAPI(
    title="Deep Health AI - Clinical Decision Support Microservice",
    description="Production REST API providing multi-modal 2D radiograph interpretation and 1D ECG arrhythmia telemetry.",
    version="1.3.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for hospital frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model Singletons
_vision_model: Optional[ResidualChestCNN] = None
_ecg_detector: Optional[ECGAnomalyDetector] = None
_default_ecg_threshold: float = 0.01037


def get_vision_model() -> ResidualChestCNN:
    global _vision_model
    if _vision_model is None:
        model = ResidualChestCNN(in_channels=1, num_classes=2)
        weights_path = "saved_models/best_chest_cnn.pt"
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=torch.device(DEVICE), weights_only=True))
        model.to(DEVICE)
        model.eval()
        _vision_model = model
    return _vision_model


def get_ecg_detector() -> ECGAnomalyDetector:
    global _ecg_detector, _default_ecg_threshold
    if _ecg_detector is None:
        model = Conv1dAutoencoder(in_channels=1)
        weights_path = "saved_models/best_ecg_autoencoder.pt"
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=torch.device(DEVICE), weights_only=True))
        model.to(DEVICE)
        model.eval()

        threshold = 0.01037
        metrics_path = "reports/ecg_metrics.json"
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r") as f:
                    data = json.load(f)
                    threshold = float(data.get("threshold", 0.01037))
            except Exception:
                pass
        _default_ecg_threshold = threshold
        _ecg_detector = ECGAnomalyDetector(model=model, threshold=threshold, device=DEVICE)
    return _ecg_detector


@app.get("/health", response_model=HealthResponse, tags=["System Telemetry"])
def health_check():
    """Returns microservice health status, compute hardware, and loaded model state."""
    v_loaded = os.path.exists("saved_models/best_chest_cnn.pt")
    e_loaded = os.path.exists("saved_models/best_ecg_autoencoder.pt")
    return HealthResponse(
        status="HEALTHY",
        service="Deep Health AI Clinical Inference Microservice",
        version="1.3.0",
        device=DEVICE,
        models_loaded={
            "ResidualChestCNN": v_loaded,
            "Conv1dAutoencoder": e_loaded
        }
    )


@app.get("/v1/models", response_model=ModelsMetadataResponse, tags=["Model Governance"])
def get_models_metadata():
    """Inspects neural network specifications, parameter counts, and tensor dimensions."""
    v_model = get_vision_model()
    e_detector = get_ecg_detector()

    v_params = sum(p.numel() for p in v_model.parameters() if p.requires_grad)
    e_params = sum(p.numel() for p in e_detector.model.parameters() if p.requires_grad)

    return ModelsMetadataResponse(
        vision=ModelInfo(
            name="ResidualChestCNN + SpatialAttention",
            architecture="Deep Residual CNN with CBAM-style Spatial Gating & Grad-CAM hook",
            trainable_parameters=v_params,
            input_shape=[1, 1, 128, 128],
            status="READY_IN_MEMORY"
        ),
        ecg=ModelInfo(
            name="Conv1dAutoencoder",
            architecture="1D Convolutional Autoencoder with Transposed Deconvolutions",
            trainable_parameters=e_params,
            input_shape=[1, 1, 256],
            status="READY_IN_MEMORY"
        )
    )


@app.post("/v1/vision/diagnose", response_model=VisionDiagnosisResponse, tags=["Computer Vision Diagnostics"])
async def diagnose_chest_radiograph(
    file: UploadFile = File(..., description="Chest radiograph image (PNG/JPG)"),
    mc_samples: int = Query(20, ge=5, le=50, description="Monte Carlo Dropout passes for uncertainty quantification")
):
    """
    Diagnoses a chest radiograph for alveolar consolidation/pneumonia.
    Returns prediction probabilities, Grad-CAM attention status, epistemic variance, and severity score.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {file.content_type}. Expected an image (image/png, image/jpeg)."
        )

    try:
        contents = await file.read()
        pil_img = Image.open(BytesIO(contents)).convert("L")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to decode image: {e}")

    model = get_vision_model()

    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    tensor_input = transform(pil_img).unsqueeze(0).to(DEVICE)

    # 1. Forward Pass
    with torch.no_grad():
        logits = model(tensor_input)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))

    class_names = ["Normal Healthy Parenchyma", "Alveolar Consolidation / Pneumonia"]
    diagnosis_str = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    # 2. MC Dropout Uncertainty
    mc_estimator = MonteCarloDropoutEstimator(model, n_samples=mc_samples, device=DEVICE)
    uncert = mc_estimator.estimate_uncertainty(tensor_input)

    # 3. Grad-CAM & Severity Calculation
    gradcam = GradCAM(model, model.layer4)
    heatmap, _, _ = gradcam.generate_heatmap(tensor_input, target_class=pred_idx)
    gradcam.remove_hooks()
    severity_info = compute_consolidation_severity(heatmap, threshold=0.40)

    return VisionDiagnosisResponse(
        filename=file.filename or "uploaded_radiograph.png",
        diagnosis=diagnosis_str,
        confidence=round(confidence, 4),
        normal_probability=round(float(probs[0]), 4),
        pneumonia_probability=round(float(probs[1]), 4),
        uncertainty=UncertaintyDetails(
            predictive_variance=round(uncert["predictive_variance"], 6),
            normalized_entropy=round(uncert["normalized_entropy"], 4),
            uncertainty_risk_level=uncert["uncertainty_risk_level"],
            human_referral_mandatory=uncert["human_referral_mandatory"],
            monte_carlo_samples=mc_samples
        ),
        severity=SeverityDetails(
            radiological_severity=severity_info["radiological_severity"],
            radiological_grade=severity_info["radiological_grade"],
            infiltrate_burden_percentage=severity_info["infiltrate_burden_percentage"],
            clinical_triage_urgency=severity_info["clinical_triage_urgency"]
        ),
        gradcam_available=True,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.post("/v1/ecg/screen", response_model=ECGScreenResponse, tags=["Electrophysiological Screening"])
def screen_ecg_signal(payload: ECGScreenRequest):
    """
    Analyzes a 1D Lead-II ECG waveform using the trained 1D Convolutional Autoencoder.
    Returns anomaly detection flag, reconstruction MSE, and electrophysiological telemetry.
    """
    raw_signal = np.array(payload.signal, dtype=np.float32)
    if len(raw_signal) < 32:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ECG signal too short. Minimum 32 samples required."
        )

    # Pad or slice to 256
    if len(raw_signal) > 256:
        processed_signal = raw_signal[:256]
    elif len(raw_signal) < 256:
        processed_signal = np.pad(raw_signal, (0, 256 - len(raw_signal)), mode="edge")
    else:
        processed_signal = raw_signal

    detector = get_ecg_detector()
    if payload.threshold_override is not None:
        detector.threshold = payload.threshold_override
    else:
        detector.threshold = _default_ecg_threshold

    res = detector.predict(processed_signal)
    telemetry_data = analyze_ecg_telemetry(processed_signal, sampling_rate=payload.sampling_rate_hz)

    return ECGScreenResponse(
        diagnosis=res["diagnosis"],
        is_anomaly=res["is_anomaly"],
        reconstruction_mse=round(res["reconstruction_error"], 6),
        decision_threshold=round(detector.threshold, 6),
        telemetry=ECGTelemetry(**telemetry_data),
        timestamp=datetime.now(timezone.utc).isoformat()
    )
