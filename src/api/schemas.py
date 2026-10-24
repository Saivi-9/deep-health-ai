"""
Pydantic Request & Response Schemas for Deep Health AI API
-----------------------------------------------------------
Defines strict type validations and OpenAPI documentation schemas.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "HEALTHY"})
    service: str = Field(..., json_schema_extra={"example": "Deep Health AI Clinical Inference Microservice"})
    version: str = Field(..., json_schema_extra={"example": "1.3.0"})
    device: str = Field(..., json_schema_extra={"example": "cpu"})
    models_loaded: Dict[str, bool]


class ModelInfo(BaseModel):
    name: str
    architecture: str
    trainable_parameters: int
    input_shape: List[int]
    status: str


class ModelsMetadataResponse(BaseModel):
    vision: ModelInfo
    ecg: ModelInfo


class UncertaintyDetails(BaseModel):
    predictive_variance: float
    normalized_entropy: float
    uncertainty_risk_level: str
    human_referral_mandatory: bool
    monte_carlo_samples: int


class SeverityDetails(BaseModel):
    radiological_severity: str
    radiological_grade: str
    infiltrate_burden_percentage: float
    clinical_triage_urgency: str


class VisionDiagnosisResponse(BaseModel):
    filename: str
    diagnosis: str
    confidence: float
    normal_probability: float
    pneumonia_probability: float
    uncertainty: UncertaintyDetails
    severity: SeverityDetails
    gradcam_available: bool
    timestamp: str


class ECGScreenRequest(BaseModel):
    signal: List[float] = Field(..., description="1D Lead-II ECG voltage samples (typically 256 samples)")
    sampling_rate_hz: int = Field(250, description="Sampling rate in Hz (default: 250 Hz)")
    threshold_override: Optional[float] = Field(None, description="Optional override for anomaly decision boundary")


class ECGTelemetry(BaseModel):
    heart_rate_bpm: int
    qrs_duration_ms: float
    pr_interval_ms: float
    qt_interval_ms: float
    qtc_interval_ms: float
    st_segment_mv: float
    r_peak_amplitude: float
    clinical_flags: Dict[str, bool]


class ECGScreenResponse(BaseModel):
    diagnosis: str
    is_anomaly: bool
    reconstruction_mse: float
    decision_threshold: float
    telemetry: ECGTelemetry
    timestamp: str
