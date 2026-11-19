"""
Pydantic Request & Response Schemas for Deep Health AI API v2.0
----------------------------------------------------------------
Defines strict type validations and OpenAPI documentation schemas across
Thoracic Vision, 1D ECG Electrophysiology, Clinical EHR Cardiology Triage,
Batch Cohort Evaluation, and Hugging Face Pre-Trained Medical Models.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "HEALTHY"})
    service: str = Field(..., json_schema_extra={"example": "Deep Health AI Clinical Inference Microservice"})
    version: str = Field(..., json_schema_extra={"example": "2.0.0"})
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
    clinical: Optional[ModelInfo] = None
    available_vision_models: List[str] = Field(default_factory=list)
    available_ecg_models: List[str] = Field(default_factory=list)


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


class HotspotPoint(BaseModel):
    x_norm: float
    y_norm: float
    peak_activation: float


class VisionDiagnosisResponse(BaseModel):
    filename: str
    model_used: str = "ResidualChestCNN"
    diagnosis: str
    confidence: float
    normal_probability: float
    pneumonia_probability: float
    uncertainty: UncertaintyDetails
    severity: SeverityDetails
    gradcam_available: bool
    gradcam_heatmap_base64: Optional[str] = None
    gradcam_overlay_base64: Optional[str] = None
    hotspot: Optional[HotspotPoint] = None
    radiomics: Dict[str, float] = Field(default_factory=dict)
    timestamp: str


class ECGScreenRequest(BaseModel):
    signal: List[float] = Field(..., description="1D Lead-II / MIT-BIH ECG voltage samples")
    sampling_rate_hz: int = Field(250, description="Sampling rate in Hz (default: 250 Hz)")
    threshold_override: Optional[float] = Field(None, description="Optional override for anomaly decision boundary")
    model_id: str = Field("autoencoder", description="Model selector: autoencoder | resnet1d | mitbih_resnet1d")


class ECGTelemetry(BaseModel):
    model_config = ConfigDict(extra="allow")
    heart_rate_bpm: int
    qrs_duration_ms: float
    pr_interval_ms: float
    qt_interval_ms: float
    qtc_interval_ms: float
    st_segment_mv: float
    r_peak_amplitude: float
    p_wave_amplitude: Optional[float] = None
    t_wave_amplitude: Optional[float] = None
    r_s_ratio: Optional[float] = None
    signal_rms: Optional[float] = None
    dominant_freq_hz: Optional[float] = None
    spectral_entropy: Optional[float] = None
    clinical_flags: Dict[str, bool]


class ECGScreenResponse(BaseModel):
    model_used: str = "Conv1dAutoencoder + ECGResNet1D"
    diagnosis: str
    is_anomaly: bool
    reconstruction_mse: float
    decision_threshold: float
    rhythm_subtype: str = "Normal Sinus Rhythm"
    subtype_probabilities: Dict[str, float] = Field(default_factory=dict)
    reconstructed_signal: List[float] = Field(default_factory=list)
    residual_profile: List[float] = Field(default_factory=list)
    telemetry: ECGTelemetry
    timestamp: str


class ClinicalTriageRequest(BaseModel):
    age: float = Field(..., ge=18, le=110)
    sex: int = Field(..., ge=0, le=1, description="1=Male, 0=Female")
    chest_pain_type: int = Field(..., ge=1, le=4, description="1=Typical Angina, 2=Atypical, 3=Non-anginal, 4=Asymptomatic")
    resting_bp: Optional[float] = Field(130.0, ge=50, le=260)
    cholesterol: Optional[float] = Field(220.0, ge=50, le=650)
    fasting_bs: int = Field(0, ge=0, le=1)
    resting_ecg: int = Field(0, ge=0, le=2)
    max_hr: float = Field(150.0, ge=50, le=230)
    exercise_angina: int = Field(0, ge=0, le=1)
    oldpeak: float = Field(0.8, ge=-2.5, le=8.0)
    st_slope: int = Field(2, ge=1, le=3)
    cxr_pneumonia_probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    ecg_is_anomaly: Optional[bool] = None
    ecg_reconstruction_mse: Optional[float] = None


class ClinicalTriageResponse(BaseModel):
    model_used: str
    cad_risk_probability: float
    predicted_label: str
    fused_risk_score: float
    triage_tier: str
    recommended_urgency: str
    multimodal_modifiers: List[str]
    clinical_risk_drivers: List[Dict[str, Any]]
    engineered_biomarkers: Dict[str, float]
    hf_consensus: Optional[Dict[str, Any]] = None
    timestamp: str


class BatchEvaluateRequest(BaseModel):
    modality: str = Field(..., description="vision | vision_real | ecg | clinical")
    limit: int = Field(20, ge=5, le=100)


class PretrainedECGRequest(BaseModel):
    signal: List[float]
    repo_id: str = "dheerajthuvara/ecg-arrhythmia-detection"
