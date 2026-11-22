"""Clinical EHR Cardiology Triage & Multi-Modal Cardiopulmonary Risk Fusion."""

from .features import (
    RAW_CLINICAL_FEATURES,
    CLINICAL_FEATURE_NAMES,
    engineer_clinical_features,
    compute_multimodal_triage_score,
    audit_patient_biomarkers,
    compute_hemodynamic_shock_index,
)
try:
    from .trainer import train_clinical_triage_suite
except ImportError:  # pragma: no cover - optional lightgbm training dependency
    train_clinical_triage_suite = None

__all__ = [
    "RAW_CLINICAL_FEATURES",
    "CLINICAL_FEATURE_NAMES",
    "engineer_clinical_features",
    "compute_multimodal_triage_score",
    "audit_patient_biomarkers",
    "compute_hemodynamic_shock_index",
    "train_clinical_triage_suite",
]
