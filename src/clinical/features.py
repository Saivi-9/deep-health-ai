"""
Clinical Cardiology EHR Feature Engineering & Multi-Modal Triage Fusion
-----------------------------------------------------------------------
Derives 20 deterministic physiological & cardiovascular stress biomarkers
(11 raw EHR attributes + 9 domain-specific interactions) with zero population leakage,
plus multimodal cardiopulmonary triage fusion across EHR + CXR + ECG.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

RAW_CLINICAL_FEATURES: List[str] = [
    "age",
    "sex",
    "chest_pain_type",
    "resting_bp",
    "cholesterol",
    "fasting_bs",
    "resting_ecg",
    "max_hr",
    "exercise_angina",
    "oldpeak",
    "st_slope",
]

ENGINEERED_CLINICAL_FEATURES: List[str] = [
    "estimated_max_hr_pct",
    "rate_pressure_product",
    "age_x_oldpeak",
    "st_severity_index",
    "metabolic_risk_count",
    "angina_pain_interaction",
    "cholesterol_to_age",
    "cardiac_ischemia_burden",
    "chronotropic_incompetence_flag",
]

CLINICAL_FEATURE_NAMES: List[str] = RAW_CLINICAL_FEATURES + ENGINEERED_CLINICAL_FEATURES


def engineer_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies row-wise deterministic clinical feature engineering without computing
    dataset-wide statistics (preserving NaNs for leak-free fold SimpleImputer).
    """
    out = df[RAW_CLINICAL_FEATURES].copy()
    for col in RAW_CLINICAL_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    age_safe = out["age"].clip(lower=18.0)
    predicted_max = (220.0 - age_safe).clip(lower=60.0)
    out["estimated_max_hr_pct"] = (out["max_hr"] / predicted_max).astype(np.float32)

    # Preserve NaN when resting_bp or max_hr is missing so training-fold imputer handles it
    out["rate_pressure_product"] = ((out["resting_bp"] * out["max_hr"]) / 1000.0).astype(np.float32)

    oldpeak_pos = out["oldpeak"].clip(lower=0.0)
    out["age_x_oldpeak"] = (age_safe * oldpeak_pos).astype(np.float32)
    out["st_severity_index"] = ((oldpeak_pos + 0.5) * (1.0 + out["exercise_angina"].fillna(0.0))).astype(np.float32)

    bp_high = (out["resting_bp"] >= 140.0).astype(float)
    chol_high = (out["cholesterol"] >= 240.0).astype(float)
    fbs_high = (out["fasting_bs"] == 1.0).astype(float)
    out["metabolic_risk_count"] = (bp_high + chol_high + fbs_high).astype(np.float32)

    out["angina_pain_interaction"] = (out["exercise_angina"] * out["chest_pain_type"]).astype(np.float32)
    out["cholesterol_to_age"] = (out["cholesterol"] / age_safe).astype(np.float32)

    ischemia_score = (
        (oldpeak_pos > 1.0).astype(float)
        + (out["st_slope"] != 1).astype(float)
        + (out["chest_pain_type"] == 4).astype(float)
    )
    out["cardiac_ischemia_burden"] = ischemia_score.astype(np.float32)
    out["chronotropic_incompetence_flag"] = ((out["max_hr"] < (0.80 * predicted_max)).astype(float)).astype(np.float32)

    return out[CLINICAL_FEATURE_NAMES]


def compute_multimodal_triage_score(
    ehr_cad_probability: float,
    patient: Dict[str, Any],
    cxr_pneumonia_probability: Optional[float] = None,
    ecg_is_anomaly: Optional[bool] = None,
    ecg_reconstruction_mse: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Combines calibrated EHR coronary risk probability with optional Chest X-Ray & ECG findings
    into an actionable cardiopulmonary acuity index and factor breakdown.
    """
    fused = float(np.clip(ehr_cad_probability, 0.0, 1.0))
    modifiers: List[str] = []

    if cxr_pneumonia_probability is not None:
        if cxr_pneumonia_probability >= 0.65:
            fused = float(min(0.99, fused + 0.10 * cxr_pneumonia_probability))
            modifiers.append(f"Concomitant alveolar consolidation (+{cxr_pneumonia_probability*100:.0f}% CXR confidence)")
        else:
            modifiers.append(f"Chest radiograph clear ({(1-cxr_pneumonia_probability)*100:.0f}% normal parenchyma)")

    if ecg_is_anomaly is not None:
        if ecg_is_anomaly:
            fused = float(min(0.99, fused + 0.12))
            mse_txt = f" (MSE={ecg_reconstruction_mse:.4f})" if ecg_reconstruction_mse is not None else ""
            modifiers.append(f"Active ECG arrhythmia / ectopic conduction abnormality{mse_txt}")
        else:
            modifiers.append("Lead-II ECG within normal sinus manifold")

    # Key clinical driver explanations
    drivers: List[Dict[str, Any]] = []
    if float(patient.get("oldpeak", 0.0) or 0.0) >= 1.5:
        drivers.append({"factor": "Exercise ST Depression (Oldpeak)", "value": f"{patient['oldpeak']} mm", "impact": "HIGH"})
    if int(patient.get("exercise_angina", 0) or 0) == 1:
        drivers.append({"factor": "Exercise-Induced Angina", "value": "Present", "impact": "HIGH"})
    if int(patient.get("chest_pain_type", 1) or 1) == 4:
        drivers.append({"factor": "Asymptomatic / Atypical Ischemic Chest Pain", "value": "Type 4", "impact": "MODERATE"})
    if float(patient.get("max_hr", 150) or 150) < 120:
        drivers.append({"factor": "Chronotropic Reserve Limitation", "value": f"{patient['max_hr']} bpm", "impact": "MODERATE"})
    if float(patient.get("resting_bp", 120) or 120) >= 140:
        drivers.append({"factor": "Resting Arterial Hypertension", "value": f"{patient['resting_bp']} mmHg", "impact": "MODERATE"})
    if float(patient.get("cholesterol", 200) or 200) >= 240:
        drivers.append({"factor": "Hyperlipidemia (Serum Cholesterol)", "value": f"{patient['cholesterol']} mg/dL", "impact": "MODERATE"})
    if not drivers:
        drivers.append({"factor": "Hemodynamic & Metabolic Profile", "value": "Within reference bounds", "impact": "LOW"})

    if fused >= 0.78:
        tier = "CRITICAL - STAT CARDIOLOGY / ICU TELEMETRY"
        urgency = "Immediate"
    elif fused >= 0.55:
        tier = "HIGH - PRIORITY CARDIOLOGY WORKUP"
        urgency = "Urgent (< 4h)"
    elif fused >= 0.35:
        tier = "MODERATE - OUTPATIENT STRESS ECHO / HOLTER"
        urgency = "Scheduled (24-72h)"
    else:
        tier = "LOW RISK - ROUTINE PRIMARY CARE"
        urgency = "Routine"

    return {
        "fused_risk_score": round(fused, 4),
        "triage_tier": tier,
        "recommended_urgency": urgency,
        "multimodal_modifiers": modifiers,
        "clinical_risk_drivers": drivers,
    }


def audit_patient_biomarkers(patient: Dict[str, Any]) -> Dict[str, Any]:
    """
    Audits a patient EHR dictionary against cardiology clinical thresholds,
    returning flagged out-of-range biomarkers and 9 engineered physiological features.
    """
    raw_row = {col: patient.get(col, 0) for col in RAW_CLINICAL_FEATURES}
    df_in = pd.DataFrame([raw_row])
    X_feat = engineer_clinical_features(df_in)
    engineered = {
        k: round(float(X_feat[k].iloc[0]), 4)
        for k in X_feat.columns
        if k not in RAW_CLINICAL_FEATURES
    }

    resting_bp = float(patient.get("resting_bp", 120) or 120)
    cholesterol = float(patient.get("cholesterol", 190) or 190)
    fasting_bs = int(patient.get("fasting_bs", 0) or 0)
    max_hr = float(patient.get("max_hr", 150) or 150)
    oldpeak = float(patient.get("oldpeak", 0.0) or 0.0)
    ca = int(patient.get("ca", 0) or 0)
    exercise_angina = int(patient.get("exercise_angina", 0) or 0)

    flags: List[Dict[str, Any]] = []
    if resting_bp >= 140:
        flags.append({
            "feature": "resting_bp",
            "value": resting_bp,
            "severity": "CRITICAL" if resting_bp >= 160 else "ELEVATED",
            "finding": "Stage 2 systolic hypertension (>= 140 mmHg)",
        })
    elif resting_bp > 120:
        flags.append({
            "feature": "resting_bp",
            "value": resting_bp,
            "severity": "BORDERLINE",
            "finding": "Elevated resting blood pressure (121-139 mmHg)",
        })

    if cholesterol >= 240:
        flags.append({
            "feature": "cholesterol",
            "value": cholesterol,
            "severity": "ELEVATED",
            "finding": "Hypercholesterolemia (>= 240 mg/dL)",
        })
    elif cholesterol > 200:
        flags.append({
            "feature": "cholesterol",
            "value": cholesterol,
            "severity": "BORDERLINE",
            "finding": "Borderline high total cholesterol (201-239 mg/dL)",
        })

    if fasting_bs == 1:
        flags.append({
            "feature": "fasting_bs",
            "value": fasting_bs,
            "severity": "ELEVATED",
            "finding": "Fasting blood glucose > 120 mg/dL",
        })

    if max_hr < 120:
        flags.append({
            "feature": "max_hr",
            "value": max_hr,
            "severity": "ELEVATED",
            "finding": "Chronotropic incompetence (peak HR < 120 bpm)",
        })

    if oldpeak >= 1.5:
        flags.append({
            "feature": "oldpeak",
            "value": oldpeak,
            "severity": "CRITICAL" if oldpeak >= 2.5 else "ELEVATED",
            "finding": "Significant exercise-induced ST depression (>= 1.5 mV)",
        })

    if ca >= 1:
        flags.append({
            "feature": "ca",
            "value": ca,
            "severity": "CRITICAL" if ca >= 2 else "ELEVATED",
            "finding": f"Fluoroscopic coronary vessel involvement ({ca} major vessels)",
        })

    if exercise_angina == 1:
        flags.append({
            "feature": "exercise_angina",
            "value": exercise_angina,
            "severity": "ELEVATED",
            "finding": "Positive exercise-induced angina pectoris",
        })

    overall_status = "NORMAL"
    if any(f["severity"] == "CRITICAL" for f in flags):
        overall_status = "CRITICAL_FINDINGS"
    elif any(f["severity"] == "ELEVATED" for f in flags):
        overall_status = "ELEVATED_FINDINGS"
    elif flags:
        overall_status = "BORDERLINE_FINDINGS"

    return {
        "overall_audit_status": overall_status,
        "abnormal_biomarker_count": len(flags),
        "flagged_biomarkers": flags,
        "engineered_biomarkers": engineered,
        "hemodynamic_indices": compute_hemodynamic_shock_index(patient),
    }


def compute_hemodynamic_shock_index(patient: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes Shock Index (HR / SBP), Age-Adjusted Shock Index (Age * SI),
    and Rate-Pressure Product (SBP * HR / 1000) to quantify myocardial oxygen demand.
    """
    age = float(patient.get("age", 50) or 50)
    resting_bp = max(float(patient.get("resting_bp", 120) or 120), 1.0)
    max_hr = float(patient.get("max_hr", 150) or 150)

    shock_index = round(max_hr / resting_bp, 4)
    age_shock_index = round(age * shock_index, 2)
    rpp = round((resting_bp * max_hr) / 1000.0, 2)

    if rpp >= 22.0 or age_shock_index >= 75.0:
        stability = "CRITICAL_MYOCARDIAL_DEMAND"
    elif rpp >= 18.0 or age_shock_index >= 55.0:
        stability = "ELEVATED_HEMODYNAMIC_STRESS"
    else:
        stability = "HEMODYNAMICALLY_STABLE"

    return {
        "shock_index": shock_index,
        "age_adjusted_shock_index": age_shock_index,
        "rate_pressure_product_k": rpp,
        "hemodynamic_stability": stability,
    }


