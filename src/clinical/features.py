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


def compute_cardiovascular_risk_profile(patient: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes an integrated cardiovascular risk stratification profile including
    estimated vascular age, atherogenic metabolic burden, myocardial ischemic burden,
    and actionable secondary-prevention targets.
    """
    age = float(patient.get("age", 50) or 50)
    resting_bp = float(patient.get("resting_bp", 120) or 120)
    cholesterol = float(patient.get("cholesterol", 190) or 190)
    fasting_bs = int(patient.get("fasting_bs", 0) or 0)
    max_hr = float(patient.get("max_hr", 150) or 150)
    oldpeak = float(patient.get("oldpeak", 0.0) or 0.0)
    ca = int(patient.get("ca", 0) or 0)
    exercise_angina = int(patient.get("exercise_angina", 0) or 0)

    bp_penalty = max(0.0, (resting_bp - 120.0) * 0.25)
    lipid_penalty = max(0.0, (cholesterol - 200.0) * 0.12)
    glycemic_penalty = 4.5 if fasting_bs == 1 else 0.0
    ischemic_penalty = oldpeak * 3.0 + ca * 4.0 + (3.5 if exercise_angina == 1 else 0.0)

    estimated_vascular_age = round(age + bp_penalty + lipid_penalty + glycemic_penalty + ischemic_penalty, 1)
    vascular_age_delta = round(estimated_vascular_age - age, 1)

    predicted_max_hr = max(220.0 - age, 100.0)
    chronotropic_reserve_pct = round((max_hr / predicted_max_hr) * 100.0, 1)

    recommendations: List[str] = []
    if resting_bp >= 130:
        recommendations.append("Titrate antihypertensive regimen toward < 120/80 mmHg systolic target.")
    if cholesterol >= 200:
        recommendations.append("Evaluate high-intensity statin therapy and lipid panel fractionation.")
    if fasting_bs == 1:
        recommendations.append("Order HbA1c and initiate glycemic control protocol.")
    if oldpeak >= 1.0 or exercise_angina == 1 or ca >= 1:
        recommendations.append("Schedule stress myocardial perfusion imaging or coronary CT angiography.")
    if not recommendations:
        recommendations.append("Maintain preventive aerobic conditioning and annual lipid/BP surveillance.")

    if vascular_age_delta >= 15.0 or ca >= 2 or oldpeak >= 2.0:
        risk_category = "VERY_HIGH_CARDIOVASCULAR_RISK"
    elif vascular_age_delta >= 7.0 or resting_bp >= 140 or cholesterol >= 240:
        risk_category = "ELEVATED_CARDIOVASCULAR_RISK"
    else:
        risk_category = "LOW_TO_MODERATE_RISK"

    return {
        "chronological_age": int(round(age)),
        "estimated_vascular_age": estimated_vascular_age,
        "vascular_age_acceleration_years": vascular_age_delta,
        "chronotropic_reserve_pct": chronotropic_reserve_pct,
        "risk_category": risk_category,
        "preventive_recommendations": recommendations,
    }


def compute_coronary_ischemic_burden(patient: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes an integrated coronary ischemic burden score and Duke Treadmill Score
    approximation from exercise ST depression (oldpeak), ST slope, exercise angina,
    fluoroscopic coronary vessel involvement (ca), and chest pain acuity.
    """
    oldpeak = max(0.0, float(patient.get("oldpeak", 0.0) or 0.0))
    st_slope = int(patient.get("st_slope", 1) or 1)
    exercise_angina = int(patient.get("exercise_angina", 0) or 0)
    ca = max(0, int(patient.get("ca", 0) or 0))
    chest_pain_type = int(patient.get("chest_pain_type", 0) or 0)
    max_hr = float(patient.get("max_hr", 150) or 150)

    # Approximate exercise tolerance duration (minutes) from peak HR and angina status
    estimated_exercise_minutes = round(max(3.0, min(12.0, (max_hr - 90.0) / 10.0 - (1.5 if exercise_angina == 1 else 0.0))), 2)
    angina_index = 2 if (exercise_angina == 1 and oldpeak >= 1.5) else (1 if exercise_angina == 1 else 0)
    duke_treadmill_score = round(estimated_exercise_minutes - (5.0 * oldpeak) - (4.0 * angina_index), 2)

    slope_weight = 1.4 if st_slope >= 2 else (1.15 if st_slope == 1 else 0.85)
    ischemic_burden_index = round(
        oldpeak * 2.5 * slope_weight
        + ca * 3.0
        + (2.5 if exercise_angina == 1 else 0.0)
        + (1.2 if chest_pain_type >= 2 else 0.0),
        2,
    )

    if duke_treadmill_score <= -11.0 or ischemic_burden_index >= 10.0 or ca >= 2:
        severity_tier = "SEVERE_MYOCARDIAL_ISCHEMIA"
        workup = "Urgent invasive coronary angiography or stress myocardial perfusion imaging."
    elif duke_treadmill_score < 5.0 or ischemic_burden_index >= 4.5 or ca == 1:
        severity_tier = "MODERATE_ISCHEMIC_BURDEN"
        workup = "Coronary CT angiography (CCTA) and anti-ischemic beta-blocker/statin optimization."
    else:
        severity_tier = "LOW_ISCHEMIC_RISK"
        workup = "Routine outpatient cardiology follow-up and primary lifestyle prevention."

    return {
        "duke_treadmill_score": duke_treadmill_score,
        "estimated_exercise_minutes": estimated_exercise_minutes,
        "angina_index": angina_index,
        "ischemic_burden_index": ischemic_burden_index,
        "fluoroscopic_vessels_involved": ca,
        "ischemia_severity_tier": severity_tier,
        "recommended_workup": workup,
    }


def compute_cardiometabolic_syndrome_index(patient: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes a cardiometabolic syndrome risk index from systolic blood pressure,
    total cholesterol, fasting glycemic status, age, and coronary reserve markers.
    """
    age = float(patient.get("age", 50) or 50)
    resting_bp = float(patient.get("resting_bp", 120) or 120)
    cholesterol = float(patient.get("cholesterol", 190) or 190)
    fasting_bs = int(patient.get("fasting_bs", 0) or 0)
    max_hr = float(patient.get("max_hr", 150) or 150)

    risk_factors = 0
    if resting_bp >= 130.0:
        risk_factors += 1
    if cholesterol >= 200.0:
        risk_factors += 1
    if fasting_bs == 1:
        risk_factors += 1
    if max_hr < 130.0:
        risk_factors += 1

    atherogenic_ratio_est = round(cholesterol / 45.0, 2)
    bp_excess = max(0.0, (resting_bp - 120.0) / 10.0)
    chol_excess = max(0.0, (cholesterol - 180.0) / 25.0)
    glycemic_weight = 2.5 if fasting_bs == 1 else 0.0
    age_weight = max(0.0, (age - 45.0) / 15.0)

    cardiometabolic_score = round(bp_excess * 1.4 + chol_excess * 1.3 + glycemic_weight + age_weight, 2)

    if risk_factors >= 3 or cardiometabolic_score >= 7.5:
        syndrome_tier = "HIGH_CARDIOMETABOLIC_RISK"
        guidance = "Initiate intensive lipid-lowering, antihypertensive, and glycemic management protocol."
    elif risk_factors == 2 or cardiometabolic_score >= 4.0:
        syndrome_tier = "MODERATE_METABOLIC_STRAIN"
        guidance = "Recommend structured metabolic lifestyle intervention and quarterly lipid/HbA1c monitoring."
    else:
        syndrome_tier = "OPTIMAL_METABOLIC_PROFILE"
        guidance = "Maintain current preventive nutrition and annual cardiovascular screening."

    return {
        "metabolic_risk_factors_count": risk_factors,
        "atherogenic_lipid_ratio_est": atherogenic_ratio_est,
        "cardiometabolic_index_score": cardiometabolic_score,
        "syndrome_tier": syndrome_tier,
        "clinical_guidance": guidance,
    }





