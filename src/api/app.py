"""
FastAPI Clinical Decision Support REST API Application (v2.0)
--------------------------------------------------------------
Serves real-time multi-modal deep learning inference, empirical benchmarks,
real Hugging Face medical datasets & pre-trained Hub checkpoints, and the
React + Vite PACS Workstation UI.
"""

import os
import glob
import json
import base64
from io import BytesIO
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import joblib
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.vision.model import ResidualChestCNN, TransferChestClassifier
from src.vision.gradcam import GradCAM, encode_gradcam_artifacts
from src.vision.uncertainty import MonteCarloDropoutEstimator
from src.vision.radiology_tools import compute_consolidation_severity, extract_radiomic_features
from src.ecg.model import Conv1dAutoencoder, ECGResNet1D
from src.ecg.anomaly_detector import ECGAnomalyDetector
from src.ecg.telemetry import analyze_ecg_telemetry
from src.clinical.features import engineer_clinical_features, compute_multimodal_triage_score, RAW_CLINICAL_FEATURES
from src.pretrained_hub import (
    get_curated_catalog,
    search_hf_medical_models,
    infer_pretrained_vision,
    infer_pretrained_ecg,
    infer_pretrained_clinical,
)
from src.api.schemas import (
    HealthResponse,
    ModelsMetadataResponse,
    ModelInfo,
    VisionDiagnosisResponse,
    UncertaintyDetails,
    SeverityDetails,
    HotspotPoint,
    ECGScreenRequest,
    ECGScreenResponse,
    ECGTelemetry,
    ClinicalTriageRequest,
    ClinicalTriageResponse,
    BatchEvaluateRequest,
    PretrainedECGRequest,
)

app = FastAPI(
    title="Deep Health AI - Clinical Decision Support Microservice v2.0",
    description="Enterprise REST API & Real-Time PACS Workstation across Thoracic Radiology, 1D ECG Electrophysiology, Clinical EHR Cardiology Triage, and Hugging Face Medical Hub.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Lazy Model Caches
_vision_models: Dict[str, torch.nn.Module] = {}
_ecg_detector: Optional[ECGAnomalyDetector] = None
_ecg_resnet: Optional[ECGResNet1D] = None
_ecg_mitbih_resnet: Optional[ECGResNet1D] = None
_clinical_bundle: Optional[Dict[str, Any]] = None
_default_ecg_threshold: float = 0.01037


def get_vision_model(model_id: str = "residual_cnn") -> torch.nn.Module:
    if model_id in _vision_models:
        return _vision_models[model_id]

    if model_id == "resnet18":
        model = TransferChestClassifier(arch="resnet18", num_classes=2, pretrained=False)
        weights_path = "saved_models/vision_resnet18.pt"
    elif model_id == "mobilenet_v2":
        model = TransferChestClassifier(arch="mobilenet_v2", num_classes=2, pretrained=False)
        weights_path = "saved_models/vision_mobilenetv2.pt"
    else:
        model = ResidualChestCNN(in_channels=1, num_classes=2)
        weights_path = "saved_models/best_chest_cnn.pt"

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=torch.device(DEVICE), weights_only=True))
    model.to(DEVICE)
    model.eval()
    _vision_models[model_id] = model
    return model


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


def get_ecg_resnet() -> Optional[ECGResNet1D]:
    global _ecg_resnet
    if _ecg_resnet is None and os.path.exists("saved_models/ecg_resnet1d.pt"):
        m = ECGResNet1D(in_channels=1, num_classes=5)
        m.load_state_dict(torch.load("saved_models/ecg_resnet1d.pt", map_location=torch.device(DEVICE), weights_only=True))
        m.to(DEVICE)
        m.eval()
        _ecg_resnet = m
    return _ecg_resnet


def get_ecg_mitbih_resnet() -> Optional[ECGResNet1D]:
    global _ecg_mitbih_resnet
    if _ecg_mitbih_resnet is None and os.path.exists("saved_models/ecg_mitbih_resnet1d.pt"):
        m = ECGResNet1D(in_channels=1, num_classes=5)
        m.load_state_dict(torch.load("saved_models/ecg_mitbih_resnet1d.pt", map_location=torch.device(DEVICE), weights_only=True))
        m.to(DEVICE)
        m.eval()
        _ecg_mitbih_resnet = m
    return _ecg_mitbih_resnet


def get_clinical_bundle() -> Optional[Dict[str, Any]]:
    global _clinical_bundle
    if _clinical_bundle is None and os.path.exists("saved_models/best_clinical_triage.joblib"):
        _clinical_bundle = joblib.load("saved_models/best_clinical_triage.joblib")
    return _clinical_bundle


def _pil_to_data_uri(pil_img: Image.Image) -> str:
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@app.get("/health", response_model=HealthResponse, tags=["System Telemetry"])
def health_check():
    """Returns microservice health status, compute hardware, and loaded model inventory."""
    return HealthResponse(
        status="HEALTHY",
        service="Deep Health AI Clinical Inference Microservice",
        version="2.0.0",
        device=DEVICE,
        models_loaded={
            "ResidualChestCNN": os.path.exists("saved_models/best_chest_cnn.pt"),
            "TransferResNet18": os.path.exists("saved_models/vision_resnet18.pt"),
            "TransferMobileNetV2": os.path.exists("saved_models/vision_mobilenetv2.pt"),
            "RadiomicLightGBM": os.path.exists("saved_models/vision_radiomic_lgbm.joblib"),
            "Conv1dAutoencoder": os.path.exists("saved_models/best_ecg_autoencoder.pt"),
            "ECGResNet1D": os.path.exists("saved_models/ecg_resnet1d.pt"),
            "ECGResNet1D_MITBIH": os.path.exists("saved_models/ecg_mitbih_resnet1d.pt"),
            "ClinicalTriageSuite": os.path.exists("saved_models/best_clinical_triage.joblib"),
        },
    )


@app.get("/v1/models", response_model=ModelsMetadataResponse, tags=["Model Governance"])
def get_models_metadata():
    """Inspects loaded neural networks, parameter counts, and available v2.0 model selectors."""
    v_model = get_vision_model("residual_cnn")
    e_detector = get_ecg_detector()
    clin = get_clinical_bundle()

    v_params = sum(p.numel() for p in v_model.parameters() if p.requires_grad)
    e_params = sum(p.numel() for p in e_detector.model.parameters() if p.requires_grad)

    avail_v = ["residual_cnn"]
    if os.path.exists("saved_models/vision_resnet18.pt"):
        avail_v.append("resnet18")
    if os.path.exists("saved_models/vision_mobilenetv2.pt"):
        avail_v.append("mobilenet_v2")

    avail_e = ["autoencoder"]
    if os.path.exists("saved_models/ecg_resnet1d.pt"):
        avail_e.append("resnet1d")
    if os.path.exists("saved_models/ecg_mitbih_resnet1d.pt"):
        avail_e.append("mitbih_resnet1d")

    return ModelsMetadataResponse(
        vision=ModelInfo(
            name="ResidualChestCNN + SpatialAttention",
            architecture="Deep Residual CNN + Transfer ResNet18 / MobileNetV2 + Radiomic-14 LightGBM",
            trainable_parameters=v_params,
            input_shape=[1, 1, 128, 128],
            status="READY_IN_MEMORY",
        ),
        ecg=ModelInfo(
            name="Conv1dAutoencoder + ECGResNet1D",
            architecture="1D Convolutional Autoencoder & 5-Class Supervised 1D-ResNet",
            trainable_parameters=e_params,
            input_shape=[1, 1, 256],
            status="READY_IN_MEMORY",
        ),
        clinical=ModelInfo(
            name=f"ClinicalTriage ({clin['best_model_name']})" if clin else "ClinicalTriage",
            architecture="Leak-Free 20-Biomarker Cardiology Ensemble (LGBM/RF/ExtraTrees/Stacking)",
            trainable_parameters=20,
            input_shape=[1, 20],
            status="READY_IN_MEMORY" if clin else "NOT_TRAINED",
        ),
        available_vision_models=avail_v,
        available_ecg_models=avail_e,
    )


@app.post("/v1/vision/diagnose", response_model=VisionDiagnosisResponse, tags=["Computer Vision Diagnostics"])
async def diagnose_chest_radiograph(
    file: UploadFile = File(..., description="Chest radiograph image (PNG/JPG)"),
    mc_samples: int = Query(20, ge=5, le=50, description="Monte Carlo Dropout passes for uncertainty quantification"),
    model_id: str = Query("residual_cnn", description="residual_cnn | resnet18 | mobilenet_v2"),
    colormap: str = Query("turbo", description="turbo | jet | plasma | inferno"),
):
    """Diagnoses a chest radiograph and returns true Grad-CAM RGBA PNG overlays + 14 radiomic biomarkers."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {file.content_type}. Expected an image (image/png, image/jpeg).",
        )

    try:
        contents = await file.read()
        pil_img = Image.open(BytesIO(contents)).convert("L")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to decode image: {e}")

    model = get_vision_model(model_id)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])
    tensor_input = transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor_input)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))

    class_names = ["Normal Healthy Parenchyma", "Alveolar Consolidation / Pneumonia"]
    diagnosis_str = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    mc_estimator = MonteCarloDropoutEstimator(model, n_samples=mc_samples, device=DEVICE)
    uncert = mc_estimator.estimate_uncertainty(tensor_input)

    target_layer = getattr(model, "gradcam_layer", getattr(model, "layer4", None))
    gradcam = GradCAM(model, target_layer)
    heatmap, _, _ = gradcam.generate_heatmap(tensor_input, target_class=pred_idx)
    gradcam.remove_hooks()

    cmap_safe = colormap if colormap in ("turbo", "jet", "plasma", "inferno") else "turbo"
    artifacts = encode_gradcam_artifacts(pil_img, heatmap, alpha=0.55, colormap_name=cmap_safe)
    severity_info = compute_consolidation_severity(heatmap, threshold=0.40)
    radiomics = extract_radiomic_features(np.asarray(pil_img.resize((128, 128)), dtype=np.float32))

    model_label_map = {
        "residual_cnn": "ResidualChestCNN + SpatialAttention",
        "resnet18": "TransferResNet18",
        "mobilenet_v2": "TransferMobileNetV2",
    }

    return VisionDiagnosisResponse(
        filename=file.filename or "uploaded_radiograph.png",
        model_used=model_label_map.get(model_id, "ResidualChestCNN"),
        diagnosis=diagnosis_str,
        confidence=round(confidence, 4),
        normal_probability=round(float(probs[0]), 4),
        pneumonia_probability=round(float(probs[1]), 4),
        uncertainty=UncertaintyDetails(
            predictive_variance=round(uncert["predictive_variance"], 6),
            normalized_entropy=round(uncert["normalized_entropy"], 4),
            uncertainty_risk_level=uncert["uncertainty_risk_level"],
            human_referral_mandatory=uncert["human_referral_mandatory"],
            monte_carlo_samples=mc_samples,
        ),
        severity=SeverityDetails(
            radiological_severity=severity_info["radiological_severity"],
            radiological_grade=severity_info["radiological_grade"],
            infiltrate_burden_percentage=severity_info["infiltrate_burden_percentage"],
            clinical_triage_urgency=severity_info["clinical_triage_urgency"],
        ),
        gradcam_available=True,
        gradcam_heatmap_base64=artifacts["heatmap_base64"],
        gradcam_overlay_base64=artifacts["overlay_base64"],
        hotspot=HotspotPoint(**artifacts["hotspot"]),
        radiomics=radiomics,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/v1/ecg/screen", response_model=ECGScreenResponse, tags=["Electrophysiological Screening"])
def screen_ecg_signal(payload: ECGScreenRequest):
    """
    Analyzes a 1D Lead-II / MIT-BIH ECG waveform using the trained Conv1dAutoencoder
    and 5-class ECGResNet1D, returning real reconstructed signal, residuals, 5-class probabilities,
    and 18 electrophysiological/spectral biomarkers.
    """
    raw_signal = np.array(payload.signal, dtype=np.float32)
    if len(raw_signal) < 32:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ECG signal too short. Minimum 32 samples required.",
        )

    if len(raw_signal) > 256:
        processed_signal = raw_signal[:256]
    elif len(raw_signal) < 256:
        processed_signal = np.pad(raw_signal, (0, 256 - len(raw_signal)), mode="edge")
    else:
        processed_signal = raw_signal

    detector = get_ecg_detector()
    detector.threshold = payload.threshold_override if payload.threshold_override is not None else _default_ecg_threshold

    res = detector.predict(processed_signal)
    recon_arr = np.asarray(res["reconstructed_signal"], dtype=np.float32).ravel()
    residuals = ((processed_signal - recon_arr) ** 2).astype(np.float32)
    telemetry_data = analyze_ecg_telemetry(processed_signal, sampling_rate=payload.sampling_rate_hz)

    rhythm_subtype = "Normal Sinus Rhythm" if not res["is_anomaly"] else "Ventricular / Conduction Anomaly"
    subtype_probs: Dict[str, float] = {}
    model_used = "Conv1dAutoencoder"

    if payload.model_id == "mitbih_resnet1d" and get_ecg_mitbih_resnet() is not None:
        m_net = get_ecg_mitbih_resnet()
        # Resample input to 180 if longer
        x_in = np.interp(np.linspace(0, 1, 180), np.linspace(0, 1, len(raw_signal)), raw_signal).astype(np.float32)
        with torch.no_grad():
            p = F.softmax(m_net(torch.from_numpy(x_in).view(1, 1, -1).to(DEVICE)), dim=1).cpu().numpy()[0]
        mit_names = ["Atrial Premature (A)", "LBBB (L)", "Normal Sinus (N)", "RBBB (R)", "PVC (V)"]
        subtype_probs = {mit_names[i]: round(float(p[i]), 4) for i in range(5)}
        best_i = int(np.argmax(p))
        rhythm_subtype = mit_names[best_i]
        model_used = "ECGResNet1D (MIT-BIH) + Conv1dAutoencoder"
    elif get_ecg_resnet() is not None:
        rn = get_ecg_resnet()
        with torch.no_grad():
            p = F.softmax(rn(torch.from_numpy(processed_signal).view(1, 1, -1).to(DEVICE)), dim=1).cpu().numpy()[0]
        r_names = ["Normal Sinus (NSR)", "PVC Ectopy", "Atrial Fibrillation", "ST-Elevation Ischemia", "LBBB"]
        subtype_probs = {r_names[i]: round(float(p[i]), 4) for i in range(5)}
        rhythm_subtype = r_names[int(np.argmax(p))]
        model_used = "ECGResNet1D + Conv1dAutoencoder"

    return ECGScreenResponse(
        model_used=model_used,
        diagnosis=res["diagnosis"],
        is_anomaly=res["is_anomaly"],
        reconstruction_mse=round(res["reconstruction_error"], 6),
        decision_threshold=round(detector.threshold, 6),
        rhythm_subtype=rhythm_subtype,
        subtype_probabilities=subtype_probs,
        reconstructed_signal=[round(float(v), 5) for v in recon_arr.tolist()],
        residual_profile=[round(float(v), 6) for v in residuals.tolist()],
        telemetry=ECGTelemetry(**telemetry_data),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/v1/benchmarks", tags=["Model Governance"])
def get_benchmarks_summary():
    """Returns real persisted training & validation metrics across Vision, ECG, Clinical EHR & HF Hub."""
    files = {
        "summary": "reports/metrics_summary.json",
        "vision": "reports/vision_metrics.json",
        "ecg": "reports/ecg_metrics.json",
        "clinical": "reports/clinical_metrics.json",
        "pretrained": "reports/pretrained_benchmarks.json",
        "metadata": "data/metadata.json",
    }
    out: Dict[str, Any] = {}
    for k, p in files.items():
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    out[k] = json.load(f)
            except Exception:
                out[k] = None
        else:
            out[k] = None
    return out


@app.get("/v1/dataset/samples", tags=["Clinical Cohorts"])
def get_dataset_samples():
    """Serves real cached clinical samples (PneumoniaMNIST X-rays, MIT-BIH/Lead-II ECGs, Cardiology patients)."""
    vision_samples = []
    manifest_csv = "data/external/pneumoniamnist_manifest.csv"
    if os.path.exists(manifest_csv):
        df_m = pd.read_csv(manifest_csv)
        df_te = df_m[df_m["source_split"] == "test_real"]
        for lbl_val, title_prefix in [(1, "Real PneumoniaMNIST Consolidation"), (0, "Real PneumoniaMNIST Normal")]:
            sub = df_te[df_te["label"] == lbl_val].head(2)
            for idx_row, (_, r) in enumerate(sub.iterrows()):
                if os.path.exists(r["relative_path"]):
                    pil = Image.open(r["relative_path"]).convert("L")
                    vision_samples.append({
                        "id": f"pmnist_{lbl_val}_{idx_row}",
                        "title": f"{title_prefix} ({r['sample_id']})",
                        "source": "danjacobellis/pneumoniamnist_224 (Test Split)",
                        "ground_truth": r["class_name"],
                        "image_base64": _pil_to_data_uri(pil),
                    })

    for folder, cls_lbl in [("pneumonia", "Pneumonia"), ("normal", "Normal")]:
        paths = sorted(glob.glob(f"data/vision/{folder}/*.png"))
        if paths:
            pil = Image.open(paths[0]).convert("L")
            vision_samples.append({
                "id": f"gen_{folder}",
                "title": f"Thoracic CXR: {cls_lbl} ({os.path.basename(paths[0])})",
                "source": "High-Variance Thoracic Simulator",
                "ground_truth": cls_lbl,
                "image_base64": _pil_to_data_uri(pil),
            })

    ecg_samples = []
    if os.path.exists("data/ecg/ecg_test.npy") and os.path.exists("data/ecg/ecg_test_multiclass.npy"):
        X_e = np.load("data/ecg/ecg_test.npy")
        y_m = np.load("data/ecg/ecg_test_multiclass.npy")
        names = ["Normal Sinus Rhythm (NSR)", "Premature Ventricular Contraction (PVC)", "Atrial Fibrillation (AFib)", "Acute ST-Elevation Ischemia", "Left Bundle Branch Block (LBBB)"]
        for cid, cname in enumerate(names):
            matches = np.where(y_m == cid)[0]
            if len(matches) > 0:
                ecg_samples.append({
                    "id": f"leadii_{cid}",
                    "title": cname,
                    "source": "Lead-II 250Hz Cohort",
                    "sampling_rate_hz": 250,
                    "signal": [round(float(x), 5) for x in X_e[int(matches[0])].tolist()],
                })

    if os.path.exists("data/external/mitbih_10k.npz"):
        loaded = np.load("data/external/mitbih_10k.npz", allow_pickle=True)
        Xm, ym = loaded["X"], loaded["y"]
        mit_labels = {"N": "MIT-BIH Real Normal (N)", "V": "MIT-BIH Real PVC Ectopy (V)", "L": "MIT-BIH Real LBBB (L)"}
        classes = [str(c) for c in loaded["classes"].tolist()]
        for code, title in mit_labels.items():
            if code in classes:
                cidx = classes.index(code)
                matches = np.where(ym == cidx)[0]
                if len(matches) > 0:
                    ecg_samples.append({
                        "id": f"mitbih_{code}",
                        "title": title,
                        "source": "PhysioNet MIT-BIH 10k Real Beat",
                        "sampling_rate_hz": 180,
                        "signal": [round(float(x), 5) for x in Xm[int(matches[0])].tolist()],
                    })

    clinical_presets = []
    if os.path.exists("data/external/cardiology_triage.csv"):
        df_c = pd.read_csv("data/external/cardiology_triage.csv").dropna(subset=["resting_bp", "cholesterol"])
        neg_row = df_c[df_c["target"] == 0].iloc[0]
        pos_row = df_c[(df_c["target"] == 1) & (df_c["oldpeak"] >= 1.8)].iloc[0]
        mid_row = df_c[(df_c["target"] == 1) & (df_c["age"].between(48, 56))].iloc[0]
        for pid, pname, r in [
            ("low_risk", f"Patient A: Asymptomatic Routine ({int(neg_row['age'])}y)", neg_row),
            ("moderate_risk", f"Patient B: Exertional Angina ({int(mid_row['age'])}y)", mid_row),
            ("critical_risk", f"Patient C: Severe Ischemic Depression ({int(pos_row['age'])}y)", pos_row),
        ]:
            record = {col: (float(r[col]) if col == "oldpeak" else int(round(float(r[col])))) for col in RAW_CLINICAL_FEATURES}
            clinical_presets.append({
                "id": pid,
                "title": pname,
                "source": str(r.get("cohort_source", "UCI Cardiology")),
                "true_outcome": int(r["target"]),
                "features": record,
            })

    return {
        "vision_samples": vision_samples,
        "ecg_samples": ecg_samples,
        "clinical_presets": clinical_presets,
    }


@app.post("/v1/clinical/triage", response_model=ClinicalTriageResponse, tags=["Cardiology EHR Triage"])
def run_clinical_triage(payload: ClinicalTriageRequest):
    """Runs Cardiology EHR 20-biomarker ensemble triage + multimodal CXR/ECG fusion + HF consensus."""
    bundle = get_clinical_bundle()
    if bundle is None:
        raise HTTPException(status_code=503, detail="Clinical triage model not trained yet. Run train_clinical.py.")

    patient_dict = {col: getattr(payload, col) for col in RAW_CLINICAL_FEATURES}
    df_in = pd.DataFrame([patient_dict])
    X_feat = engineer_clinical_features(df_in)
    prob = float(bundle["pipeline"].predict_proba(X_feat)[0, 1])

    fusion = compute_multimodal_triage_score(
        ehr_cad_probability=prob,
        patient=patient_dict,
        cxr_pneumonia_probability=payload.cxr_pneumonia_probability,
        ecg_is_anomaly=payload.ecg_is_anomaly,
        ecg_reconstruction_mse=payload.ecg_reconstruction_mse,
    )

    hf_res = None
    try:
        hf_res = infer_pretrained_clinical(patient_dict)
    except Exception:
        hf_res = None

    engineered_dict = {k: round(float(X_feat[k].iloc[0]), 4) for k in X_feat.columns if k not in RAW_CLINICAL_FEATURES}

    return ClinicalTriageResponse(
        model_used=f"{bundle.get('best_model_name', 'StackingEnsemble')} (20 Clinical Biomarkers)",
        cad_risk_probability=round(prob, 4),
        predicted_label="High Coronary / Cardiac Risk" if prob >= 0.5 else "Low Coronary Risk",
        fused_risk_score=fusion["fused_risk_score"],
        triage_tier=fusion["triage_tier"],
        recommended_urgency=fusion["recommended_urgency"],
        multimodal_modifiers=fusion["multimodal_modifiers"],
        clinical_risk_drivers=fusion["clinical_risk_drivers"],
        engineered_biomarkers=engineered_dict,
        hf_consensus=hf_res,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/v1/batch/evaluate", tags=["Cohort Epidemiology"])
def evaluate_live_cohort(req: BatchEvaluateRequest):
    """Executes real batch inference across held-out patient cohorts (never mocked)."""
    from batch_processor import process_vision_cohort, process_ecg_cohort, process_clinical_cohort

    if req.modality == "vision_real":
        manifest = "data/external/pneumoniamnist_manifest.csv"
        df_m = pd.read_csv(manifest)
        paths = df_m[df_m["source_split"] == "test_real"]["relative_path"].tolist()[: req.limit]
        res = process_vision_cohort(paths, device=DEVICE)
    elif req.modality == "vision":
        norm = sorted(glob.glob("data/vision/normal/*.png"))[: max(2, req.limit // 2)]
        pneu = sorted(glob.glob("data/vision/pneumonia/*.png"))[: max(2, req.limit - len(norm))]
        res = process_vision_cohort(pneu + norm, device=DEVICE)
    elif req.modality == "ecg":
        sigs = np.load("data/ecg/ecg_test.npy")[: req.limit]
        res = process_ecg_cohort(sigs, device=DEVICE)
    elif req.modality == "clinical":
        df_c = pd.read_csv("data/external/cardiology_triage.csv").tail(req.limit)
        res = process_clinical_cohort(df_c)
    else:
        raise HTTPException(status_code=400, detail="modality must be vision | vision_real | ecg | clinical")

    df_out = res.pop("results_dataframe")
    records = df_out.replace({np.nan: None}).to_dict(orient="records")
    return {"summary": res, "records": records}


@app.get("/v1/pretrained/catalog", tags=["Hugging Face Pre-Trained Hub"])
def pretrained_catalog():
    """Lists verified Hugging Face pre-trained medical models and their empirical test benchmarks."""
    bench = None
    if os.path.exists("reports/pretrained_benchmarks.json"):
        with open("reports/pretrained_benchmarks.json", "r") as f:
            bench = json.load(f)
    return {"catalog": get_curated_catalog(), "benchmarks": bench}


@app.get("/v1/pretrained/search", tags=["Hugging Face Pre-Trained Hub"])
def pretrained_search(q: str = Query("chest xray pneumonia", min_length=2), limit: int = Query(10, ge=1, le=20)):
    """Searches live Hugging Face Hub API for public clinical AI checkpoints."""
    return {"query": q, "results": search_hf_medical_models(query=q, limit=limit)}


@app.post("/v1/pretrained/vision", tags=["Hugging Face Pre-Trained Hub"])
async def pretrained_vision_infer(
    file: UploadFile = File(...),
    repo_id: str = Query("Purino/mobilenetv2-chest-xray-pneumonia-256-fullframe-gray"),
):
    contents = await file.read()
    pil_img = Image.open(BytesIO(contents)).convert("L")
    res = infer_pretrained_vision(pil_img, repo_id=repo_id, device=DEVICE)
    return res


@app.post("/v1/pretrained/ecg", tags=["Hugging Face Pre-Trained Hub"])
def pretrained_ecg_infer(payload: PretrainedECGRequest):
    sig = np.asarray(payload.signal, dtype=np.float32)
    return infer_pretrained_ecg(sig, repo_id=payload.repo_id, device=DEVICE)


@app.post("/v1/pretrained/clinical", tags=["Hugging Face Pre-Trained Hub"])
def pretrained_clinical_infer(payload: ClinicalTriageRequest):
    patient_dict = {col: getattr(payload, col) for col in RAW_CLINICAL_FEATURES}
    return infer_pretrained_clinical(patient_dict)


@app.get("/v1/system/diagnostics", tags=["System Operations"])
def system_diagnostics():
    """Returns real-time checkpoint readiness, artifact sizes, and dataset inventory."""
    checkpoints = [
        "saved_models/best_chest_cnn.pt",
        "saved_models/vision_resnet18.pt",
        "saved_models/vision_mobilenetv2.pt",
        "saved_models/vision_radiomic_lgbm.joblib",
        "saved_models/best_ecg_autoencoder.pt",
        "saved_models/ecg_resnet1d.pt",
        "saved_models/ecg_mitbih_resnet1d.pt",
        "saved_models/ecg_lgbm.joblib",
        "saved_models/best_clinical_triage.joblib",
    ]
    checkpoint_inventory = []
    for ckpt in checkpoints:
        exists = os.path.exists(ckpt)
        size_kb = round(os.path.getsize(ckpt) / 1024.0, 2) if exists else 0.0
        checkpoint_inventory.append({
            "path": ckpt,
            "available": exists,
            "size_kb": size_kb,
        })

    available_count = sum(1 for c in checkpoint_inventory if c["available"])
    return {
        "status": "READY" if available_count > 0 else "DEGRADED",
        "device": DEVICE,
        "torch_version": torch.__version__,
        "checkpoints_available": available_count,
        "checkpoints_total": len(checkpoint_inventory),
        "checkpoints": checkpoint_inventory,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# Mount Production React PACS Frontend if built
frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist"))
if os.path.exists(frontend_dist):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
