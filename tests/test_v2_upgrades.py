"""
Comprehensive Unit & Integration Tests for Deep Health AI v2.0 Upgrades
-----------------------------------------------------------------------
Covers:
1. 14-Biomarker Radiomics & TransferChestClassifier (.gradcam_layer + GradCAM RGBA)
2. 18-Biomarker ECG Telemetry & Supervised 5-Class ECGResNet1D
3. 20-Biomarker Clinical EHR Feature Engineering & Multimodal Triage Fusion
4. Hugging Face Pre-Trained Medical Hub Catalog
5. v2.0 FastAPI REST Endpoints (/v1/benchmarks, /v1/dataset/samples, /v1/vision/diagnose,
   /v1/ecg/screen, /v1/clinical/triage, /v1/batch/evaluate, /v1/pretrained/catalog)
"""

from io import BytesIO
import os
import math
import numpy as np
import pandas as pd
import torch
from PIL import Image
from fastapi.testclient import TestClient

from src.vision.radiology_tools import RADIOMIC_FEATURE_NAMES, extract_radiomic_features
from src.vision.model import TransferChestClassifier
from src.vision.gradcam import GradCAM, encode_gradcam_artifacts
from src.ecg.telemetry import ECG_FEATURE_NAMES, analyze_ecg_telemetry, extract_ecg_feature_matrix
from src.ecg.model import ECGResNet1D
from src.clinical.features import (
    CLINICAL_FEATURE_NAMES,
    RAW_CLINICAL_FEATURES,
    engineer_clinical_features,
    compute_multimodal_triage_score,
)
from src.pretrained_hub import get_curated_catalog
from src.api.app import app

client = TestClient(app)


def test_radiomic_features_14_biomarkers():
    img = (np.random.rand(128, 128) * 255).astype(np.uint8)
    feats = extract_radiomic_features(img)
    assert set(feats.keys()) == set(RADIOMIC_FEATURE_NAMES)
    assert len(feats) == 14
    for val in feats.values():
        assert isinstance(val, float)
        assert math.isfinite(val)


def test_transfer_chest_classifier_and_gradcam():
    model = TransferChestClassifier(arch="resnet18", num_classes=2, pretrained=False, freeze_stem=True)
    x = torch.randn(1, 1, 128, 128)
    logits = model(x)
    assert logits.shape == (1, 2)
    assert model.gradcam_layer is not None

    cam = GradCAM(model, model.gradcam_layer)
    heatmap, pred_cls, conf = cam.generate_heatmap(x)
    cam.remove_hooks()
    assert heatmap.shape == (128, 128)
    assert pred_cls in (0, 1)
    assert 0.0 <= float(conf) <= 1.0

    pil = Image.fromarray((np.random.rand(128, 128) * 255).astype(np.uint8), mode="L")
    artifacts = encode_gradcam_artifacts(pil, heatmap, colormap_name="turbo")
    assert artifacts["heatmap_base64"].startswith("data:image/png;base64,")
    assert artifacts["overlay_base64"].startswith("data:image/png;base64,")
    assert "x_norm" in artifacts["hotspot"] and "y_norm" in artifacts["hotspot"]


def test_ecg_18_telemetry_and_resnet1d():
    sigs = np.random.randn(3, 256).astype(np.float32)
    tele = analyze_ecg_telemetry(sigs[0], sampling_rate=250)
    for key in ECG_FEATURE_NAMES:
        assert key in tele
        assert math.isfinite(float(tele[key]))

    mat = extract_ecg_feature_matrix(sigs, sampling_rate=250)
    assert mat.shape == (3, 18)

    net = ECGResNet1D(in_channels=1, num_classes=5)
    out = net(torch.from_numpy(sigs).unsqueeze(1))
    assert out.shape == (3, 5)


def test_clinical_feature_engineering_and_multimodal_fusion():
    sample = {
        "age": 62,
        "sex": 1,
        "chest_pain_type": 4,
        "resting_bp": 152,
        "cholesterol": 265,
        "fasting_bs": 1,
        "resting_ecg": 1,
        "max_hr": 108,
        "exercise_angina": 1,
        "oldpeak": 2.4,
        "st_slope": 3,
    }
    df = pd.DataFrame([sample, {**sample, "resting_bp": np.nan, "cholesterol": np.nan}])
    feat_df = engineer_clinical_features(df)
    assert list(feat_df.columns) == CLINICAL_FEATURE_NAMES
    assert len(CLINICAL_FEATURE_NAMES) == 20
    # Ensure row 2 preserves NaN for resting_bp/cholesterol rather than leaking dataset stats
    assert pd.isna(feat_df.loc[1, "rate_pressure_product"])
    assert pd.isna(feat_df.loc[1, "cholesterol_to_age"])

    fusion_low = compute_multimodal_triage_score(0.20, {**sample, "oldpeak": 0.2, "exercise_angina": 0, "chest_pain_type": 2, "max_hr": 165, "resting_bp": 115, "cholesterol": 180})
    fusion_high = compute_multimodal_triage_score(
        0.72, sample, cxr_pneumonia_probability=0.91, ecg_is_anomaly=True, ecg_reconstruction_mse=0.055
    )
    assert fusion_high["fused_risk_score"] > fusion_low["fused_risk_score"]
    assert "CRITICAL" in fusion_high["triage_tier"]
    assert len(fusion_high["multimodal_modifiers"]) == 2


def test_pretrained_catalog_structure():
    catalog = get_curated_catalog()
    assert len(catalog) >= 5
    domains = {m["domain"] for m in catalog}
    assert {"vision", "ecg", "clinical"}.issubset(domains)


def test_v2_api_endpoints_end_to_end():
    r_bench = client.get("/v1/benchmarks")
    assert r_bench.status_code == 200
    assert "summary" in r_bench.json()

    r_samples = client.get("/v1/dataset/samples")
    assert r_samples.status_code == 200
    sdata = r_samples.json()
    assert "vision_samples" in sdata
    assert "ecg_samples" in sdata
    assert "clinical_presets" in sdata

    r_cat = client.get("/v1/pretrained/catalog")
    assert r_cat.status_code == 200
    assert len(r_cat.json()["catalog"]) >= 5

    # Vision v2 payload includes Grad-CAM RGBA PNG + 14 radiomics
    img = Image.fromarray((np.random.rand(128, 128) * 255).astype(np.uint8))
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    r_vis = client.post(
        "/v1/vision/diagnose?mc_samples=6&model_id=residual_cnn&colormap=turbo",
        files={"file": ("cxr.png", buf, "image/png")},
    )
    assert r_vis.status_code == 200
    vjson = r_vis.json()
    assert vjson["gradcam_heatmap_base64"].startswith("data:image/png;base64,")
    assert len(vjson["radiomics"]) == 14

    # ECG v2 payload includes reconstructed_signal, residual_profile, subtype_probabilities
    sig = (np.sin(np.linspace(0, 6 * np.pi, 256))).tolist()
    r_ecg = client.post("/v1/ecg/screen", json={"signal": sig, "sampling_rate_hz": 250})
    assert r_ecg.status_code == 200
    ejson = r_ecg.json()
    assert len(ejson["reconstructed_signal"]) == 256
    assert len(ejson["residual_profile"]) == 256
    assert "dominant_freq_hz" in ejson["telemetry"]

    # Clinical endpoint if trained
    if os.path.exists("saved_models/best_clinical_triage.joblib"):
        r_clin = client.post(
            "/v1/clinical/triage",
            json={
                "age": 60,
                "sex": 1,
                "chest_pain_type": 4,
                "resting_bp": 148,
                "cholesterol": 255,
                "fasting_bs": 1,
                "resting_ecg": 1,
                "max_hr": 118,
                "exercise_angina": 1,
                "oldpeak": 2.1,
                "st_slope": 3,
                "cxr_pneumonia_probability": 0.85,
                "ecg_is_anomaly": True,
                "ecg_reconstruction_mse": 0.042,
            },
        )
        assert r_clin.status_code == 200
        cjson = r_clin.json()
        assert 0.0 <= cjson["cad_risk_probability"] <= 1.0
        assert len(cjson["engineered_biomarkers"]) == 9
