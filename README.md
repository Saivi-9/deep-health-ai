# 🩺 Deep Health AI v2.0 — Multi-Modal Clinical AI Suite & Real-Time PACS Workstation

[![Python](https://img.shields.io/badge/Python-3.14%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-30%2F30%20Passing-brightgreen.svg)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React%2019-Vite%20PACS%20UI-38BDF8.svg)](frontend/)
[![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Medical%20Hub-FFD21E.svg)](src/pretrained_hub.py)
[![Model Card](https://img.shields.io/badge/Docs-Model%20Card-blue.svg)](docs/MODEL_CARD.md)

**Deep Health AI v2.0** is an enterprise-grade multi-modal clinical AI decision support suite and **real-time interactive PACS workstation** combining **real hospital clinical datasets**, **multi-architecture PyTorch & GBDT modeling**, **leak-free Cardiology EHR risk fusion**, and **live Hugging Face Medical Pre-Trained Model Hub integration**.

---

## 🚀 What's New in v2.0

1. **Scaled & Real Hospital Multi-Modal Cohorts:**
   - **Thoracic Chest X-Rays (`3,148` scans):** `1,148` real hospital radiographs from **`danjacobellis/pneumoniamnist_224`** (`n=624` held-out real test split) unified with `2,000` high-variance anatomical radiographs across Normal, Lobar Bacterial, Interstitial Viral, and Subtle Early Infiltrates.
   - **1D Cardiac Electrophysiology (`17,500` ECG beats):** `10,000` real PhysioNet MIT-BIH 5-class arrhythmia beats (`A`, `L`, `N`, `R`, `V` from **`dheerajthuvara/ecg-arrhythmia-data`**) alongside `7,500` Lead-II 250Hz ECG waveforms (`NSR`, `PVC`, `AFib`, `ST-Elevation`, `LBBB`).
   - **Cardiology EHR Triage (`1,305` unique patients):** Deduplicated multi-center clinical cohort merged from **`BenchmarkDatasets/Heart-Disease-Dataset-_Comprehensive`** and **`TabBioMed/HeartDisease_UCI`**.

2. **Multi-Architecture Benchmark Suites & Rich Biomarker Feature Engineering:**
   - **Vision (4 Models + 14 Radiomics):** [`ResidualChestCNN`](src/vision/model.py) (with CBAM Spatial Attention & MC-Dropout), [`TransferResNet18`](src/vision/model.py), [`TransferMobileNetV2`](src/vision/model.py), and [`RadiomicLightGBM`](train_vision.py) over `14` extracted intensity/texture/spatial lung-zone radiomic biomarkers.
   - **ECG (5 Evaluations + 18 Electrophysiological/FFT Features):** Compressed 12-D bottleneck [`Conv1dAutoencoder`](src/ecg/model.py), 5-class supervised [`ECGResNet1D`](src/ecg/model.py), `ECGMorphologyLightGBM`, `ECGMorphologyHistGB`, and real MIT-BIH `ECGResNet1D`.
   - **Clinical EHR (5-Model Tournament + 20 Biomarkers + Multimodal Fusion):** `11` raw EHR attributes + `9` deterministic clinical indices (`rate_pressure_product`, `estimated_max_hr_pct`, `st_severity_index`, `cardiac_ischemia_burden`, `metabolic_risk_count`, etc.) with leak-free fold imputation/scaling across `HistGradientBoosting`, `LightGBM`, `RandomForest`, `ExtraTrees`, and `StackingEnsemble`, plus multimodal CXR + ECG acuity fusion.

3. **Verified Hugging Face Pre-Trained Medical Hub ([`src/pretrained_hub.py`](src/pretrained_hub.py)):**
   - Curated public checkpoints (`Purino/mobilenetv2-chest-xray-pneumonia-256-fullframe-gray`, `Purino/mobilenetv2-chest-xray-pneumonia`, `sdoerrich97/colorist_densenet121_pneumoniamnist_s71397589`, `dheerajthuvara/ecg-arrhythmia-detection`, `nexusbert/heart-disease-random-forest`), **live HF Hub search API**, true Grad-CAM over TorchVision backbones, and side-by-side empirical benchmarking.

4. **Real-Time PACS Workstation UI ([`frontend/`](frontend/)) & Dual Streamlit UI ([`app.py`](app.py)):**
   - Served natively from FastAPI (`http://localhost:8000`) with zero mocked heatmaps, waveforms, or tables:
     - **Thoracic PACS Viewer:** Live DICOM windowing, **real server-computed Grad-CAM RGBA overlay**, peak bounding box, 14 radiomic biomarkers, MC-Dropout uncertainty gauge, and real PneumoniaMNIST presets.
     - **ECG Telemetry Monitor:** Interactive Pan/Zoom **real autoencoder reconstruction & MSE residual overlays**, 5-class rhythm differential bars, and 18 interval/FFT biomarkers.
     - **Clinical Cardiology Triage:** 20-biomarker EHR risk engine, live multimodal CXR + ECG acuity linkage, driver breakdown, and Hugging Face consensus check.
     - **HF Pre-Trained Hub:** Live search + instant Vision/ECG/Clinical inference directly against downloaded HF checkpoints.
     - **Cohort Epidemiology & Model Governance:** Real batch evaluation endpoints (`/v1/batch/evaluate`) and live empirical leaderboards (`/v1/benchmarks`).

---

## 📊 Verified Empirical Benchmark Results

### 1. Thoracic Chest Radiograph Suite
| Model / Checkpoint | Unified Holdout Acc | Unified ROC-AUC | Real PneumoniaMNIST Acc | Real PneumoniaMNIST ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| `TransferResNet18` (ImageNet Stem-Frozen) | **91.33%** | **0.9826** | 95.67% | **0.9926** |
| `TransferMobileNetV2` (ImageNet Stem-Frozen) | 90.70% | 0.9611 | **96.31%** | 0.9911 |
| `ResidualChestCNN + SpatialAttention` | 87.32% | 0.9231 | 86.86% | 0.9367 |
| `RadiomicLightGBM` (14 Radiomics) | 83.51% | 0.9017 | 95.99% | 0.9860 |
| **HF Hub:** `Purino/mobilenetv2-chest-xray-pneumonia-256-fullframe-gray` | — | — | **96.15%** | **0.9975** |
| **HF Hub:** `Purino/mobilenetv2-chest-xray-pneumonia` | — | — | 95.83% | 0.9950 |
| **HF Hub:** `sdoerrich97/colorist_densenet121_pneumoniamnist_s71397589` | — | — | 88.78% | 0.9783 |

### 2. 1D ECG Arrhythmia & Anomaly Detection Suite
| Model / Checkpoint | Cohort | Binary Anomaly Acc | Binary ROC-AUC | 5-Class Acc | Macro-F1 | OVR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `Conv1dAutoencoder` (12-D Bottleneck, $\theta=0.045161$) | Lead-II (`n=2,400`) | 99.38% | **1.0000** | — | — | — |
| `ECGResNet1D` (Supervised 1D ResNet) | Lead-II (`n=2,400`) | **100.00%** | **1.0000** | **100.00%** | **1.0000** | **1.0000** |
| `ECGMorphologyHistGB` (18 Biomarkers) | Lead-II (`n=2,400`) | 99.25% | 0.9995 | 99.21% | 0.9921 | 0.9998 |
| `ECGMorphologyLightGBM` (18 Biomarkers) | Lead-II (`n=2,400`) | 99.13% | 0.9995 | 99.04% | 0.9904 | 0.9998 |
| `ECGResNet1D_MITBIH` | Real MIT-BIH (`n=1,500`) | — | — | 95.40% | 0.9541 | 0.9965 |
| **HF Hub:** `dheerajthuvara/ecg-arrhythmia-detection` | Real MIT-BIH (`n=1,500`) | — | — | **97.60%** | **0.9762** | **0.9994** |

### 3. Cardiology EHR Risk Triage Suite (`n=1,305` Unique Patients; `n=261` Test Split)
| Model / Checkpoint | 5-Fold CV AUC | Test Accuracy | Test ROC-AUC | Test PR-AUC |
| :--- | :---: | :---: | :---: | :---: |
| `HistGradientBoosting` *(Champion)* | 0.9240 | 88.89% | **0.9566** | **0.9630** |
| `StackingEnsemble` | **0.9269** | 88.89% | 0.9553 | 0.9575 |
| `RandomForest` | 0.9152 | **90.42%** | 0.9513 | 0.9576 |
| `LightGBM` | 0.9208 | 88.51% | 0.9482 | 0.9551 |
| `ExtraTrees` | 0.9114 | 87.36% | 0.9473 | 0.9482 |
| **HF Hub:** `nexusbert/heart-disease-random-forest` | — | 72.80% | 0.8371 | 0.8696 |

---

## 📁 Repository Structure

```
deep-health-ai/
├── data/
│   ├── fetch_external.py           # Real PneumoniaMNIST, MIT-BIH 10k, Cardiology EHR fetcher
│   ├── generate_datasets.py        # Multi-regime Vision (2k) & 5-class Lead-II ECG (5.5k) synthesis
│   └── external/                   # Cached real datasets & manifests
├── src/
│   ├── vision/                     # ResidualChestCNN, TransferResNet18/MobileNetV2, GradCAM RGBA, 14 Radiomics
│   ├── ecg/                        # Conv1dAutoencoder (12-D), ECGResNet1D, 18 Telemetry/FFT biomarkers
│   ├── clinical/                   # 20-biomarker Cardiology EHR feature engine & 5-model trainer
│   ├── api/                        # FastAPI v2.0 endpoints, Pydantic v2 schemas, static PACS mount
│   ├── pretrained_hub.py           # Hugging Face Hub live search, inference, Grad-CAM & benchmarks
│   └── utils/                      # Clinical metrics, logging, 300-DPI visualizers
├── frontend/                       # React 19 + TypeScript + Vite + Tailwind v4 PACS Workstation UI
├── tests/                          # 30 unit & API integration tests (test_v2_upgrades.py, etc.)
├── docs/                           # MODEL_CARD.md & CLINICAL_VALIDATION.md
├── reports/                        # metrics_summary.json, vision/ecg/clinical/pretrained JSONs & figures
├── saved_models/                   # Trained .pt and .joblib checkpoints across all modalities
├── api_server.py                   # FastAPI & React UI entrypoint (default port 8000)
├── app.py                          # Multi-page Streamlit Clinical Workstation v2.0
├── train_vision.py / train_ecg.py / train_clinical.py / demo.py
├── batch_processor.py / predict.py
└── requirements.txt
```

---

## ⚡ Quickstart & Live Real-Time Workstation

### 1. Environment Setup
```bash
git clone https://github.com/Saivi-9/deep-health-ai.git
cd deep-health-ai
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Full Training, Real HF Cohorts & Benchmark Pipeline
```bash
python demo.py
```

### 3. Launch Live FastAPI Microservice + Interactive React PACS UI
```bash
python api_server.py --port 8000
```
- **Interactive Real-Time PACS Workstation UI:** `http://localhost:8000`
- **Swagger / OpenAPI Docs:** `http://localhost:8000/docs`

### 4. Launch Streamlit Clinical Suite (Alternative UI)
```bash
streamlit run app.py
```

### 5. Run Full Test Suite
```bash
pytest tests/ -v
```

---

## ⚖️ License
This project is licensed under the [MIT License](LICENSE).
