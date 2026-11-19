# 📋 Clinical Model Card: Deep Health AI Suite v2.0

Following the standard proposed by Mitchell et al. (2019) and FDA Guidance for AI/ML-Based Software as a Medical Device (SaMD).

---

## 1. Model Details

- **Developer:** Saivi-9 (Open Source Clinical AI Research)
- **Model Version:** 2.0.0
- **Modalities & Architectures:**
  1. **2D Thoracic Radiology (`ResidualChestCNN`, `TransferResNet18`, `TransferMobileNetV2`, `RadiomicLightGBM`):** Deep residual CNN with CBAM Spatial Attention & Grad-CAM, transfer-learned TorchVision backbones (stem-frozen ImageNet weights), and a 14-biomarker radiomic GBDT baseline trained on **3,148** chest radiographs (**2,000** high-variance scans + **1,148** real hospital PneumoniaMNIST scans from `danjacobellis/pneumoniamnist_224`).
  2. **1D Cardiac Electrophysiology (`Conv1dAutoencoder`, `ECGResNet1D`, `ECGMorphologyLightGBM`, `ECGMorphologyHistGB`, `ECGResNet1D_MITBIH`):** Compressed 12-D bottleneck unsupervised NSR reconstruction autoencoder (`21:1` ratio), 5-class supervised 1D-ResNet, 18-feature morphological/FFT GBDT models, and a supervised 5-class MIT-BIH classifier trained on **17,500** total ECG beats (**7,500** Lead-II 5-class + **10,000** real PhysioNet MIT-BIH beats from `dheerajthuvara/ecg-arrhythmia-data`).
  3. **Cardiology EHR Triage (`HistGradientBoosting`, `LightGBM`, `RandomForest`, `ExtraTrees`, `StackingEnsemble`):** 20-biomarker leak-free ensemble (`11` raw clinical attributes + `9` deterministic physiological/ischemic interactions + multimodal CXR/ECG acuity fusion) trained on **1,305** deduplicated multi-center Cardiology EHR patients (`BenchmarkDatasets/Heart-Disease-Dataset-_Comprehensive` + `TabBioMed/HeartDisease_UCI`).
  4. **Hugging Face Pre-Trained Medical Hub:** Verified public benchmarks (`Purino/mobilenetv2-chest-xray-pneumonia-256-fullframe-gray`, `Purino/mobilenetv2-chest-xray-pneumonia`, `sdoerrich97/colorist_densenet121_pneumoniamnist_s71397589`, `dheerajthuvara/ecg-arrhythmia-detection`, `nexusbert/heart-disease-random-forest`).
- **License:** MIT License

---

## 2. Intended Use

- **Primary Intended Use:** Clinical decision support (CDS) for emergency triage, assisting licensed radiologists and cardiologists in prioritizing chest radiographs exhibiting acute consolidation/pneumonia, differentiating cardiac arrhythmias on 1D ECG streams, and stratifying coronary artery disease risk with multi-modal acuity fusion.
- **Primary Target Users:** Hospital clinicians, triage nurses, diagnostic radiologists, and telemetry technicians.
- **Out-of-Scope Use Cases:**
  - Standalone autonomous diagnosis without human-in-the-loop oversight.
  - Non-thoracic radiographic projections (e.g., abdominal or extremity X-rays).

---

## 3. Quantitative Performance & Benchmarks

All metrics measured empirically on isolated, leak-free holdout splits:

### A. 2D Thoracic Radiograph Interpretation (`n=473` Unified Holdout / `n=624` Real PneumoniaMNIST Holdout)

| Architecture | Unified Acc | Unified ROC-AUC | Real PneumoniaMNIST Acc | Real PneumoniaMNIST ROC-AUC | Real Sensitivity | Real Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ResidualChestCNN` | 87.32% | 0.9231 | 86.86% | 0.9367 | 95.38% | 72.65% |
| `TransferResNet18` | **91.33%** | **0.9826** | 95.67% | **0.9926** | 95.13% | **96.58%** |
| `TransferMobileNetV2` | 90.70% | 0.9611 | **96.31%** | 0.9911 | 97.44% | 94.44% |
| `RadiomicLightGBM` (14 features) | 83.51% | 0.9017 | 95.99% | 0.9860 | **97.69%** | 93.16% |
| HF: `Purino/mobilenetv2-...-256-gray` | — | — | **96.15%** | **0.9975** | 94.36% | **99.15%** |
| HF: `Purino/mobilenetv2-chest-xray-pneumonia` | — | — | 95.83% | 0.9950 | 95.13% | 97.01% |
| HF: `sdoerrich97/...densenet121_pneumoniamnist` | — | — | 88.78% | 0.9783 | **99.74%** | 70.51% |

### B. 1D Cardiac Electrophysiology (`n=2,400` Lead-II 5-Class Test / `n=1,500` Real MIT-BIH Test)

| Architecture | Binary Anomaly Acc | Binary ROC-AUC | 5-Class Acc | Macro-F1 | Multi-Class OVR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Conv1dAutoencoder` (12-D Bottleneck, $\theta=0.045161$) | 99.38% | **1.0000** | — | — | — |
| `ECGResNet1D` (Lead-II 5-Class) | **100.00%** | **1.0000** | **100.00%** | **1.0000** | **1.0000** |
| `ECGMorphologyHistGB` (18 biomarkers) | 99.25% | 0.9995 | 99.21% | 0.9921 | 0.9998 |
| `ECGMorphologyLightGBM` (18 biomarkers) | 99.13% | 0.9995 | 99.04% | 0.9904 | 0.9998 |
| `ECGResNet1D_MITBIH` (10k Real Beats) | — | — | 95.40% | 0.9541 | 0.9965 |
| HF: `dheerajthuvara/ecg-arrhythmia-detection` | — | — | **97.60%** | **0.9762** | **0.9994** |

### C. Cardiology EHR Risk Stratification (`n=1,305` Unique Patients; `n=261` Holdout Test)

| Model / Strategy | 5-Fold CV ROC-AUC | Test Accuracy | Test Sensitivity | Test Specificity | Test ROC-AUC | Test PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`HistGradientBoosting`** *(Champion)* | 0.9240 | 88.89% | 86.23% | 91.87% | **0.9566** | **0.9630** |
| `StackingEnsemble` | **0.9269** | 88.89% | 85.51% | **92.68%** | 0.9553 | 0.9575 |
| `RandomForest` | 0.9152 | **90.42%** | **88.41%** | **92.68%** | 0.9513 | 0.9576 |
| `LightGBM` | 0.9208 | 88.51% | 85.51% | 91.87% | 0.9482 | 0.9551 |
| `ExtraTrees` | 0.9114 | 87.36% | 85.51% | 89.43% | 0.9473 | 0.9482 |
| HF: `nexusbert/heart-disease-random-forest` (7 raw Z) | — | 72.80% | 75.36% | 69.92% | 0.8371 | 0.8696 |

---

## 4. Explainability & Uncertainty Quantification

1. **Grad-CAM RGBA Attention Mapping:**
   - Computes target-class gradients wrt the final spatial conv layer (`.gradcam_layer`, supporting both `ResidualChestCNN` and TorchVision `ResNet18` / `MobileNetV2` / `DenseNet121`), returning transparent RGBA overlays and localized peak bounding coordinates.
2. **Monte Carlo Dropout Epistemic Uncertainty:**
   - Executes $N$ stochastic passes (configurable 5–50) with active dropout to compute predictive variance $\sigma^2$ and normalized Shannon entropy.
   - **Triage Escalation Thresholds:** $\sigma^2 > 0.035$ or Entropy $> 0.60$ mandates mandatory secondary specialist review.

---

## 5. Ethical, Data Leakage & Governance Safeguards

- **Strict Anti-Leakage Engineering:** Multi-source Cardiology EHR cohorts are URL/feature-deduplicated prior to `train_test_split`; missing clinical measurements (`resting_bp`, `cholesterol`, `fasting_bs`) are preserved as `NaN` through deterministic row-wise transforms so fold `SimpleImputer` / `StandardScaler` fit strictly inside each training fold.
- **Zero-Mock Policy:** Every Grad-CAM PNG, ECG reconstruction waveform, residual array, batch cohort evaluation, and benchmark leaderboard in the FastAPI backend and React PACS UI is computed server-side from saved model weights and real cached cohorts.
