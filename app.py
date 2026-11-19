"""
Deep Health AI v2.0 - Clinical Decision Support & PACS Workstation (Streamlit)
------------------------------------------------------------------------------
Multi-modal clinical diagnostics platform supporting:
1. Thoracic Radiology (ResidualChestCNN, ResNet18, MobileNetV2, 14 Radiomics + Real PneumoniaMNIST)
2. 1D ECG Electrophysiology (Conv1dAutoencoder + 5-Class ECGResNet1D + MIT-BIH 10k + 18 Biomarkers)
3. Real Cardiology EHR Risk & Multimodal Fusion (Stacking Ensemble on 1,305 Real Patients)
4. Hugging Face Pre-Trained Medical Hub (Empirical Benchmarks + Live Model Search)
5. Quantitative Governance & Printable Consultation Report Export

Launch:
    streamlit run app.py
"""

import os
import glob
import json
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.vision.model import ResidualChestCNN, TransferChestClassifier
from src.vision.gradcam import GradCAM
from src.vision.uncertainty import MonteCarloDropoutEstimator
from src.vision.radiology_tools import apply_radiology_window, compute_consolidation_severity, extract_radiomic_features
from src.ecg.model import Conv1dAutoencoder, ECGResNet1D
from src.ecg.anomaly_detector import ECGAnomalyDetector
from src.ecg.telemetry import analyze_ecg_telemetry
from src.clinical.features import engineer_clinical_features, compute_multimodal_triage_score
from src.pretrained_hub import get_curated_catalog, search_hf_medical_models, infer_pretrained_clinical

st.set_page_config(
    page_title="Deep Health AI v2.0 - Clinical PACS Workstation",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0B1120; color: #E2E8F0; }
    .pacs-header {
        background: linear-gradient(90deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid #334155; border-radius: 10px;
        padding: 1.1rem 1.5rem; margin-bottom: 1rem;
    }
    .pacs-title { font-size: 1.75rem; font-weight: 800; color: #38BDF8; }
    .pacs-sub { font-size: 0.9rem; color: #94A3B8; margin-top: 0.2rem; }
    .badge-normal {
        background-color: rgba(34, 197, 94, 0.15); color: #4ADE80;
        border: 1px solid rgba(34, 197, 94, 0.35); padding: 0.35rem 0.85rem;
        border-radius: 6px; font-weight: 700; font-size: 1.05rem;
    }
    .badge-pathology {
        background-color: rgba(239, 68, 68, 0.15); color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.35); padding: 0.35rem 0.85rem;
        border-radius: 6px; font-weight: 700; font-size: 1.05rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_vision_model(model_key: str = "residual_cnn", device: str = "cpu"):
    if model_key == "resnet18" and os.path.exists("saved_models/vision_resnet18.pt"):
        m = TransferChestClassifier(arch="resnet18", num_classes=2, pretrained=False)
        m.load_state_dict(torch.load("saved_models/vision_resnet18.pt", map_location=device, weights_only=True))
    elif model_key == "mobilenet_v2" and os.path.exists("saved_models/vision_mobilenetv2.pt"):
        m = TransferChestClassifier(arch="mobilenet_v2", num_classes=2, pretrained=False)
        m.load_state_dict(torch.load("saved_models/vision_mobilenetv2.pt", map_location=device, weights_only=True))
    else:
        m = ResidualChestCNN(in_channels=1, num_classes=2)
        if os.path.exists("saved_models/best_chest_cnn.pt"):
            m.load_state_dict(torch.load("saved_models/best_chest_cnn.pt", map_location=device, weights_only=True))
    m.to(device).eval()
    return m


@st.cache_resource
def load_ecg_suite(device: str = "cpu"):
    ae = Conv1dAutoencoder(in_channels=1)
    if os.path.exists("saved_models/best_ecg_autoencoder.pt"):
        ae.load_state_dict(torch.load("saved_models/best_ecg_autoencoder.pt", map_location=device, weights_only=True))
    ae.to(device).eval()

    threshold = 0.01037
    if os.path.exists("reports/ecg_metrics.json"):
        with open("reports/ecg_metrics.json", "r") as f:
            threshold = float(json.load(f).get("threshold", 0.01037))
    detector = ECGAnomalyDetector(model=ae, threshold=threshold, device=device)

    rn = None
    if os.path.exists("saved_models/ecg_resnet1d.pt"):
        rn = ECGResNet1D(in_channels=1, num_classes=5)
        rn.load_state_dict(torch.load("saved_models/ecg_resnet1d.pt", map_location=device, weights_only=True))
        rn.to(device).eval()
    return detector, threshold, rn


@st.cache_resource
def load_clinical_bundle():
    p = "saved_models/best_clinical_triage.joblib"
    return joblib.load(p) if os.path.exists(p) else None


device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.markdown("### Deep Health AI v2.0 Controls")
st.sidebar.markdown(f"**Compute Node:** `{device.upper()}`")
patient_mrn = st.sidebar.text_input("Patient MRN", value="MRN-849201")
patient_name = st.sidebar.text_input("Patient Full Name", value="Eleanor Vance")
mc_samples = st.sidebar.slider("MC Dropout Passes", 10, 40, 20, 5)
gradcam_colormap = st.sidebar.selectbox("Grad-CAM Colormap", ["turbo", "jet", "plasma", "inferno"])

st.markdown("""
<div class="pacs-header">
    <div class="pacs-title">🩺 Deep Health AI v2.0: Multi-Modal Clinical PACS Workstation</div>
    <div class="pacs-sub">
        Real PneumoniaMNIST &amp; MIT-BIH Cohorts • 4 Vision Architectures • 5-Class ECG Differential • Real Cardiology EHR Fusion • Hugging Face Medical Hub
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🫁 Thoracic Radiology & Radiomics",
    "💓 1D-ECG 5-Class Telemetry",
    "🫀 Cardiology EHR & Multimodal Triage",
    "🤗 Hugging Face Medical Hub",
    "📊 Benchmarks & Consultation Export",
])

with tab1:
    c_in, c_out = st.columns([1, 1.3])
    with c_in:
        vision_arch = st.selectbox("Vision Model Architecture", ["residual_cnn", "resnet18", "mobilenet_v2"])
        study_opts = {}
        if os.path.exists("data/external/pneumoniamnist_manifest.csv"):
            df_m = pd.read_csv("data/external/pneumoniamnist_manifest.csv")
            df_te = df_m[df_m["source_split"] == "test_real"]
            for lbl_val, tag in [(1, "Real PneumoniaMNIST Consolidation"), (0, "Real PneumoniaMNIST Normal")]:
                sub = df_te[df_te["label"] == lbl_val].head(2)
                for _, r in sub.iterrows():
                    study_opts[f"{tag} ({r['sample_id']})"] = r["relative_path"]
        p_files = sorted(glob.glob("data/vision/pneumonia/*.png"))
        n_files = sorted(glob.glob("data/vision/normal/*.png"))
        if p_files:
            study_opts["Generated: Alveolar Consolidation"] = p_files[0]
        if n_files:
            study_opts["Generated: Clear Normal Parenchyma"] = n_files[0]

        up_file = st.file_uploader("Or Upload Chest Radiograph (PNG/JPG)", type=["png", "jpg", "jpeg"])
        selected_title = st.selectbox("Clinical Study Preset", list(study_opts.keys())) if study_opts else ""
        win_preset = st.selectbox("Window Preset", ["Standard Grayscale", "Lung Window", "Bone Window", "Inverted Radiograph (Fluoroscopy)"])
        overlay_alpha = st.slider("Grad-CAM Alpha", 0.0, 1.0, 0.55, 0.05)

        raw_img = Image.open(up_file).convert("L") if up_file else (Image.open(study_opts[selected_title]).convert("L") if selected_title else None)

    with c_out:
        if raw_img is not None:
            v_model = load_vision_model(vision_arch, device=device)
            np_img = np.array(raw_img.resize((128, 128)), dtype=np.float32)
            processed_win = apply_radiology_window(np_img, win_preset)

            tf = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize([0.485], [0.229])])
            tensor_in = tf(raw_img).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = F.softmax(v_model(tensor_in), dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))

            class_labels = ["Normal Healthy Parenchyma", "Alveolar Consolidation / Pneumonia"]
            uncert = MonteCarloDropoutEstimator(v_model, n_samples=mc_samples, device=device).estimate_uncertainty(tensor_in)
            target_layer = getattr(v_model, "gradcam_layer", v_model.layer4)
            gc = GradCAM(v_model, target_layer)
            hm, _, _ = gc.generate_heatmap(tensor_in, target_class=pred_idx)
            gc.remove_hooks()
            sev = compute_consolidation_severity(hm, threshold=0.40)
            rad_feats = extract_radiomic_features(np_img)

            badge_cls = "badge-pathology" if pred_idx == 1 else "badge-normal"
            st.markdown(f'<div class="{badge_cls}">FINDING ({vision_arch}): {class_labels[pred_idx].upper()}</div>', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Pneumonia Prob", f"{probs[1]*100:.1f}%")
            m2.metric("Infiltrate Burden", f"{sev['infiltrate_burden_percentage']}%")
            m3.metric("Severity Grade", sev["radiological_grade"])
            m4.metric("MC Variance", f"{uncert['predictive_variance']:.5f}")

            fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), facecolor="#0B1120")
            for ax in axes:
                ax.axis("off")
            axes[0].imshow(np_img / 255.0, cmap="gray")
            axes[0].set_title("Raw CXR", color="#94A3B8", fontsize=9)
            axes[1].imshow(processed_win, cmap="gray")
            axes[1].set_title(win_preset, color="#94A3B8", fontsize=9)
            axes[2].imshow(processed_win, cmap="gray")
            axes[2].imshow(hm, cmap=gradcam_colormap, alpha=overlay_alpha, vmin=0, vmax=1)
            axes[2].set_title(f"Grad-CAM ({gradcam_colormap})", color="#38BDF8", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption("14 Quantitative Radiomic Biomarkers:")
            st.dataframe(pd.DataFrame([rad_feats]).T.rename(columns={0: "Value"}), use_container_width=True)
            st.session_state["latest_radiology"] = {
                "study_title": up_file.name if up_file else selected_title,
                "diagnosis": class_labels[pred_idx],
                "pneumonia_prob": float(probs[1]),
                "confidence": float(probs[pred_idx]),
                "severity": sev,
                "uncertainty": uncert,
            }

with tab2:
    detector, default_th, ecg_rn = load_ecg_suite(device=device)
    e_in, e_plot = st.columns([1, 1.8])
    with e_in:
        rhythm_map = {"Normal Sinus (NSR)": 0, "PVC Ectopy": 1, "Atrial Fibrillation": 2, "ST-Elevation Ischemia": 3, "LBBB": 4}
        choice = st.selectbox("Lead-II Rhythm Class", list(rhythm_map.keys()))
        custom_th = st.slider("Anomaly Threshold", 0.004, 0.050, float(default_th), 0.001, format="%.5f")
        current_sig = None
        if os.path.exists("data/ecg/ecg_test.npy") and os.path.exists("data/ecg/ecg_test_multiclass.npy"):
            Xe = np.load("data/ecg/ecg_test.npy")
            ym = np.load("data/ecg/ecg_test_multiclass.npy")
            hits = np.where(ym == rhythm_map[choice])[0]
            if len(hits) > 0:
                current_sig = Xe[hits[0]]

    with e_plot:
        if current_sig is not None:
            detector.threshold = custom_th
            res = detector.predict(current_sig)
            tele = analyze_ecg_telemetry(current_sig, sampling_rate=250)
            recon = res["reconstructed_signal"]
            residuals = (current_sig - recon) ** 2

            subtype_str = "Autoencoder Unsupervised"
            if ecg_rn is not None:
                with torch.no_grad():
                    ps = F.softmax(ecg_rn(torch.from_numpy(current_sig).view(1, 1, -1).to(device)), dim=1).cpu().numpy()[0]
                names = ["NSR", "PVC", "AFib", "ST-Elevation", "LBBB"]
                subtype_str = f"{names[int(np.argmax(ps))]} ({np.max(ps)*100:.1f}%)"

            st.markdown(f"**Diagnosis:** `{res['diagnosis']}` • **ResNet1D Differential:** `{subtype_str}`")
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("QRS Duration", f"{tele['qrs_duration_ms']} ms")
            t2.metric("QTc (Bazett)", f"{tele['qtc_interval_ms']} ms")
            t3.metric("Dominant Freq", f"{tele['dominant_freq_hz']} Hz")
            t4.metric("Recon MSE", f"{res['reconstruction_error']:.5f}")

            t_ms = np.arange(len(current_sig)) * 4.0
            fig_e = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12)
            fig_e.add_trace(go.Scatter(x=t_ms, y=current_sig, name="Patient Lead-II", line=dict(color="#38BDF8", width=2)), row=1, col=1)
            fig_e.add_trace(go.Scatter(x=t_ms, y=recon, name="Conv1D-AE x̂", line=dict(color="#4ADE80", dash="dash")), row=1, col=1)
            fig_e.add_trace(go.Scatter(x=t_ms, y=residuals, name="Squared Residual", fill="tozeroy", line=dict(color="#F87171")), row=2, col=1)
            fig_e.add_hline(y=custom_th, line_dash="dot", line_color="#EF4444", row=2, col=1)
            fig_e.update_layout(height=390, paper_bgcolor="#0F172A", plot_bgcolor="#0B1120", font=dict(color="#94A3B8"))
            st.plotly_chart(fig_e, use_container_width=True)
            st.session_state["latest_ecg_telemetry"] = {"is_anomaly": res["is_anomaly"], "mse": res["reconstruction_error"], "diagnosis": res["diagnosis"], "telemetry": tele}

with tab3:
    st.subheader("Real Cardiology EHR Risk Stratification & Multimodal Triage")
    clin_bundle = load_clinical_bundle()
    if clin_bundle is None:
        st.warning("Clinical triage model not yet trained. Run `python train_clinical.py`.")
    else:
        c1, c2, c3 = st.columns(3)
        age = c1.slider("Age", 25, 85, 62)
        sex = c2.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = c3.selectbox("Chest Pain Type", [1, 2, 3, 4], index=3)
        bp = c1.slider("Resting BP (mmHg)", 85, 210, 145)
        chol = c2.slider("Cholesterol (mg/dL)", 120, 450, 265)
        max_hr = c3.slider("Max Heart Rate", 70, 205, 122)
        oldpeak = c1.slider("ST Oldpeak (mm)", 0.0, 5.5, 2.1, 0.1)
        exang = c2.selectbox("Exercise Angina", [0, 1], index=1, format_func=lambda x: "Yes" if x == 1 else "No")
        slope = c3.selectbox("ST Slope", [1, 2, 3], index=1)

        pt_dict = {"age": age, "sex": sex, "chest_pain_type": cp, "resting_bp": bp, "cholesterol": chol, "fasting_bs": 0, "resting_ecg": 1, "max_hr": max_hr, "exercise_angina": exang, "oldpeak": oldpeak, "st_slope": slope}
        X_feat = engineer_clinical_features(pd.DataFrame([pt_dict]))
        cad_prob = float(clin_bundle["pipeline"].predict_proba(X_feat)[0, 1])

        rad_prob = st.session_state.get("latest_radiology", {}).get("pneumonia_prob")
        ecg_anom = st.session_state.get("latest_ecg_telemetry", {}).get("is_anomaly")
        ecg_mse = st.session_state.get("latest_ecg_telemetry", {}).get("mse")

        fusion = compute_multimodal_triage_score(cad_prob, pt_dict, cxr_pneumonia_probability=rad_prob, ecg_is_anomaly=ecg_anom, ecg_reconstruction_mse=ecg_mse)
        hf_pred = infer_pretrained_clinical(pt_dict)

        k1, k2, k3 = st.columns(3)
        k1.metric("Clinical Ensemble Prob", f"{cad_prob*100:.1f}%")
        k2.metric("Multimodal Fused Score", f"{fusion['fused_risk_score']*100:.1f}%")
        k3.metric("HF RF Consensus", f"{hf_pred['risk_probability']*100:.1f}%")
        st.info(f"**Triage Tier:** {fusion['triage_tier']} ({fusion['recommended_urgency']})")

with tab4:
    st.subheader("Hugging Face Pre-Trained Medical Hub & Live Search")
    catalog = get_curated_catalog()
    st.dataframe(pd.DataFrame(catalog)[["repo_id", "domain", "display_name", "architecture", "input_spec"]], use_container_width=True, hide_index=True)
    if os.path.exists("reports/pretrained_benchmarks.json"):
        with open("reports/pretrained_benchmarks.json", "r") as f:
            st.json(json.load(f))
    q_hf = st.text_input("Search Hugging Face Hub for Medical Models:", value="chest xray pneumonia")
    if st.button("Search Hugging Face Hub"):
        st.dataframe(pd.DataFrame(search_hf_medical_models(q_hf, limit=8)), use_container_width=True, hide_index=True)

with tab5:
    st.subheader("Live Empirical Benchmark Summary & Report Export")
    if os.path.exists("reports/metrics_summary.json"):
        with open("reports/metrics_summary.json", "r") as f:
            ms = json.load(f)
        st.json(ms)
    else:
        st.info("Run `python demo.py` to generate full empirical benchmarks.")
