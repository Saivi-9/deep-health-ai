"""
Deep Health AI - Multi-Modal Clinical Diagnostic Web Application
----------------------------------------------------------------
Interactive web interface for:
1. Chest Radiograph Diagnosis with Grad-CAM Explainability & MC Dropout Uncertainty.
2. Lead-II ECG Cardiac Arrhythmia Anomaly Screening.
3. Live Benchmark Analytics & Model Cards.
4. Printable Clinical Patient Report Generation.

Launch via:
    streamlit run app.py
"""

import os
import json
import base64
from datetime import datetime
from io import BytesIO

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.vision.model import ResidualChestCNN
from src.vision.gradcam import GradCAM
from src.vision.uncertainty import MonteCarloDropoutEstimator
from src.ecg.model import Conv1dAutoencoder
from src.ecg.anomaly_detector import ECGAnomalyDetector
from src.utils.plotting import plot_gradcam_overlay, plot_ecg_reconstructions

# Page configuration
st.set_page_config(
    page_title="Deep Health AI - Clinical Diagnostic Suite",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom healthcare aesthetic styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .status-normal {
        color: #16A34A;
        font-weight: 700;
        font-size: 1.3rem;
    }
    .status-pathology {
        color: #DC2626;
        font-weight: 700;
        font-size: 1.3rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_vision_model(weights_path: str = "saved_models/best_chest_cnn.pt", device: str = "cpu"):
    """Loads and caches the Residual Chest CNN model."""
    model = ResidualChestCNN(in_channels=1, num_classes=2)
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=torch.device(device))
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@st.cache_resource
def load_ecg_detector(
    weights_path: str = "saved_models/best_ecg_autoencoder.pt",
    metrics_path: str = "reports/ecg_metrics.json",
    device: str = "cpu"
):
    """Loads and caches the 1D ECG Autoencoder and threshold."""
    model = Conv1dAutoencoder(in_channels=1)
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=torch.device(device))
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    threshold = 0.01037
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                data = json.load(f)
                threshold = float(data.get("threshold", 0.01037))
        except Exception:
            pass

    detector = ECGAnomalyDetector(model=model, threshold=threshold, device=device)
    return detector, threshold


# Sidebar configuration
st.sidebar.image("https://img.icons8.com/fluency/96/stethoscope.png", width=64)
st.sidebar.title("System Controls")
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"⚡ **Compute Hardware:** `{device.upper()}`")

st.sidebar.markdown("---")
st.sidebar.subheader("Diagnostic Parameters")
mc_samples = st.sidebar.slider("MC Dropout Uncertainty Samples", min_value=5, max_value=50, value=20, step=5)
enable_gradcam = st.sidebar.checkbox("Compute Grad-CAM Attention Heatmaps", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("🩺 **Deep Health AI Platform** v1.2.0\nNative PyTorch • Dual-Modality Clinical Diagnostics")

# Header
st.markdown('<div class="main-header">🩺 Deep Health AI: Clinical Diagnostic Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Multi-modal deep learning for 2D Thoracic Radiograph interpretation & 1D ECG Arrhythmia detection with explainability and epistemic uncertainty quantification.</div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🫁 Chest Radiograph Interpretation",
    "💓 ECG Arrhythmia Screening",
    "📊 Benchmarks & Model Cards",
    "📄 Patient Report Generator"
])

# ==========================================
# TAB 1: Chest Radiograph Diagnostics
# ==========================================
with tab1:
    st.subheader("Thoracic Radiograph Diagnostic Assistant")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Input Radiograph Scan")
        input_mode = st.radio("Input Source:", ["Preset Sample Scan", "Upload Custom Radiograph (.png, .jpg)"], horizontal=True)

        img_to_diagnose = None
        selected_sample_name = ""

        if input_mode == "Preset Sample Scan":
            sample_options = {
                "Pneumonia Sample (Alveolar Infiltrate)": "data/vision/pneumonia/pneumonia_0005.png",
                "Normal Healthy Lung Sample": "data/vision/normal/normal_0005.png",
                "Pneumonia Sample 2": "data/vision/pneumonia/pneumonia_0012.png",
                "Normal Healthy Lung Sample 2": "data/vision/normal/normal_0012.png"
            }
            sample_choice = st.selectbox("Choose Sample Case:", list(sample_options.keys()))
            sample_path = sample_options[sample_choice]
            selected_sample_name = sample_choice

            if os.path.exists(sample_path):
                img_to_diagnose = Image.open(sample_path).convert("L")
                st.image(img_to_diagnose, caption=f"Selected: {sample_choice}", width=320)
            else:
                st.warning(f"Sample scan not found at {sample_path}. Please generate datasets first via `python data/generate_datasets.py`.")
        else:
            uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                img_to_diagnose = Image.open(uploaded_file).convert("L")
                selected_sample_name = uploaded_file.name
                st.image(img_to_diagnose, caption="Uploaded Radiograph", width=320)

        run_vision_btn = st.button("🔍 Run Thoracic Diagnostic Analysis", type="primary", use_container_width=True)

    with col2:
        st.markdown("#### Diagnostic Findings & Explainability")
        if run_vision_btn and img_to_diagnose is not None:
            with st.spinner("Executing ResidualChestCNN forward pass, MC Dropout sampling, and Grad-CAM backpropagation..."):
                vision_model = load_vision_model(device=device)

                # Preprocess image
                eval_transform = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485], std=[0.229])
                ])
                tensor_img = eval_transform(img_to_diagnose).unsqueeze(0).to(device)

                # 1. Standard prediction
                with torch.no_grad():
                    logits = vision_model(tensor_img)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                    pred_idx = int(np.argmax(probs))

                class_names = ["Normal Healthy Lungs", "Pathological Consolidation / Pneumonia"]
                pred_label = class_names[pred_idx]
                confidence = float(probs[pred_idx])

                # 2. Uncertainty via MC Dropout
                uncertainty_estimator = MonteCarloDropoutEstimator(vision_model, n_samples=mc_samples, device=device)
                uncertainty_res = uncertainty_estimator.estimate_uncertainty(tensor_img)

                # 3. Grad-CAM
                gradcam_heatmap = None
                if enable_gradcam:
                    gradcam = GradCAM(vision_model, vision_model.layer4)
                    gradcam_heatmap, _, _ = gradcam.generate_heatmap(tensor_img, target_class=pred_idx)
                    gradcam.remove_hooks()

                # Display Results
                if pred_idx == 1:
                    st.error(f"🚨 **Primary Diagnosis:** {pred_label}")
                else:
                    st.success(f"✅ **Primary Diagnosis:** {pred_label}")

                # Metrics metrics columns
                mcol1, mcol2, mcol3 = st.columns(3)
                mcol1.metric("Model Confidence", f"{confidence * 100:.1f}%")
                mcol2.metric("Epistemic Variance", f"{uncertainty_res['predictive_variance']:.4f}")
                mcol3.metric("Shannon Entropy", f"{uncertainty_res['normalized_entropy']:.3f}")

                # Clinical referral status
                if uncertainty_res["human_referral_mandatory"]:
                    st.warning(f"⚠️ **Actionable Flag:** {uncertainty_res['uncertainty_risk_level']}")
                else:
                    st.info(f"ℹ️ **Actionable Flag:** {uncertainty_res['uncertainty_risk_level']}")

                # Probabilities breakdown
                st.markdown("**Diagnostic Probability Distribution:**")
                st.progress(float(probs[1]), text=f"Pneumonia: {probs[1]*100:.1f}% | Normal: {probs[0]*100:.1f}%")

                # Visual Explainability Overlays
                if gradcam_heatmap is not None:
                    st.markdown("#### 🗺️ Grad-CAM Class Activation Heatmap Overlay")
                    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
                    raw_np = np.array(img_to_diagnose.resize((128, 128)), dtype=np.float32) / 255.0

                    axes[0].imshow(raw_np, cmap="gray")
                    axes[0].set_title("Input Radiograph", fontsize=10, fontweight="bold")
                    axes[0].axis("off")

                    axes[1].imshow(gradcam_heatmap, cmap="jet", vmin=0, vmax=1)
                    axes[1].set_title("Salient Activation Heatmap", fontsize=10, fontweight="bold")
                    axes[1].axis("off")

                    axes[2].imshow(raw_np, cmap="gray")
                    axes[2].imshow(gradcam_heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
                    axes[2].set_title("Fused Diagnostic Overlay", fontsize=10, fontweight="bold")
                    axes[2].axis("off")

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                # Store into session state for report generation
                st.session_state["latest_vision"] = {
                    "sample_name": selected_sample_name,
                    "diagnosis": pred_label,
                    "confidence": confidence,
                    "probs": probs.tolist(),
                    "uncertainty": uncertainty_res,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
        elif img_to_diagnose is None:
            st.info("👈 Please select or upload a chest radiograph and click 'Run Thoracic Diagnostic Analysis'.")

# ==========================================
# TAB 2: ECG Arrhythmia Screening
# ==========================================
with tab2:
    st.subheader("Lead-II ECG Cardiac Arrhythmia Anomaly Detector")
    ecg_col1, ecg_col2 = st.columns([1, 1])

    with ecg_col1:
        st.markdown("#### Input Cardiac Signal")
        ecg_input_mode = st.radio("Signal Source:", ["Preset Cardiac Beat Sample", "Upload 1D ECG Array (.npy, .csv)"], horizontal=True)

        signal_to_screen = None
        ecg_sample_desc = ""

        if ecg_input_mode == "Preset Cardiac Beat Sample":
            ecg_sample_options = {
                "Premature Ventricular Contraction (Arrhythmia)": 1,
                "Normal Sinus Rhythm (Healthy Conduction)": 0
            }
            ecg_choice = st.selectbox("Select Waveform Morphology:", list(ecg_sample_options.keys()))
            target_class_flag = ecg_sample_options[ecg_choice]
            ecg_sample_desc = ecg_choice

            test_npy = "data/ecg/ecg_test.npy"
            test_lbls_npy = "data/ecg/ecg_test_labels.npy"
            if os.path.exists(test_npy) and os.path.exists(test_lbls_npy):
                test_signals = np.load(test_npy)
                test_labels = np.load(test_lbls_npy)
                match_indices = np.where(test_labels == target_class_flag)[0]
                if len(match_indices) > 0:
                    signal_to_screen = test_signals[match_indices[0]]
            else:
                st.warning("ECG dataset not found in `data/ecg/`. Please run dataset generator first.")
        else:
            uploaded_ecg = st.file_uploader("Upload 1D ECG waveform", type=["npy", "csv"])
            if uploaded_ecg is not None:
                if uploaded_ecg.name.endswith(".npy"):
                    loaded_arr = np.load(uploaded_ecg)
                else:
                    loaded_arr = np.loadtxt(uploaded_ecg, delimiter=",")
                if loaded_arr.ndim > 1:
                    loaded_arr = loaded_arr[0]
                signal_to_screen = loaded_arr[:256]
                ecg_sample_desc = uploaded_ecg.name

        if signal_to_screen is not None:
            fig, ax = plt.subplots(figsize=(7, 2.8))
            ax.plot(signal_to_screen, color="#1E3A8A", lw=2, label="Lead-II Voltage")
            ax.set_title("Input Cardiac Waveform (256 samples)", fontsize=10, fontweight="bold")
            ax.set_xlabel("Time (samples @ 250Hz)")
            ax.set_ylabel("Voltage (mV)")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        run_ecg_btn = st.button("💓 Run Cardiac Anomaly Screening", type="primary", use_container_width=True)

    with ecg_col2:
        st.markdown("#### Reconstruction Dynamics & Anomaly Assessment")
        if run_ecg_btn and signal_to_screen is not None:
            with st.spinner("Executing Conv1dAutoencoder reconstruction and residual anomaly analysis..."):
                detector, threshold = load_ecg_detector(device=device)
                result = detector.predict(signal_to_screen)

                is_anomaly = result["is_anomaly"]
                mse_error = result["reconstruction_error"]
                reconstructed = result["reconstructed_signal"]

                if is_anomaly:
                    st.error(f"🚨 **Diagnosis:** {result['diagnosis']}")
                else:
                    st.success(f"✅ **Diagnosis:** {result['diagnosis']}")

                # Metric columns
                ecol1, ecol2, ecol3 = st.columns(3)
                ecol1.metric("Reconstruction MSE", f"{mse_error:.5f}")
                ecol2.metric("Decision Boundary (θ)", f"{threshold:.5f}")
                ecol3.metric("Arrhythmia Flag", "TRUE (High Risk)" if is_anomaly else "FALSE (Normal)")

                # Reconstruction Plot
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)

                ax1.plot(signal_to_screen, color="#1F77B4", label="Original Signal", lw=2)
                ax1.plot(reconstructed, color="#2CA02C", linestyle="--", label="Autoencoder Reconstruction", lw=2)
                ax1.set_ylabel("Normalized Voltage")
                ax1.set_title("Signal vs Latent Manifold Reconstruction", fontsize=11, fontweight="bold")
                ax1.legend(loc="upper right")
                ax1.grid(True, linestyle="--", alpha=0.5)

                residual = (signal_to_screen - reconstructed) ** 2
                res_color = "#DC2626" if is_anomaly else "#4B5563"
                ax2.plot(residual, color=res_color, label="Squared Error Residuals", lw=1.8)
                ax2.axhline(threshold, color="red", linestyle=":", lw=1.5, label=f"Anomaly Threshold ({threshold:.4f})")
                ax2.set_xlabel("Time (samples)")
                ax2.set_ylabel("Error Residual")
                ax2.legend(loc="upper right")
                ax2.grid(True, linestyle="--", alpha=0.5)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # Store in session state for report
                st.session_state["latest_ecg"] = {
                    "sample_name": ecg_sample_desc,
                    "diagnosis": result["diagnosis"],
                    "is_anomaly": is_anomaly,
                    "reconstruction_mse": mse_error,
                    "threshold": threshold,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
        elif signal_to_screen is None:
            st.info("👈 Please select or upload an ECG waveform and click 'Run Cardiac Anomaly Screening'.")

# ==========================================
# TAB 3: Benchmarks & Model Cards
# ==========================================
with tab3:
    st.subheader("Clinical Deep Learning Benchmark & Model Specifications")

    bcol1, bcol2 = st.columns(2)
    with bcol1:
        st.markdown("### 🫁 Vision Model: `ResidualChestCNN`")
        st.markdown("""
        - **Architecture:** Hierarchical 4-stage Residual Convolutional Network with identity shortcuts.
        - **Attention:** Spatial Attention Module (CBAM-inspired) dynamic spatial gating.
        - **Explainability:** Grad-CAM backpropagation on 256-channel `layer4` conv layer.
        - **Uncertainty:** Epistemic uncertainty estimation via Monte Carlo Dropout ($N=20$).
        - **Input Resolution:** $128 \\times 128$ grayscale radiograph.
        """)
        st.markdown("""
        | Metric | Evaluation Result |
        | :--- | :--- |
        | **Test Accuracy** | **100.00%** |
        | **Sensitivity (Recall)** | **100.00%** |
        | **Specificity** | **100.00%** |
        | **ROC-AUC** | **1.0000** |
        """)

    with bcol2:
        st.markdown("### 💓 ECG Model: `Conv1dAutoencoder`")
        st.markdown("""
        - **Architecture:** 1D Convolutional Autoencoder with latent bottleneck manifold.
        - **Supervision:** Unsupervised learning of healthy Normal Sinus Rhythm (NSR).
        - **Anomaly Detection:** Reconstruction Mean Squared Error ($L_{\\text{recon}} = ||x - \\hat{x}||^2$).
        - **Threshold:** Calibrated to 97.5th percentile of healthy validation beats ($\\theta = 0.01037$).
        - **Temporal Resolution:** 256 temporal samples (1 cardiac cycle @ 250 Hz).
        """)
        st.markdown("""
        | Metric | Evaluation Result |
        | :--- | :--- |
        | **Reconstruction ROC-AUC** | **0.9714** |
        | **Sinus Specificity** | **99.00%** |
        | **Arrhythmia Precision (PPV)** | **98.68%** |
        | **Overall Accuracy** | **87.00%** |
        """)

    st.markdown("---")
    st.markdown("### 📈 Published Validation Diagnostics & Curves")
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        if os.path.exists("reports/figures/vision_roc_pr_curves.png"):
            st.image("reports/figures/vision_roc_pr_curves.png", caption="Vision ROC & Precision-Recall Curves")
        if os.path.exists("reports/figures/vision_confusion_matrix.png"):
            st.image("reports/figures/vision_confusion_matrix.png", caption="Diagnostic Confusion Matrix")
    with pcol2:
        if os.path.exists("reports/figures/ecg_anomaly_histogram_and_threshold.png"):
            st.image("reports/figures/ecg_anomaly_histogram_and_threshold.png", caption="ECG Reconstruction Error Distribution & Threshold")
        if os.path.exists("reports/figures/ecg_reconstruction_examples.png"):
            st.image("reports/figures/ecg_reconstruction_examples.png", caption="ECG Normal vs Arrhythmic Reconstructions")

# ==========================================
# TAB 4: Patient Report Generator
# ==========================================
with tab4:
    st.subheader("Automated Clinical Patient Report Export")
    st.markdown("Consolidates multi-modal findings into an official clinical consultation report.")

    rcol1, rcol2 = st.columns([1, 1])
    with rcol1:
        patient_id = st.text_input("Patient Medical Record Number (MRN):", value="MRN-2026-9481")
        physician_name = st.text_input("Reviewing Physician / Clinician:", value="Dr. S. Timmons, MD (Cardiothoracic Radiology)")
        patient_notes = st.text_area("Clinical Observations & Notes:", value="Patient presented with acute dyspnea, mild pleuritic chest discomfort, and intermittent palpitations.")

    with rcol2:
        st.markdown("#### Report Summary Preview")
        has_vision = "latest_vision" in st.session_state
        has_ecg = "latest_ecg" in st.session_state

        if not has_vision and not has_ecg:
            st.info("ℹ️ Run at least one diagnostic analysis in Tab 1 or Tab 2 to populate clinical findings.")
        else:
            if has_vision:
                v = st.session_state["latest_vision"]
                st.write(f"**Chest Radiograph:** {v['diagnosis']} (Confidence: {v['confidence']*100:.1f}%)")
                st.write(f"• Epistemic Variance: `{v['uncertainty']['predictive_variance']:.4f}` ({v['uncertainty']['uncertainty_risk_level']})")

            if has_ecg:
                e = st.session_state["latest_ecg"]
                st.write(f"**Cardiac Rhythm:** {e['diagnosis']} (Reconstruction MSE: `{e['reconstruction_mse']:.5f}`)")

            # Generate HTML Report
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 40px; color: #1e293b; }}
                    .header {{ border-bottom: 2px solid #0284c7; padding-bottom: 12px; margin-bottom: 24px; }}
                    .title {{ font-size: 24px; font-weight: bold; color: #0f172a; }}
                    .meta {{ font-size: 13px; color: #64748b; margin-top: 6px; }}
                    .section {{ margin-top: 24px; padding: 16px; background-color: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0; }}
                    .section-title {{ font-size: 16px; font-weight: bold; color: #0284c7; margin-bottom: 8px; }}
                    .badge-pos {{ background: #fee2e2; color: #b91c1c; padding: 4px 8px; border-radius: 4px; font-weight: bold; }}
                    .badge-neg {{ background: #dcfce7; color: #15803d; padding: 4px 8px; border-radius: 4px; font-weight: bold; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <div class="title">🩺 Deep Health AI - Clinical Diagnostic Consultation</div>
                    <div class="meta">Patient MRN: {patient_id} | Physician: {physician_name} | Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
                </div>

                <div class="section">
                    <div class="section-title">Clinical Indication & Notes</div>
                    <p>{patient_notes}</p>
                </div>

                {'<div class="section"><div class="section-title">Thoracic Radiograph Assessment (ResidualChestCNN + Grad-CAM)</div><p><b>Diagnosis:</b> ' + v['diagnosis'] + '</p><p><b>Model Confidence:</b> ' + f"{v['confidence']*100:.2f}%" + '</p><p><b>Uncertainty Level:</b> ' + v['uncertainty']['uncertainty_risk_level'] + '</p></div>' if has_vision else ''}

                {'<div class="section"><div class="section-title">Lead-II Electrocardiogram Screening (Conv1dAutoencoder)</div><p><b>Cardiac Rhythm:</b> ' + e['diagnosis'] + '</p><p><b>Reconstruction MSE:</b> ' + f"{e['reconstruction_mse']:.6f}" + ' (Threshold: ' + f"{e['threshold']:.6f}" + ')</p></div>' if has_ecg else ''}

                <div style="margin-top: 32px; font-size: 11px; color: #94a3b8; text-align: center;">
                    Generated by Deep Health AI Clinical Decision Support System. For investigational and diagnostic support purposes.
                </div>
            </body>
            </html>
            """

            st.download_button(
                label="📥 Download Clinical Patient Report (.html)",
                data=html_content,
                file_name=f"Clinical_Report_{patient_id}_{datetime.now().strftime('%Y%m%d')}.html",
                mime="text/html",
                use_container_width=True
            )
