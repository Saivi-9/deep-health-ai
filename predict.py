"""
Deep Health AI - Multi-Modal Clinical Inference CLI (v2.0)
----------------------------------------------------------
Runs deep learning inference on Chest Radiographs, 1D ECG waveforms, or Clinical EHR JSON.

Usage Examples:
    python predict.py --image data/vision/pneumonia/pneumonia_0005.png --gradcam --arch residual_cnn
    python predict.py --ecg data/ecg/ecg_test.npy
"""

import os
import argparse
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.vision.model import ResidualChestCNN, TransferChestClassifier
from src.vision.gradcam import GradCAM
from src.vision.radiology_tools import extract_radiomic_features
from src.ecg.model import Conv1dAutoencoder, ECGResNet1D
from src.ecg.anomaly_detector import ECGAnomalyDetector
from src.ecg.telemetry import analyze_ecg_telemetry
from src.utils.plotting import plot_gradcam_overlay
from src.utils.logger import get_logger

logger = get_logger("PredictCLI")


def predict_image(
    image_path: str,
    model_path: str = "saved_models/best_chest_cnn.pt",
    generate_gradcam: bool = False,
    arch: str = "residual_cnn",
):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Vision model weights not found at {model_path}. Train the model first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if arch == "resnet18":
        model = TransferChestClassifier(arch="resnet18", num_classes=2, pretrained=False)
    elif arch == "mobilenet_v2":
        model = TransferChestClassifier(arch="mobilenet_v2", num_classes=2, pretrained=False)
    else:
        model = ResidualChestCNN(in_channels=1, num_classes=2)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    class_names = ["Normal", "Pathology_Pneumonia"]
    img = Image.open(image_path).convert("L")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    tensor_img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor_img)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))

    radiomics = extract_radiomic_features(np.asarray(img.resize((128, 128)), dtype=np.float32))
    confidence = float(probs[pred_idx])
    pred_label = class_names[pred_idx]

    print("\n" + "=" * 55)
    print(f"CHEST RADIOGRAPH CLINICAL DIAGNOSTIC REPORT ({arch})")
    print("=" * 55)
    print(f"File Path:             {image_path}")
    print(f"Predicted Class:       {pred_label}")
    print(f"Confidence:            {confidence * 100:.2f}%")
    print(f"Normal Probability:    {probs[0] * 100:.2f}%")
    print(f"Pneumonia Probability: {probs[1] * 100:.2f}%")
    print(f"Opacity Fraction:      {radiomics['consolidation_opacity_fraction']:.4f}")
    print("=" * 55)

    if generate_gradcam:
        target_layer = getattr(model, "gradcam_layer", model.layer4)
        gradcam = GradCAM(model, target_layer)
        heatmap, _, _ = gradcam.generate_heatmap(tensor_img, target_class=pred_idx)
        out_overlay = "reports/figures/prediction_gradcam_overlay.png"
        os.makedirs(os.path.dirname(out_overlay), exist_ok=True)
        raw_np = np.array(img.resize((128, 128)), dtype=np.float32) / 255.0
        plot_gradcam_overlay(
            image=raw_np,
            heatmap=heatmap,
            pred_class=pred_label,
            true_class="Unspecified",
            confidence=confidence,
            save_path=out_overlay,
        )
        gradcam.remove_hooks()
        print(f"Visual Explainability overlay saved to: {out_overlay}\n")


def predict_ecg(
    ecg_path: str,
    model_path: str = "saved_models/best_ecg_autoencoder.pt",
    metrics_path: str = "reports/ecg_metrics.json",
):
    if not os.path.exists(ecg_path):
        raise FileNotFoundError(f"ECG file not found at: {ecg_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ECG model weights not found at {model_path}. Train the model first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Conv1dAutoencoder(in_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    threshold = 0.05
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            data = json.load(f)
            threshold = data.get("threshold", 0.05)

    detector = ECGAnomalyDetector(model=model, threshold=threshold, device=device)
    signal = np.load(ecg_path) if ecg_path.endswith(".npy") else np.loadtxt(ecg_path, delimiter=",")
    if signal.ndim > 1:
        signal = signal[0]
    signal = signal[:256]

    result = detector.predict(signal)
    telemetry = analyze_ecg_telemetry(signal, sampling_rate=250)

    subtype_str = "N/A"
    if os.path.exists("saved_models/ecg_resnet1d.pt"):
        rn = ECGResNet1D(in_channels=1, num_classes=5)
        rn.load_state_dict(torch.load("saved_models/ecg_resnet1d.pt", map_location=device, weights_only=True))
        rn.to(device)
        rn.eval()
        with torch.no_grad():
            p = F.softmax(rn(torch.from_numpy(signal.astype(np.float32)).view(1, 1, -1).to(device)), dim=1).cpu().numpy()[0]
            classes = ["NSR", "PVC", "AFib", "ST-Elevation", "LBBB"]
            subtype_str = f"{classes[int(np.argmax(p))]} ({np.max(p)*100:.1f}%)"

    print("\n" + "=" * 55)
    print("ECG CARDIAC BIO-SIGNAL SCREENING REPORT")
    print("=" * 55)
    print(f"File Path:            {ecg_path}")
    print(f"Diagnosis:            {result['diagnosis']}")
    print(f"5-Class Rhythm:       {subtype_str}")
    print(f"Anomaly Detected:     {result['is_anomaly']}")
    print(f"Reconstruction MSE:   {result['reconstruction_error']:.6f}")
    print(f"Decision Threshold:   {result['threshold']:.6f}")
    print(f"QRS / QTc:            {telemetry['qrs_duration_ms']} ms / {telemetry['qtc_interval_ms']} ms")
    print("=" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Deep Health AI Inference Tool")
    parser.add_argument("--image", type=str, help="Path to chest radiograph image (.png, .jpg)")
    parser.add_argument("--arch", choices=["residual_cnn", "resnet18", "mobilenet_v2"], default="residual_cnn")
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM explainability heatmap")
    parser.add_argument("--ecg", type=str, help="Path to 1D ECG signal (.npy, .csv)")
    args = parser.parse_args()

    if args.image:
        ckpt = {
            "residual_cnn": "saved_models/best_chest_cnn.pt",
            "resnet18": "saved_models/vision_resnet18.pt",
            "mobilenet_v2": "saved_models/vision_mobilenetv2.pt",
        }[args.arch]
        predict_image(args.image, model_path=ckpt, generate_gradcam=args.gradcam, arch=args.arch)
    elif args.ecg:
        predict_ecg(args.ecg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
