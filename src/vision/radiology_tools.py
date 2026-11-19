"""
Radiological Image Windowing, Quantitative Radiomics & Severity Analysis
------------------------------------------------------------------------
Implements clinical radiology tools:
- DICOM-style Windowing presets (Lung, Bone, Mediastinal, Fluoroscopy Invert)
- Quantitative Alveolar Infiltrate Burden & Severity Scoring
- 14 Quantitative Radiomic Texture, Symmetry & Parenchymal Biomarkers
"""

from typing import Dict, Any, List
import numpy as np


RADIOMIC_FEATURE_NAMES: List[str] = [
    "intensity_mean",
    "intensity_std",
    "intensity_iqr",
    "intensity_p10",
    "intensity_p90",
    "intensity_p95",
    "intensity_skewness",
    "intensity_kurtosis",
    "shannon_entropy",
    "left_right_asymmetry",
    "basilar_apical_ratio",
    "sobel_edge_density",
    "central_peripheral_ratio",
    "consolidation_opacity_fraction",
]


def apply_radiology_window(
    image: np.ndarray,
    window_type: str = "Lung Window"
) -> np.ndarray:
    """
    Applies standard diagnostic Hounsfield/radiographic window leveling.
    """
    img = np.asarray(image, dtype=np.float32)
    norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

    if window_type == "Lung Window":
        w_min, w_max = 0.05, 0.65
        windowed = np.clip((norm - w_min) / (w_max - w_min), 0.0, 1.0)
    elif window_type == "Bone Window":
        w_min, w_max = 0.45, 0.95
        windowed = np.clip((norm - w_min) / (w_max - w_min), 0.0, 1.0)
    elif window_type == "Mediastinum / Soft-Tissue":
        w_min, w_max = 0.25, 0.80
        windowed = np.clip((norm - w_min) / (w_max - w_min), 0.0, 1.0)
    elif window_type == "Inverted Radiograph (Fluoroscopy)":
        windowed = 1.0 - norm
    else:
        windowed = norm

    return (windowed * 255.0).astype(np.uint8)


def extract_radiomic_features(image: np.ndarray) -> Dict[str, float]:
    """
    Extracts 14 quantitative first-order statistical, spatial asymmetry,
    and gradient texture radiomic biomarkers from a 2D grayscale chest radiograph.
    """
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    if arr.max() > 1.0:
        arr = arr / 255.0

    h, w = arr.shape
    flat = arr.ravel()
    mean_v = float(np.mean(flat))
    std_v = float(np.std(flat) + 1e-8)
    q25, q75 = np.percentile(flat, [25, 75])
    p10, p90, p95 = np.percentile(flat, [10, 90, 95])

    centered = (flat - mean_v) / std_v
    skew = float(np.mean(centered**3))
    kurt = float(np.mean(centered**4) - 3.0)

    # Shannon entropy over 32 intensity bins
    hist, _ = np.histogram(flat, bins=32, range=(0.0, 1.0))
    probs = hist.astype(np.float64) / (hist.sum() + 1e-12)
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs)))

    # Spatial lung zone ratios
    mid_w = w // 2
    right_lung_roi = arr[int(0.18 * h):int(0.75 * h), int(0.12 * w):mid_w]
    left_lung_roi = arr[int(0.18 * h):int(0.75 * h), mid_w:int(0.88 * w)]
    lr_asym = float(abs(np.mean(right_lung_roi) - np.mean(left_lung_roi)))

    apical = arr[int(0.15 * h):int(0.45 * h), int(0.15 * w):int(0.85 * w)]
    basilar = arr[int(0.48 * h):int(0.80 * h), int(0.15 * w):int(0.85 * w)]
    basilar_apical = float((np.mean(basilar) + 1e-4) / (np.mean(apical) + 1e-4))

    # Gradient texture density (finite differences)
    gx = np.abs(np.diff(arr, axis=1, prepend=arr[:, :1]))
    gy = np.abs(np.diff(arr, axis=0, prepend=arr[:1, :]))
    edge_density = float(np.mean(np.hypot(gx, gy)))

    # Central mediastinal vs lateral lung parenchyma ratio
    center_roi = arr[int(0.25 * h):int(0.75 * h), int(0.36 * w):int(0.64 * w)]
    central_periph = float((np.mean(center_roi) + 1e-4) / (mean_v + 1e-4))

    # High-opacity consolidation fraction within lung zones (> 0.55 intensity)
    lung_zone = arr[int(0.20 * h):int(0.78 * h), int(0.12 * w):int(0.88 * w)]
    opacity_frac = float(np.mean(lung_zone > 0.52))

    return {
        "intensity_mean": round(mean_v, 5),
        "intensity_std": round(std_v, 5),
        "intensity_iqr": round(float(q75 - q25), 5),
        "intensity_p10": round(float(p10), 5),
        "intensity_p90": round(float(p90), 5),
        "intensity_p95": round(float(p95), 5),
        "intensity_skewness": round(skew, 5),
        "intensity_kurtosis": round(kurt, 5),
        "shannon_entropy": round(entropy, 5),
        "left_right_asymmetry": round(lr_asym, 5),
        "basilar_apical_ratio": round(basilar_apical, 5),
        "sobel_edge_density": round(edge_density, 5),
        "central_peripheral_ratio": round(central_periph, 5),
        "consolidation_opacity_fraction": round(opacity_frac, 5),
    }


def compute_consolidation_severity(
    heatmap: np.ndarray,
    threshold: float = 0.45
) -> Dict[str, Any]:
    """
    Quantifies radiological severity by measuring the spatial area of high-activation
    pathological focus relative to total thoracic lung parenchyma.
    """
    heatmap = np.asarray(heatmap, dtype=np.float32)
    total_pixels = heatmap.size
    salient_pixels = np.sum(heatmap >= threshold)
    infiltrate_percentage = float((salient_pixels / max(1, total_pixels)) * 100.0)

    if infiltrate_percentage < 3.0:
        severity = "Normal / Clear Parenchyma"
        grade = "Grade 0"
        urgency = "Routine Outpatient"
    elif infiltrate_percentage < 12.0:
        severity = "Mild Patchy Infiltrate"
        grade = "Grade 1 (Early/Focal)"
        urgency = "Standard Clinical Follow-up"
    elif infiltrate_percentage < 25.0:
        severity = "Moderate Lobar Consolidation"
        grade = "Grade 2 (Segmental Infiltrate)"
        urgency = "Urgent Medical Attention"
    else:
        severity = "Severe Diffuse Alveolar Consolidation"
        grade = "Grade 3 (Bilateral/Multilobar)"
        urgency = "Immediate Inpatient Admission Recommended"

    return {
        "infiltrate_burden_percentage": round(infiltrate_percentage, 1),
        "radiological_severity": severity,
        "radiological_grade": grade,
        "clinical_triage_urgency": urgency,
        "salient_pixel_count": int(salient_pixels),
    }
