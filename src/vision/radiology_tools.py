"""
Radiological Image Windowing, Enhancement & Quantitative Severity Analysis
---------------------------------------------------------------------------
Implements clinical radiology tools:
- DICOM-style Windowing presets (Lung, Bone, Mediastinal)
- Contrast inversion (Fluoroscopic negative mode)
- Quantitative Alveolar Infiltrate Burden & Severity Scoring
"""

from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image, ImageOps, ImageEnhance


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
        # Enhances alveolar radiolucency and subtle infiltrates
        # Center: 0.35, Width: 0.60
        w_min = 0.05
        w_max = 0.65
        windowed = np.clip((norm - w_min) / (w_max - w_min), 0.0, 1.0)
    elif window_type == "Bone Window":
        # Highlights high-density thoracic cage, clavicles, and ribs
        w_min = 0.45
        w_max = 0.95
        windowed = np.clip((norm - w_min) / (w_max - w_min), 0.0, 1.0)
    elif window_type == "Mediastinum / Soft-Tissue":
        # Highlights cardiac silhouette, aortic arch, and hilum
        w_min = 0.25
        w_max = 0.80
        windowed = np.clip((norm - w_min) / (w_max - w_min), 0.0, 1.0)
    elif window_type == "Inverted Radiograph (Fluoroscopy)":
        windowed = 1.0 - norm
    else:
        windowed = norm

    return (windowed * 255.0).astype(np.uint8)


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
    infiltrate_percentage = float((salient_pixels / total_pixels) * 100.0)

    # Classify severity
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
        "salient_pixel_count": int(salient_pixels)
    }
