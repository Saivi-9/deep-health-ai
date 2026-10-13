"""
Synthetic Clinical Dataset Generator for Deep Health AI
--------------------------------------------------------
Generates high-fidelity simulated biomedical datasets:
1. Chest Radiographs (Normal vs Pathological Opacity/Pneumonia)
2. Lead-II Electrocardiogram (ECG) waveforms (Normal Sinus Rhythm vs Arrhythmia / Ventricular Ectopy)
"""

import os
import json
import numpy as np
from PIL import Image

def generate_chest_xray(label: int, size: int = 128, rng: np.random.Generator = None) -> np.ndarray:
    """
    Simulates a 2D chest radiograph with realistic anatomical structures:
    thoracic cage, diaphragmatic domes, mediastinum/cardiac silhouette, ribs,
    and pathological alveolar consolidations if label == 1.
    """
    if rng is None:
        rng = np.random.default_rng()

    img = np.zeros((size, size), dtype=np.float32)
    y, x = np.ogrid[:size, :size]

    # Normalized coordinates [-1, 1]
    nx = (x - size / 2) / (size / 2)
    ny = (y - size / 2) / (size / 2)

    # 1. Background soft-tissue attenuation & body contour (oval thorax)
    body_mask = (nx**2 / 0.85**2 + (ny - 0.1)**2 / 0.9**2) <= 1.0
    img[body_mask] = 0.35 + 0.05 * rng.standard_normal(np.sum(body_mask))

    # 2. Lung fields (two dark, radiolucent cavities)
    # Right lung (anatomical right = image left)
    right_lung_mask = ((nx + 0.38)**2 / 0.24**2 + (ny + 0.05)**2 / 0.55**2) <= 1.0
    # Left lung (anatomical left = image right)
    left_lung_mask = ((nx - 0.38)**2 / 0.22**2 + (ny + 0.08)**2 / 0.50**2) <= 1.0

    img[right_lung_mask] = 0.08 + 0.03 * rng.standard_normal(np.sum(right_lung_mask))
    img[left_lung_mask] = 0.09 + 0.03 * rng.standard_normal(np.sum(left_lung_mask))

    # 3. Cardiac and mediastinal shadow (dense radio-opaque central structure)
    cardiac_mask = ((nx - 0.12)**2 / 0.28**2 + (ny - 0.25)**2 / 0.35**2) <= 1.0
    img[cardiac_mask] = np.maximum(img[cardiac_mask], 0.65 + 0.04 * rng.standard_normal(np.sum(cardiac_mask)))

    # Central mediastinal spine column
    spine_mask = (np.abs(nx) < 0.08) & (ny > -0.9) & (ny < 0.9)
    img[spine_mask] = np.maximum(img[spine_mask], 0.55 + 0.03 * rng.standard_normal(np.sum(spine_mask)))

    # 4. Diaphragmatic domes
    right_diaphragm = (ny > 0.45) & (nx < 0.0) & body_mask
    left_diaphragm = (ny > 0.50) & (nx >= 0.0) & body_mask
    img[right_diaphragm] = np.maximum(img[right_diaphragm], 0.50)
    img[left_diaphragm] = np.maximum(img[left_diaphragm], 0.52)

    # 5. Clavicles (collar bones)
    clavicle_right = (np.abs(ny - (-0.55 + 0.15 * nx)) < 0.04) & (nx < -0.1) & body_mask
    clavicle_left = (np.abs(ny - (-0.55 - 0.15 * nx)) < 0.04) & (nx > 0.1) & body_mask
    img[clavicle_right] = np.maximum(img[clavicle_right], 0.70)
    img[clavicle_left] = np.maximum(img[clavicle_left], 0.70)

    # 6. Rib cage shadows (horizontal-curved radio-opaque ribs)
    for rib_y in np.linspace(-0.4, 0.4, 6):
        curved_y = rib_y + 0.08 * (nx**2)
        rib_mask = (np.abs(ny - curved_y) < 0.035) & (np.abs(nx) > 0.12) & body_mask
        img[rib_mask] += 0.12

    # 7. Pulmonary vascular branching (hilar bronchovascular markings)
    hilar_branch = np.exp(-((nx + 0.2)**2 + (ny + 0.05)**2) / 0.05) * 0.15
    hilar_branch += np.exp(-((nx - 0.2)**2 + (ny + 0.05)**2) / 0.05) * 0.15
    img += hilar_branch

    # 8. Pathology: Alveolar consolidation / Pneumonia infiltrate
    if label == 1:
        # Patchy cloud-like opacity in lower right or lower left lung lobe
        patch_side = rng.choice([-0.35, 0.35])
        patch_y = rng.uniform(0.05, 0.35)
        radius_x = rng.uniform(0.12, 0.20)
        radius_y = rng.uniform(0.10, 0.18)

        consolidation_dist = ((nx - patch_side)**2 / radius_x**2 + (ny - patch_y)**2 / radius_y**2)
        consolidation_mask = consolidation_dist <= 1.0
        
        # Dense ground-glass opacity with granular high-frequency infiltrate texture
        intensity = (1.0 - consolidation_dist[consolidation_mask]) * rng.uniform(0.45, 0.65)
        infiltrate_texture = rng.uniform(0.8, 1.2, size=np.sum(consolidation_mask))
        img[consolidation_mask] += intensity * infiltrate_texture

    # Add realistic radiographic sensor noise & clamp [0, 1]
    img += 0.02 * rng.standard_normal((size, size))
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def generate_ecg_beat(is_anomaly: bool, length: int = 256, rng: np.random.Generator = None) -> np.ndarray:
    """
    Simulates a Lead-II ECG waveform using Gaussian wave components:
    P-wave, Q-dip, R-peak, S-dip, T-wave.
    If is_anomaly == True, generates Premature Ventricular Contractions (PVC),
    ST-segment elevation, or inverted abnormal morphology.
    """
    if rng is None:
        rng = np.random.default_rng()

    t = np.linspace(0, 1.0, length)
    baseline_wander = 0.05 * np.sin(2 * np.pi * 0.5 * t + rng.uniform(0, np.pi))

    if not is_anomaly:
        # Normal Sinus Rhythm (NSR)
        p_amp, p_center, p_width = rng.uniform(0.12, 0.18), 0.22, 0.025
        q_amp, q_center, q_width = rng.uniform(-0.15, -0.10), 0.38, 0.010
        r_amp, r_center, r_width = rng.uniform(1.10, 1.35), 0.42, 0.018
        s_amp, s_center, s_width = rng.uniform(-0.30, -0.20), 0.46, 0.012
        t_amp, t_center, t_width = rng.uniform(0.25, 0.35), 0.68, 0.055

        signal = (
            p_amp * np.exp(-((t - p_center) ** 2) / (2 * p_width ** 2)) +
            q_amp * np.exp(-((t - q_center) ** 2) / (2 * q_width ** 2)) +
            r_amp * np.exp(-((t - r_center) ** 2) / (2 * r_width ** 2)) +
            s_amp * np.exp(-((t - s_center) ** 2) / (2 * s_width ** 2)) +
            t_amp * np.exp(-((t - t_center) ** 2) / (2 * t_width ** 2))
        )
    else:
        # Cardiac Arrhythmia / Ventricular Ectopy (PVC or ST elevation)
        anomaly_type = rng.choice(["pvc", "st_elevation", "inverted_ectopic"])

        if anomaly_type == "pvc":
            # Premature Ventricular Contraction: No P wave, wide bizarre notched QRS, discordant T wave
            r_amp = rng.uniform(-1.2, -0.8) if rng.random() > 0.5 else rng.uniform(1.3, 1.7)
            r_center = rng.uniform(0.35, 0.45)
            r_width = rng.uniform(0.045, 0.070)  # Much wider QRS
            t_amp = -0.5 * np.sign(r_amp)
            t_center = r_center + 0.25
            t_width = 0.060
            signal = (
                r_amp * np.exp(-((t - r_center) ** 2) / (2 * r_width ** 2)) +
                t_amp * np.exp(-((t - t_center) ** 2) / (2 * t_width ** 2))
            )
        elif anomaly_type == "st_elevation":
            # Myocardial Infarction / Ischemic ST Elevation
            signal = (
                0.15 * np.exp(-((t - 0.22) ** 2) / (2 * 0.025 ** 2)) +
                -0.12 * np.exp(-((t - 0.38) ** 2) / (2 * 0.010 ** 2)) +
                1.25 * np.exp(-((t - 0.42) ** 2) / (2 * 0.018 ** 2)) +
                -0.10 * np.exp(-((t - 0.46) ** 2) / (2 * 0.012 ** 2)) +
                0.55 * np.exp(-((t - 0.60) ** 2) / (2 * 0.090 ** 2)) # Elevated broad ST-T plateau
            )
        else:
            # Inverted ectopic beat
            signal = (
                -0.95 * np.exp(-((t - 0.40) ** 2) / (2 * 0.035 ** 2)) +
                0.40 * np.exp(-((t - 0.65) ** 2) / (2 * 0.060 ** 2))
            )

    noise = 0.02 * rng.standard_normal(length)
    ecg = signal + baseline_wander + noise
    # Z-score normalization for standard biosignal processing
    ecg = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)
    return ecg.astype(np.float32)


def generate_all_datasets(base_dir: str = "."):
    rng = np.random.default_rng(42)
    print("=" * 60)
    print("GENERATING DEEP HEALTH AI DATASETS")
    print("=" * 60)

    # 1. Chest X-Ray Vision Dataset
    vision_dir = os.path.join(base_dir, "data", "vision")
    normal_dir = os.path.join(vision_dir, "normal")
    pneumonia_dir = os.path.join(vision_dir, "pneumonia")
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(pneumonia_dir, exist_ok=True)

    n_vision_samples_per_class = 200
    print(f"\n[1/2] Generating {n_vision_samples_per_class * 2} Chest X-Ray Radiographs (128x128)...")
    for i in range(n_vision_samples_per_class):
        # Normal
        norm_img = generate_chest_xray(label=0, size=128, rng=rng)
        Image.fromarray(norm_img).save(os.path.join(normal_dir, f"normal_{i:04d}.png"))

        # Pneumonia / Pathological
        pneu_img = generate_chest_xray(label=1, size=128, rng=rng)
        Image.fromarray(pneu_img).save(os.path.join(pneumonia_dir, f"pneumonia_{i:04d}.png"))

    print(f"  -> Saved {n_vision_samples_per_class} normal scans to {normal_dir}")
    print(f"  -> Saved {n_vision_samples_per_class} pneumonia scans to {pneumonia_dir}")

    # 2. ECG Bio-Signals Dataset
    ecg_dir = os.path.join(base_dir, "data", "ecg")
    os.makedirs(ecg_dir, exist_ok=True)

    print("\n[2/2] Generating Lead-II ECG Bio-Signals (256 samples/beat)...")
    n_normal_train = 800
    n_normal_val = 200
    n_test_normal = 100
    n_test_anomaly = 100

    normal_train = np.array([generate_ecg_beat(is_anomaly=False, length=256, rng=rng) for _ in range(n_normal_train)])
    normal_val = np.array([generate_ecg_beat(is_anomaly=False, length=256, rng=rng) for _ in range(n_normal_val)])

    test_normal = [generate_ecg_beat(is_anomaly=False, length=256, rng=rng) for _ in range(n_test_normal)]
    test_anomaly = [generate_ecg_beat(is_anomaly=True, length=256, rng=rng) for _ in range(n_test_anomaly)]

    test_data = np.array(test_normal + test_anomaly)
    test_labels = np.array([0] * n_test_normal + [1] * n_test_anomaly, dtype=np.int64)

    # Shuffle test set
    perm = rng.permutation(len(test_labels))
    test_data = test_data[perm]
    test_labels = test_labels[perm]

    np.save(os.path.join(ecg_dir, "ecg_train_normal.npy"), normal_train)
    np.save(os.path.join(ecg_dir, "ecg_val_normal.npy"), normal_val)
    np.save(os.path.join(ecg_dir, "ecg_test.npy"), test_data)
    np.save(os.path.join(ecg_dir, "ecg_test_labels.npy"), test_labels)

    metadata = {
        "vision": {
            "total_images": n_vision_samples_per_class * 2,
            "classes": ["Normal", "Pneumonia"],
            "resolution": [128, 128],
            "channels": 1
        },
        "ecg": {
            "train_normal_beats": n_normal_train,
            "val_normal_beats": n_normal_val,
            "test_total_beats": len(test_labels),
            "test_anomaly_ratio": float(np.mean(test_labels)),
            "signal_length": 256
        }
    }
    with open(os.path.join(base_dir, "data", "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  -> Saved {n_normal_train} train beats & {n_normal_val} val beats to {ecg_dir}")
    print(f"  -> Saved {len(test_labels)} test beats ({np.sum(test_labels)} anomalies) to {ecg_dir}")
    print("=" * 60)
    print("Dataset generation completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    generate_all_datasets()
