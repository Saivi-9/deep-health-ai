"""
Upgraded Multi-Regime Clinical Dataset Generator for Deep Health AI v2.0
------------------------------------------------------------------------
Generates high-variance biomedical datasets paired with real HF external cohorts:
1. Chest Radiographs (Normal with anatomical & mild vascular variance vs Bacterial/Viral Consolidation)
   - Scaled from 400 -> 2,000 high-variance scans (+ 1,148 real PneumoniaMNIST scans = 3,148 total)
2. Lead-II Electrocardiogram (ECG) waveforms (5 rhythm regimes: NSR, PVC, AFib, ST-Elevation, LBBB)
   - Scaled from 1,200 -> 5,500 waveforms (+ 10,000 real MIT-BIH beats = 15,500 ECG beats total)
"""

import os
import json
import numpy as np
import pandas as pd
from PIL import Image


def generate_chest_xray(
    label: int,
    size: int = 128,
    rng: np.random.Generator = None,
    subtype: str = None,
) -> np.ndarray:
    """
    Simulates a 2D chest radiograph with patient-to-patient anatomical variability,
    mild vascular/exposure mimics in normal scans, and diverse consolidation subtypes
    (focal lobar bacterial, diffuse interstitial viral, subtle early infiltrate) when label == 1.
    """
    if rng is None:
        rng = np.random.default_rng()

    img = np.zeros((size, size), dtype=np.float32)
    y, x = np.ogrid[:size, :size]

    # Patient-specific slight positional shift & exposure gain
    shift_x = float(rng.uniform(-0.04, 0.04))
    shift_y = float(rng.uniform(-0.04, 0.04))
    nx = (x - size / 2) / (size / 2) - shift_x
    ny = (y - size / 2) / (size / 2) - shift_y

    # 1. Background soft-tissue attenuation & oval thorax
    body_w = float(rng.uniform(0.80, 0.90))
    body_h = float(rng.uniform(0.86, 0.94))
    body_mask = (nx**2 / body_w**2 + (ny - 0.1)**2 / body_h**2) <= 1.0
    base_tissue = float(rng.uniform(0.30, 0.42))
    img[body_mask] = base_tissue + 0.055 * rng.standard_normal(np.sum(body_mask))

    # 2. Lung fields with individualized radii
    r_rx = float(rng.uniform(0.21, 0.26))
    l_rx = float(rng.uniform(0.20, 0.25))
    lung_ry = float(rng.uniform(0.50, 0.58))
    right_lung_mask = ((nx + 0.37)**2 / r_rx**2 + (ny + 0.05)**2 / lung_ry**2) <= 1.0
    left_lung_mask = ((nx - 0.37)**2 / l_rx**2 + (ny + 0.08)**2 / (lung_ry - 0.04)**2) <= 1.0

    parenchyma_darkness = float(rng.uniform(0.06, 0.14))
    img[right_lung_mask] = parenchyma_darkness + 0.035 * rng.standard_normal(np.sum(right_lung_mask))
    img[left_lung_mask] = (parenchyma_darkness + 0.01) + 0.035 * rng.standard_normal(np.sum(left_lung_mask))

    # 3. Cardiac and mediastinal shadow (slight cardiothoracic variation)
    cardiac_rx = float(rng.uniform(0.25, 0.33))
    cardiac_mask = ((nx - 0.12)**2 / cardiac_rx**2 + (ny - 0.25)**2 / 0.35**2) <= 1.0
    img[cardiac_mask] = np.maximum(
        img[cardiac_mask],
        float(rng.uniform(0.58, 0.72)) + 0.04 * rng.standard_normal(np.sum(cardiac_mask)),
    )

    # Mediastinal spine column
    spine_mask = (np.abs(nx) < 0.08) & (ny > -0.9) & (ny < 0.9)
    img[spine_mask] = np.maximum(img[spine_mask], 0.54 + 0.03 * rng.standard_normal(np.sum(spine_mask)))

    # 4. Diaphragmatic domes
    dome_level = float(rng.uniform(0.42, 0.52))
    right_diaphragm = (ny > dome_level) & (nx < 0.0) & body_mask
    left_diaphragm = (ny > dome_level + 0.05) & (nx >= 0.0) & body_mask
    img[right_diaphragm] = np.maximum(img[right_diaphragm], 0.50)
    img[left_diaphragm] = np.maximum(img[left_diaphragm], 0.52)

    # 5. Clavicles
    clavicle_right = (np.abs(ny - (-0.55 + 0.15 * nx)) < 0.04) & (nx < -0.1) & body_mask
    clavicle_left = (np.abs(ny - (-0.55 - 0.15 * nx)) < 0.04) & (nx > 0.1) & body_mask
    img[clavicle_right] = np.maximum(img[clavicle_right], 0.68)
    img[clavicle_left] = np.maximum(img[clavicle_left], 0.68)

    # 6. Rib cage shadows
    rib_strength = float(rng.uniform(0.09, 0.15))
    for rib_y in np.linspace(-0.4, 0.4, 6):
        curved_y = rib_y + 0.08 * (nx**2)
        rib_mask = (np.abs(ny - curved_y) < 0.035) & (np.abs(nx) > 0.12) & body_mask
        img[rib_mask] += rib_strength

    # 7. Pulmonary bronchovascular markings (stronger in ~20% of normal scans to mimic borderline CXRs)
    hilar_gain = float(rng.uniform(0.14, 0.30)) if (label == 0 and rng.random() < 0.25) else float(rng.uniform(0.11, 0.18))
    hilar_branch = np.exp(-((nx + 0.22)**2 + (ny + 0.05)**2) / 0.065) * hilar_gain
    hilar_branch += np.exp(-((nx - 0.22)**2 + (ny + 0.05)**2) / 0.065) * hilar_gain
    img += hilar_branch

    # 8. Pathology regimes
    if label == 1:
        chosen_sub = subtype or str(rng.choice(["lobar_bacterial", "interstitial_viral", "subtle_early"], p=[0.45, 0.35, 0.20]))
        if chosen_sub == "lobar_bacterial":
            patch_side = float(rng.choice([-0.35, 0.35]))
            patch_y = float(rng.uniform(0.02, 0.36))
            rx = float(rng.uniform(0.13, 0.22))
            ry = float(rng.uniform(0.11, 0.19))
            dist = (nx - patch_side)**2 / rx**2 + (ny - patch_y)**2 / ry**2
            mask = dist <= 1.0
            intensity = (1.0 - dist[mask]) * float(rng.uniform(0.32, 0.56))
            img[mask] += intensity * rng.uniform(0.75, 1.25, size=np.sum(mask))
        elif chosen_sub == "interstitial_viral":
            for side in (-0.34, 0.34):
                patch_y = float(rng.uniform(-0.08, 0.28))
                rx = float(rng.uniform(0.15, 0.22))
                ry = float(rng.uniform(0.18, 0.26))
                dist = (nx - side)**2 / rx**2 + (ny - patch_y)**2 / ry**2
                mask = dist <= 1.0
                intensity = (1.0 - dist[mask]) * float(rng.uniform(0.18, 0.34))
                img[mask] += intensity * rng.uniform(0.75, 1.30, size=np.sum(mask))
        else:
            # Subtle early focal infiltrate (harder boundary case)
            patch_side = float(rng.choice([-0.33, 0.33]))
            patch_y = float(rng.uniform(0.08, 0.34))
            dist = (nx - patch_side)**2 / 0.14**2 + (ny - patch_y)**2 / 0.12**2
            mask = dist <= 1.0
            intensity = (1.0 - dist[mask]) * float(rng.uniform(0.14, 0.26))
            img[mask] += intensity * rng.uniform(0.85, 1.15, size=np.sum(mask))

    # Realistic radiographic sensor noise + mild vignette
    noise_sigma = float(rng.uniform(0.02, 0.045))
    img += noise_sigma * rng.standard_normal((size, size))
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def generate_ecg_beat(
    is_anomaly: bool,
    length: int = 256,
    rng: np.random.Generator = None,
    rhythm_class: str = None,
) -> np.ndarray:
    """
    Simulates a 256-sample Lead-II ECG waveform across 5 clinical morphology classes:
    - 'nsr': Normal Sinus Rhythm
    - 'pvc': Premature Ventricular Contraction (wide notched QRS, discordant T)
    - 'st_elevation': Acute Ischemic ST-Segment Elevation / Hyperacute T
    - 'afib': Atrial Fibrillation (absent P-wave + fibrillatory f-wave baseline + irregular rate)
    - 'lbbb': Left Bundle Branch Block (wide slurred monophasic R wave >=130 ms)
    """
    if rng is None:
        rng = np.random.default_rng()

    t = np.linspace(0, 1.0, length)
    baseline_wander = float(rng.uniform(0.03, 0.08)) * np.sin(2 * np.pi * rng.uniform(0.35, 0.8) * t + rng.uniform(0, np.pi))

    if rhythm_class is None:
        rhythm_class = "nsr" if not is_anomaly else str(rng.choice(["pvc", "st_elevation", "afib", "lbbb"]))

    if rhythm_class == "nsr":
        r_shift = float(rng.uniform(-0.025, 0.025))
        p_amp = float(rng.uniform(0.10, 0.19))
        q_amp = float(rng.uniform(-0.18, -0.08))
        r_amp = float(rng.uniform(1.05, 1.40))
        s_amp = float(rng.uniform(-0.34, -0.18))
        t_amp = float(rng.uniform(0.22, 0.38))

        signal = (
            p_amp * np.exp(-((t - (0.22 + r_shift)) ** 2) / (2 * 0.026 ** 2)) +
            q_amp * np.exp(-((t - (0.38 + r_shift)) ** 2) / (2 * 0.011 ** 2)) +
            r_amp * np.exp(-((t - (0.42 + r_shift)) ** 2) / (2 * 0.018 ** 2)) +
            s_amp * np.exp(-((t - (0.46 + r_shift)) ** 2) / (2 * 0.013 ** 2)) +
            t_amp * np.exp(-((t - (0.68 + r_shift)) ** 2) / (2 * 0.056 ** 2))
        )
    elif rhythm_class == "pvc":
        r_amp = float(rng.uniform(-1.3, -0.85)) if rng.random() > 0.5 else float(rng.uniform(1.3, 1.75))
        r_center = float(rng.uniform(0.35, 0.46))
        r_width = float(rng.uniform(0.042, 0.072))
        t_amp = -0.48 * np.sign(r_amp)
        signal = (
            r_amp * np.exp(-((t - r_center) ** 2) / (2 * r_width ** 2)) +
            t_amp * np.exp(-((t - (r_center + 0.25)) ** 2) / (2 * 0.062 ** 2))
        )
    elif rhythm_class == "st_elevation":
        r_shift = float(rng.uniform(-0.02, 0.02))
        signal = (
            0.14 * np.exp(-((t - (0.22 + r_shift)) ** 2) / (2 * 0.025 ** 2)) +
            -0.12 * np.exp(-((t - (0.38 + r_shift)) ** 2) / (2 * 0.010 ** 2)) +
            1.20 * np.exp(-((t - (0.42 + r_shift)) ** 2) / (2 * 0.018 ** 2)) +
            -0.06 * np.exp(-((t - (0.46 + r_shift)) ** 2) / (2 * 0.012 ** 2)) +
            float(rng.uniform(0.42, 0.65)) * np.exp(-((t - (0.59 + r_shift)) ** 2) / (2 * 0.095 ** 2))
        )
    elif rhythm_class == "afib":
        r_shift = float(rng.uniform(-0.06, 0.06))
        fib_noise = 0.06 * np.sin(2 * np.pi * rng.uniform(5.5, 9.0) * t + rng.uniform(0, 2 * np.pi))
        fib_noise += 0.04 * np.cos(2 * np.pi * rng.uniform(10.0, 14.0) * t)
        signal = (
            fib_noise +
            -0.12 * np.exp(-((t - (0.37 + r_shift)) ** 2) / (2 * 0.011 ** 2)) +
            1.15 * np.exp(-((t - (0.41 + r_shift)) ** 2) / (2 * 0.017 ** 2)) +
            -0.25 * np.exp(-((t - (0.45 + r_shift)) ** 2) / (2 * 0.013 ** 2)) +
            0.22 * np.exp(-((t - (0.67 + r_shift)) ** 2) / (2 * 0.055 ** 2))
        )
    else:  # lbbb
        r_shift = float(rng.uniform(-0.02, 0.02))
        signal = (
            0.13 * np.exp(-((t - (0.21 + r_shift)) ** 2) / (2 * 0.025 ** 2)) +
            0.95 * np.exp(-((t - (0.39 + r_shift)) ** 2) / (2 * 0.032 ** 2)) +
            0.88 * np.exp(-((t - (0.45 + r_shift)) ** 2) / (2 * 0.032 ** 2)) +
            -0.35 * np.exp(-((t - (0.68 + r_shift)) ** 2) / (2 * 0.065 ** 2))
        )

    noise = float(rng.uniform(0.018, 0.045)) * rng.standard_normal(length)
    ecg = signal + baseline_wander + noise
    ecg = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)
    return ecg.astype(np.float32)


def generate_all_datasets(base_dir: str = "."):
    rng = np.random.default_rng(42)
    print("=" * 64)
    print("GENERATING DEEP HEALTH AI v2.0 DATASETS")
    print("=" * 64)

    # 1. Chest X-Ray Vision Dataset (Scaled from 400 -> 2,000 images)
    vision_dir = os.path.join(base_dir, "data", "vision")
    normal_dir = os.path.join(vision_dir, "normal")
    pneumonia_dir = os.path.join(vision_dir, "pneumonia")
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(pneumonia_dir, exist_ok=True)

    n_vision_per_class = 1000
    print(f"\n[1/2] Generating {n_vision_per_class * 2} High-Variance Chest X-Rays (128x128)...")
    manifest_rows = []
    for i in range(n_vision_per_class):
        norm_img = generate_chest_xray(label=0, size=128, rng=rng)
        norm_path = os.path.join(normal_dir, f"normal_{i:04d}.png")
        Image.fromarray(norm_img).save(norm_path)
        manifest_rows.append({"filename": f"normal_{i:04d}.png", "label": 0, "subtype": "Normal"})

        subtype = str(rng.choice(["lobar_bacterial", "interstitial_viral", "subtle_early"], p=[0.45, 0.35, 0.20]))
        pneu_img = generate_chest_xray(label=1, size=128, rng=rng, subtype=subtype)
        pneu_path = os.path.join(pneumonia_dir, f"pneumonia_{i:04d}.png")
        Image.fromarray(pneu_img).save(pneu_path)
        manifest_rows.append({"filename": f"pneumonia_{i:04d}.png", "label": 1, "subtype": subtype})

    pd.DataFrame(manifest_rows).to_csv(os.path.join(vision_dir, "vision_manifest.csv"), index=False)
    print(f"  -> Saved {n_vision_per_class} normal & {n_vision_per_class} pathology radiographs to {vision_dir}")

    # 2. ECG Bio-Signals Dataset (Scaled from 1,200 -> 5,500 multi-class waveforms)
    ecg_dir = os.path.join(base_dir, "data", "ecg")
    os.makedirs(ecg_dir, exist_ok=True)

    print("\n[2/2] Generating Lead-II ECG Bio-Signals (5,500 beats across 5 rhythm classes)...")
    n_normal_train = 2500
    n_normal_val = 600
    n_per_class_test = 480  # 5 * 480 = 2,400 beats across 5 classes
    rhythm_classes = ["nsr", "pvc", "afib", "st_elevation", "lbbb"]

    normal_train = np.array([generate_ecg_beat(False, 256, rng, "nsr") for _ in range(n_normal_train)])
    normal_val = np.array([generate_ecg_beat(False, 256, rng, "nsr") for _ in range(n_normal_val)])

    # Multi-class supervised split (2,000 train beats across 5 classes for supervised ResNet1D / GBDT)
    sup_X, sup_y = [], []
    for cls_idx, r_name in enumerate(rhythm_classes):
        for _ in range(400):
            sup_X.append(generate_ecg_beat(cls_idx != 0, 256, rng, r_name))
            sup_y.append(cls_idx)
    sup_X = np.array(sup_X, dtype=np.float32)
    sup_y = np.array(sup_y, dtype=np.int64)
    perm_sup = rng.permutation(len(sup_y))
    sup_X, sup_y = sup_X[perm_sup], sup_y[perm_sup]

    test_signals, test_bin_labels, test_multi_labels = [], [], []
    for cls_idx, r_name in enumerate(rhythm_classes):
        for _ in range(n_per_class_test):
            test_signals.append(generate_ecg_beat(cls_idx != 0, 256, rng, r_name))
            test_bin_labels.append(0 if cls_idx == 0 else 1)
            test_multi_labels.append(cls_idx)

    test_data = np.array(test_signals, dtype=np.float32)
    test_labels = np.array(test_bin_labels, dtype=np.int64)
    test_multi = np.array(test_multi_labels, dtype=np.int64)

    perm = rng.permutation(len(test_labels))
    test_data = test_data[perm]
    test_labels = test_labels[perm]
    test_multi = test_multi[perm]

    np.save(os.path.join(ecg_dir, "ecg_train_normal.npy"), normal_train)
    np.save(os.path.join(ecg_dir, "ecg_val_normal.npy"), normal_val)
    np.save(os.path.join(ecg_dir, "ecg_test.npy"), test_data)
    np.save(os.path.join(ecg_dir, "ecg_test_labels.npy"), test_labels)
    np.save(os.path.join(ecg_dir, "ecg_test_multiclass.npy"), test_multi)
    np.savez_compressed(os.path.join(ecg_dir, "ecg_supervised_train.npz"), X=sup_X, y=sup_y, classes=np.array(rhythm_classes))

    ext_meta_path = os.path.join(base_dir, "data", "external", "external_metadata.json")
    ext_meta = {}
    if os.path.exists(ext_meta_path):
        with open(ext_meta_path, "r", encoding="utf-8") as f:
            ext_meta = json.load(f)

    metadata = {
        "vision": {
            "total_images": n_vision_per_class * 2,
            "external_real_images": ext_meta.get("pneumoniamnist", {}).get("total_images", 0),
            "classes": ["Normal", "Pneumonia"],
            "subtypes": ["Normal", "Lobar_Bacterial", "Interstitial_Viral", "Subtle_Early"],
            "resolution": [128, 128],
            "channels": 1,
        },
        "ecg": {
            "train_normal_beats": n_normal_train,
            "val_normal_beats": n_normal_val,
            "supervised_train_beats": len(sup_y),
            "test_total_beats": len(test_labels),
            "external_real_mitbih_beats": ext_meta.get("mitbih_ecg", {}).get("total_beats", 0),
            "test_anomaly_ratio": float(np.mean(test_labels)),
            "rhythm_classes": rhythm_classes,
            "signal_length": 256,
        },
        "clinical_ehr": {
            "external_real_patients": ext_meta.get("cardiology_ehr", {}).get("total_patients", 0),
            "positive_rate": ext_meta.get("cardiology_ehr", {}).get("positive_rate", 0.0),
        },
    }
    with open(os.path.join(base_dir, "data", "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"  -> Saved {n_normal_train} train + {n_normal_val} val + {len(sup_y)} supervised + {len(test_labels)} test ECG beats")
    print("=" * 64)
    print("Dataset generation v2.0 completed successfully!")
    print("=" * 64)


if __name__ == "__main__":
    generate_all_datasets()
