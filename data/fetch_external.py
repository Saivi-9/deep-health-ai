"""
External Real-World Clinical Dataset Fetcher (Hugging Face Hub)
---------------------------------------------------------------
Downloads and caches compact, peer-reviewed clinical benchmark datasets:
1. Real Chest Radiographs: `danjacobellis/pneumoniamnist_224` (val + test = 1,148 224x224 scans)
2. Real MIT-BIH 5-Class ECG Arrhythmias: `dheerajthuvara/ecg-arrhythmia-data` (10,000 stratified beats across A, L, N, R, V)
3. Real Clinical Cardiology / Heart Triage Cohort:
   Merged & deduplicated from `BenchmarkDatasets/Heart-Disease-Dataset-_Comprehensive` & `TabBioMed/HeartDisease_UCI`
"""

import os
import json
from io import BytesIO
from typing import Dict, Any

import numpy as np
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download

from src.utils.logger import get_logger

logger = get_logger("FetchExternal")

MITBIH_CLASS_MAP = {
    0: "A",  # Atrial Premature Contraction (APC)
    1: "L",  # Left Bundle Branch Block (LBBB)
    2: "N",  # Normal Sinus Rhythm (NSR)
    3: "R",  # Right Bundle Branch Block (RBBB)
    4: "V",  # Premature Ventricular Contraction (PVC)
}

MITBIH_FULL_NAMES = {
    "N": "Normal Sinus Rhythm (MIT-BIH Class N)",
    "L": "Left Bundle Branch Block (MIT-BIH Class L)",
    "R": "Right Bundle Branch Block (MIT-BIH Class R)",
    "V": "Premature Ventricular Contraction / PVC (MIT-BIH Class V)",
    "A": "Atrial Premature Contraction / APC (MIT-BIH Class A)",
}


def fetch_real_pneumoniamnist(base_dir: str = ".", force: bool = False) -> Dict[str, Any]:
    """Extracts 1,148 real 224x224 chest radiographs from PneumoniaMNIST validation & test splits."""
    out_dir = os.path.join(base_dir, "data", "external", "pneumoniamnist")
    manifest_path = os.path.join(base_dir, "data", "external", "pneumoniamnist_manifest.csv")

    if os.path.exists(manifest_path) and not force:
        df_m = pd.read_csv(manifest_path)
        if len(df_m) >= 1000:
            logger.info(f"Using cached real PneumoniaMNIST cohort ({len(df_m)} radiographs).")
            return {
                "dataset": "danjacobellis/pneumoniamnist_224",
                "total_images": int(len(df_m)),
                "normal_count": int((df_m["label"] == 0).sum()),
                "pneumonia_count": int((df_m["label"] == 1).sum()),
                "manifest_path": manifest_path,
            }

    os.makedirs(os.path.join(out_dir, "normal"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "pneumonia"), exist_ok=True)

    rows = []
    idx_counter = 0
    for split_name, parquet_file in [
        ("train_real", "data/validation-00000-of-00001.parquet"),
        ("test_real", "data/test-00000-of-00001.parquet"),
    ]:
        p = hf_hub_download("danjacobellis/pneumoniamnist_224", parquet_file, repo_type="dataset")
        df = pd.read_parquet(p)
        for _, row in df.iterrows():
            raw_lbl = row["label"]
            lbl = int(raw_lbl[0]) if hasattr(raw_lbl, "__len__") else int(raw_lbl)
            cls_name = "pneumonia" if lbl == 1 else "normal"
            img_field = row["image"]
            if isinstance(img_field, dict) and "bytes" in img_field:
                pil = Image.open(BytesIO(img_field["bytes"])).convert("L")
            else:
                pil = img_field.convert("L")

            fname = f"real_cxr_{cls_name}_{idx_counter:04d}.png"
            fpath = os.path.join(out_dir, cls_name, fname)
            pil.save(fpath)
            rows.append({
                "sample_id": f"PMNIST-{idx_counter:04d}",
                "relative_path": os.path.relpath(fpath, base_dir).replace("\\", "/"),
                "label": lbl,
                "class_name": "Pneumonia" if lbl == 1 else "Normal",
                "source_split": split_name,
            })
            idx_counter += 1

    df_m = pd.DataFrame(rows)
    df_m.to_csv(manifest_path, index=False)
    logger.info(f"Saved {len(df_m)} real PneumoniaMNIST chest X-rays to {out_dir}")
    return {
        "dataset": "danjacobellis/pneumoniamnist_224",
        "total_images": int(len(df_m)),
        "normal_count": int((df_m["label"] == 0).sum()),
        "pneumonia_count": int((df_m["label"] == 1).sum()),
        "manifest_path": manifest_path,
    }


def fetch_real_mitbih_ecg(base_dir: str = ".", per_class: int = 2000, force: bool = False) -> Dict[str, Any]:
    """Downloads MIT-BIH 5-class dataset and saves a stratified 10,000-beat benchmark cohort."""
    out_npz = os.path.join(base_dir, "data", "external", "mitbih_10k.npz")
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)

    if os.path.exists(out_npz) and not force:
        loaded = np.load(out_npz, allow_pickle=True)
        X, y = loaded["X"], loaded["y"]
        logger.info(f"Using cached MIT-BIH cohort ({len(X)} real beats across 5 classes).")
        return {
            "dataset": "dheerajthuvara/ecg-arrhythmia-data",
            "total_beats": int(len(X)),
            "beat_length": int(X.shape[1]),
            "classes": ["A", "L", "N", "R", "V"],
            "npz_path": out_npz,
        }

    px = hf_hub_download("dheerajthuvara/ecg-arrhythmia-data", "data/processed/X.npy", repo_type="dataset")
    py = hf_hub_download("dheerajthuvara/ecg-arrhythmia-data", "data/processed/y.npy", repo_type="dataset")

    X_all = np.load(px, mmap_mode="r")
    y_all = np.load(py)

    rng = np.random.default_rng(42)
    selected_idx = []
    for cls_id in range(5):
        cls_indices = np.where(y_all == cls_id)[0]
        take = min(per_class, len(cls_indices))
        chosen = rng.choice(cls_indices, size=take, replace=False)
        selected_idx.append(chosen)

    idx = np.concatenate(selected_idx)
    rng.shuffle(idx)
    X_sub = np.array(X_all[idx], dtype=np.float32)
    y_sub = np.array(y_all[idx], dtype=np.int64)
    classes = np.array(["A", "L", "N", "R", "V"])

    np.savez_compressed(out_npz, X=X_sub, y=y_sub, classes=classes)
    logger.info(f"Saved {len(X_sub)} real stratified MIT-BIH ECG beats to {out_npz}")
    return {
        "dataset": "dheerajthuvara/ecg-arrhythmia-data",
        "total_beats": int(len(X_sub)),
        "beat_length": int(X_sub.shape[1]),
        "classes": classes.tolist(),
        "npz_path": out_npz,
    }


def fetch_real_cardiology_ehr(base_dir: str = ".", force: bool = False) -> Dict[str, Any]:
    """
    Fetches and merges real clinical cardiology cohorts from Hugging Face Hub:
    - `BenchmarkDatasets/Heart-Disease-Dataset-_Comprehensive` (1,190 multi-center patients)
    - `TabBioMed/HeartDisease_UCI` (920 UCI multi-hospital patients)
    Preserves missing values for leak-free pipeline imputation.
    """
    out_csv = os.path.join(base_dir, "data", "external", "cardiology_triage.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    if os.path.exists(out_csv) and not force:
        df = pd.read_csv(out_csv)
        logger.info(f"Using cached real Cardiology EHR cohort ({len(df)} unique patient encounters).")
        return {
            "dataset": "BenchmarkDatasets/Heart-Disease-Dataset-_Comprehensive + TabBioMed/HeartDisease_UCI",
            "total_patients": int(len(df)),
            "positive_rate": round(float(df["target"].mean()), 4),
            "csv_path": out_csv,
        }

    # 1. Comprehensive dataset (11 core features, clean numeric)
    comp_dfs = []
    for sp in ["data/train.parquet", "data/val.parquet", "data/test.parquet"]:
        p = hf_hub_download("BenchmarkDatasets/Heart-Disease-Dataset-_Comprehensive", sp, repo_type="dataset")
        comp_dfs.append(pd.read_parquet(p))
    df_comp = pd.concat(comp_dfs, ignore_index=True)
    df_comp = df_comp.rename(columns={
        "f_0": "age",
        "f_1": "sex",
        "f_2": "chest_pain_type",
        "f_3": "resting_bp",
        "f_4": "cholesterol",
        "f_5": "fasting_bs",
        "f_6": "resting_ecg",
        "f_7": "max_hr",
        "f_8": "exercise_angina",
        "f_9": "oldpeak",
        "f_10": "st_slope",
    })
    core_cols = [
        "age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
        "fasting_bs", "resting_ecg", "max_hr", "exercise_angina",
        "oldpeak", "st_slope", "target",
    ]
    df_comp = df_comp[core_cols].copy()
    df_comp["cohort_source"] = "Comprehensive_MultiCenter"

    # 2. TabBioMed UCI dataset
    uci_dfs = []
    for sp in ["data/train.parquet", "data/val.parquet", "data/test.parquet"]:
        p = hf_hub_download("TabBioMed/HeartDisease_UCI", sp, repo_type="dataset")
        uci_dfs.append(pd.read_parquet(p))
    df_uci = pd.concat(uci_dfs, ignore_index=True)
    df_uci = df_uci.rename(columns={
        "f_0_age": "age",
        "f_1_sex": "sex",
        "f_2_cp": "chest_pain_type",
        "f_3_trestbps": "resting_bp",
        "f_4_chol": "cholesterol",
        "f_5_fbs": "fasting_bs",
        "f_6_restecg": "resting_ecg",
        "f_7_thalach": "max_hr",
        "f_8_exang": "exercise_angina",
        "f_9_oldpeak": "oldpeak",
        "f_10_slope": "st_slope",
    })
    df_uci = df_uci[core_cols].copy()
    for col in core_cols:
        df_uci[col] = pd.to_numeric(df_uci[col], errors="coerce")
    df_uci["cohort_source"] = "UCI_Cleveland_Hungary_VA"

    merged = pd.concat([df_comp, df_uci], ignore_index=True)
    # Clean physically impossible zero cholesterol/BP entries recorded as missing codes
    merged.loc[merged["resting_bp"] <= 40, "resting_bp"] = np.nan
    merged.loc[merged["cholesterol"] <= 40, "cholesterol"] = np.nan
    merged["target"] = merged["target"].astype(int).clip(0, 1)

    # Deduplicate exact duplicate encounters across the two UCI derivatives
    before = len(merged)
    merged = merged.drop_duplicates(subset=core_cols).reset_index(drop=True)
    logger.info(f"Merged cardiology EHR cohorts: {before} -> {len(merged)} unique records after deduplication.")

    merged.to_csv(out_csv, index=False)
    return {
        "dataset": "BenchmarkDatasets/Heart-Disease-Dataset-_Comprehensive + TabBioMed/HeartDisease_UCI",
        "total_patients": int(len(merged)),
        "positive_rate": round(float(merged["target"].mean()), 4),
        "csv_path": out_csv,
    }


def fetch_all_external_datasets(base_dir: str = ".", force: bool = False) -> Dict[str, Any]:
    """Downloads/verifies all real clinical Hugging Face datasets."""
    summary = {
        "pneumoniamnist": fetch_real_pneumoniamnist(base_dir=base_dir, force=force),
        "mitbih_ecg": fetch_real_mitbih_ecg(base_dir=base_dir, force=force),
        "cardiology_ehr": fetch_real_cardiology_ehr(base_dir=base_dir, force=force),
    }
    ext_meta_path = os.path.join(base_dir, "data", "external", "external_metadata.json")
    os.makedirs(os.path.dirname(ext_meta_path), exist_ok=True)
    with open(ext_meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    res = fetch_all_external_datasets(base_dir=".")
    print(json.dumps(res, indent=2))
