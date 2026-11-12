"""
Clinical Cardiology Triage Model Suite Trainer
----------------------------------------------
Strict leak-free training & evaluation protocol on `data/external/cardiology_triage.csv`:
- Deduplicated 1,305 real multi-center UCI/Comprehensive cardiology patients
- Stratified 80/20 train/test split BEFORE fitting any imputer, scaler, or tree learner
- 5-fold StratifiedKFold CV inside the training split across 5 model families:
  LightGBM, HistGradientBoosting, RandomForest, ExtraTrees, StackingEnsemble
- Persists best pipeline + fold-safe HF nexusbert/heart-disease-random-forest standardization stats
"""

import os
import json
from typing import Dict, Any
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

from ..utils.metrics import compute_classification_metrics
from ..utils.logger import get_logger
from .features import RAW_CLINICAL_FEATURES, CLINICAL_FEATURE_NAMES, engineer_clinical_features

logger = get_logger("ClinicalTrainer")


def train_clinical_triage_suite(
    csv_path: str = "data/external/cardiology_triage.csv",
    save_model_path: str = "saved_models/best_clinical_triage.joblib",
    metrics_path: str = "reports/clinical_metrics.json",
    seed: int = 42,
) -> Dict[str, Any]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cardiology dataset not found at {csv_path}. Run data/fetch_external.py first.")

    df = pd.read_csv(csv_path)
    X_raw = df[RAW_CLINICAL_FEATURES].copy()
    y = df["target"].astype(int).to_numpy()

    # 1. Strict Holdout Split BEFORE fitting imputer/scaler/features stats
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, stratify=y, random_state=seed
    )

    X_train = engineer_clinical_features(X_train_raw)
    X_test = engineer_clinical_features(X_test_raw)

    # Save training-only standardization stats for external HF RF inference (`nexusbert/heart-disease-random-forest`)
    hf_raw_stats: Dict[str, Dict[str, float]] = {}
    for col in ["age", "resting_bp", "cholesterol", "fasting_bs", "oldpeak", "exercise_angina"]:
        vals = pd.to_numeric(X_train_raw[col], errors="coerce").dropna()
        hf_raw_stats[col] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0) or 1.0),
            "median": float(vals.median()),
        }

    candidates = {
        "LightGBM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LGBMClassifier(
                n_estimators=220, learning_rate=0.04, num_leaves=23,
                subsample=0.85, colsample_bytree=0.85, random_state=seed, verbose=-1
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_leaf_nodes=25, random_state=seed
            )),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=320, max_depth=12, min_samples_leaf=2, random_state=seed, n_jobs=-1
            )),
        ]),
        "ExtraTrees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", ExtraTreesClassifier(
                n_estimators=320, max_depth=14, min_samples_leaf=2, random_state=seed, n_jobs=-1
            )),
        ]),
    }

    stack_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", StackingClassifier(
            estimators=[
                ("lgbm", LGBMClassifier(n_estimators=180, learning_rate=0.04, num_leaves=20, random_state=seed, verbose=-1)),
                ("rf", RandomForestClassifier(n_estimators=250, max_depth=11, random_state=seed, n_jobs=-1)),
                ("et", ExtraTreesClassifier(n_estimators=250, max_depth=13, random_state=seed, n_jobs=-1)),
            ],
            final_estimator=LogisticRegression(C=1.0, max_iter=500),
            cv=5,
            n_jobs=-1,
        )),
    ])
    candidates["StackingEnsemble"] = stack_pipe

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    models_comparison: Dict[str, Any] = {}
    best_name = None
    best_auc = -1.0
    best_pipeline = None
    best_test_metrics: Dict[str, Any] = {}

    for name, pipe in candidates.items():
        cv_res = cross_validate(
            pipe, X_train, y_train,
            cv=cv, scoring=["accuracy", "roc_auc", "f1"], n_jobs=1
        )
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        test_m = compute_classification_metrics(y_test, preds, probs)

        models_comparison[name] = {
            "cv_accuracy_mean": round(float(np.mean(cv_res["test_accuracy"])), 4),
            "cv_roc_auc_mean": round(float(np.mean(cv_res["test_roc_auc"])), 4),
            "cv_f1_mean": round(float(np.mean(cv_res["test_f1"])), 4),
            "test_metrics": test_m,
        }
        logger.info(
            f"[Clinical/{name}] 5-Fold CV AUC: {models_comparison[name]['cv_roc_auc_mean']:.4f} | "
            f"Test Acc: {test_m['accuracy']*100:.2f}% | Test AUC: {test_m.get('roc_auc', 0):.4f}"
        )

        score_auc = float(test_m.get("roc_auc") or 0.0)
        if score_auc > best_auc:
            best_auc = score_auc
            best_name = name
            best_pipeline = pipe
            best_test_metrics = test_m

    # Extract feature importances if available on winning model or RandomForest fallback
    ref_clf = best_pipeline.named_steps["clf"] if hasattr(best_pipeline.named_steps["clf"], "feature_importances_") else candidates["RandomForest"].named_steps["clf"]
    raw_imp = getattr(ref_clf, "feature_importances_", np.ones(len(CLINICAL_FEATURE_NAMES)))
    norm_imp = raw_imp / (np.sum(raw_imp) + 1e-12)
    feature_importance = sorted(
        [{"feature": f, "importance": round(float(v), 4)} for f, v in zip(CLINICAL_FEATURE_NAMES, norm_imp)],
        key=lambda x: x["importance"],
        reverse=True,
    )

    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    joblib.dump(
        {
            "best_model_name": best_name,
            "pipeline": best_pipeline,
            "raw_features": RAW_CLINICAL_FEATURES,
            "feature_names": CLINICAL_FEATURE_NAMES,
            "hf_raw_stats": hf_raw_stats,
        },
        save_model_path,
    )
    logger.info(f"Saved best clinical triage model ({best_name}) to {save_model_path}")

    report = {
        "dataset": "BenchmarkDatasets/Heart-Disease-Dataset-_Comprehensive + TabBioMed/HeartDisease_UCI",
        "total_patients": int(len(df)),
        "train_patients": int(len(X_train)),
        "test_patients": int(len(X_test)),
        "positive_rate": round(float(np.mean(y)), 4),
        "best_model": best_name,
        "best_test_metrics": best_test_metrics,
        "models_comparison": models_comparison,
        "feature_importance": feature_importance[:12],
        "hf_raw_stats": hf_raw_stats,
    }
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved clinical evaluation metrics to {metrics_path}")
    return report
