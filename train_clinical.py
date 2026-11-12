"""
Clinical Cardiology EHR Risk & Triage Model Training CLI
--------------------------------------------------------
Usage:
    python train_clinical.py
"""

import os
from src.clinical.trainer import train_clinical_triage_suite
from src.utils.logger import get_logger

logger = get_logger("TrainClinicalCLI")


def main():
    csv_path = "data/external/cardiology_triage.csv"
    if not os.path.exists(csv_path):
        logger.info("Real cardiology cohort not found. Downloading external clinical cohorts...")
        from data.fetch_external import fetch_real_cardiology_ehr
        fetch_real_cardiology_ehr(base_dir=".")

    report = train_clinical_triage_suite(
        csv_path=csv_path,
        save_model_path="saved_models/best_clinical_triage.joblib",
        metrics_path="reports/clinical_metrics.json",
    )
    logger.info(f"Clinical tournament complete. Winner: {report['best_model']} (ROC-AUC={report['best_test_metrics'].get('roc_auc')})")


if __name__ == "__main__":
    main()
