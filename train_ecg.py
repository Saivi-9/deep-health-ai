"""
ECG 1D Autoencoder Training & Anomaly Evaluation CLI
---------------------------------------------------
Usage:
    python train_ecg.py [--epochs 20] [--lr 0.001] [--batch-size 32]
"""

import os
import argparse
import json
import torch

from src.ecg.dataset import load_ecg_dataloaders
from src.ecg.model import Conv1dAutoencoder
from src.ecg.trainer import ECGAutoencoderTrainer
from src.ecg.anomaly_detector import ECGAnomalyDetector
from src.ecg.evaluate import evaluate_ecg_anomaly_detection
from src.utils.plotting import plot_training_history
from src.utils.logger import get_logger

logger = get_logger("TrainECGCLI")


def main():
    parser = argparse.ArgumentParser(description="Train 1D ECG Autoencoder for Cardiac Anomaly Detection")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--data-dir", type=str, default="data/ecg", help="Data directory")
    parser.add_argument("--save-model", type=str, default="saved_models/best_ecg_autoencoder.pt", help="Model checkpoint path")
    parser.add_argument("--figures-dir", type=str, default="reports/figures", help="Figures output directory")
    args = parser.parse_args()

    # Verify dataset exists, if not generate
    if not os.path.exists(os.path.join(args.data_dir, "ecg_train_normal.npy")):
        logger.info("ECG Dataset not found. Generating datasets...")
        from data.generate_datasets import generate_all_datasets
        generate_all_datasets()

    logger.info("Loading ECG dataloaders...")
    train_loader, val_loader, test_loader, test_signals, test_labels = load_ecg_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using compute device: {device}")

    model = Conv1dAutoencoder(in_channels=1)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Initialized: Conv1dAutoencoder ({total_params:,} trainable parameters)")

    trainer = ECGAutoencoderTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        patience=5,
        device=device
    )

    history = trainer.fit(save_path=args.save_model)

    # Plot training loss curve
    history_plot_path = os.path.join(args.figures_dir, "ecg_training_history.png")
    plot_training_history(history, history_plot_path, title="1D ECG Autoencoder")
    logger.info(f"Training curves saved to {history_plot_path}")

    # Initialize Anomaly Detector and Calibrate Threshold
    detector = ECGAnomalyDetector(model=model, device=device)
    threshold = detector.calibrate_threshold(val_loader, percentile=97.5)

    # Evaluate on test set
    test_metrics = evaluate_ecg_anomaly_detection(
        detector=detector,
        test_loader=test_loader,
        test_labels=test_labels,
        figures_dir=args.figures_dir
    )

    # Save detector state (threshold and metadata)
    metadata = {
        "threshold": float(threshold),
        "metrics": test_metrics
    }
    metrics_save_path = "reports/ecg_metrics.json"
    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
    with open(metrics_save_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved ECG test metrics to {metrics_save_path}")


if __name__ == "__main__":
    main()
