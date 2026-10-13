"""
Chest X-Ray Vision Classifier Training & Evaluation CLI
-------------------------------------------------------
Usage:
    python train_vision.py [--epochs 15] [--lr 0.001] [--batch-size 32]
"""

import os
import argparse
import json
import torch

from src.vision.dataset import get_vision_dataloaders
from src.vision.model import ResidualChestCNN
from src.vision.trainer import VisionTrainer
from src.vision.evaluate import evaluate_vision_model
from src.utils.plotting import plot_training_history
from src.utils.logger import get_logger

logger = get_logger("TrainVisionCLI")


def main():
    parser = argparse.ArgumentParser(description="Train Chest X-Ray Deep Residual CNN")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--data-dir", type=str, default="data/vision", help="Data directory")
    parser.add_argument("--save-model", type=str, default="saved_models/best_chest_cnn.pt", help="Checkpoint path")
    parser.add_argument("--figures-dir", type=str, default="reports/figures", help="Figures output directory")
    args = parser.parse_args()

    # Verify dataset exists, if not generate
    if not os.path.exists(os.path.join(args.data_dir, "normal")):
        logger.info("Dataset not found. Generating datasets...")
        from data.generate_datasets import generate_all_datasets
        generate_all_datasets()

    logger.info("Loading Chest Radiograph dataloaders...")
    train_loader, val_loader, test_loader, class_names = get_vision_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using compute device: {device}")

    model = ResidualChestCNN(in_channels=1, num_classes=len(class_names))
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Initialized: ResidualChestCNN ({total_params:,} trainable parameters)")

    trainer = VisionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        patience=5,
        device=device
    )

    history = trainer.fit(save_path=args.save_model)

    # Plot training loss & accuracy
    history_plot_path = os.path.join(args.figures_dir, "vision_training_history.png")
    plot_training_history(history, history_plot_path, title="Chest X-Ray Residual CNN")
    logger.info(f"Training curves saved to {history_plot_path}")

    # Evaluate on test set
    test_metrics = evaluate_vision_model(
        model=model,
        test_loader=test_loader,
        class_names=class_names,
        figures_dir=args.figures_dir,
        device=device
    )

    metrics_save_path = "reports/vision_metrics.json"
    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
    with open(metrics_save_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Saved vision test metrics to {metrics_save_path}")


if __name__ == "__main__":
    main()
