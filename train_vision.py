"""
Chest X-Ray Multi-Architecture Vision Training & Evaluation CLI
-----------------------------------------------------------------
Trains and benchmarks 4 clinical vision architectures on the unified real + domain
chest radiograph cohort (data/vision + real PneumoniaMNIST):
1. ResidualChestCNN + SpatialAttention (`saved_models/best_chest_cnn.pt`)
2. TransferResNet18 (`saved_models/vision_resnet18.pt`)
3. TransferMobileNetV2 (`saved_models/vision_mobilenetv2.pt`)
4. Radiomic-14 LightGBM (`saved_models/vision_radiomic_lgbm.joblib`)

Also evaluates all 4 models on the isolated real PneumoniaMNIST holdout test set.

Usage:
    python train_vision.py [--epochs 8] [--lr 0.001] [--batch-size 32]
"""

import os
import argparse
import json
import torch

from src.vision.dataset import get_vision_dataloaders, get_real_pneumoniamnist_test_loader
from src.vision.model import ResidualChestCNN, TransferChestClassifier
from src.vision.trainer import VisionTrainer, train_radiomic_lgbm
from src.vision.evaluate import evaluate_vision_model
from src.utils.plotting import plot_training_history
from src.utils.logger import get_logger

logger = get_logger("TrainVisionCLI")


def train_and_benchmark_vision(
    epochs: int = 10,
    lr: float = 0.001,
    batch_size: int = 32,
    data_dir: str = "data/vision",
    save_model: str = "saved_models/best_chest_cnn.pt",
    figures_dir: str = "reports/figures",
    transfer_epochs: int = 6,
) -> dict:
    if not os.path.exists(os.path.join(data_dir, "normal")):
        logger.info("Dataset not found. Generating datasets...")
        from data.generate_datasets import generate_all_datasets
        generate_all_datasets()

    logger.info("Loading unified Chest Radiograph dataloaders (generated + real PneumoniaMNIST)...")
    train_loader, val_loader, test_loader, class_names = get_vision_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        include_real_external=True,
    )
    real_test_loader = get_real_pneumoniamnist_test_loader(batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using compute device: {device}")

    models_comparison = {}

    # 1. Primary Custom Architecture: ResidualChestCNN + SpatialAttention
    cnn_model = ResidualChestCNN(in_channels=1, num_classes=len(class_names))
    total_params = sum(p.numel() for p in cnn_model.parameters() if p.requires_grad)
    logger.info(f"1/4 Training ResidualChestCNN ({total_params:,} params)...")
    cnn_trainer = VisionTrainer(
        model=cnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=lr,
        num_epochs=epochs,
        patience=5,
        device=device,
    )
    history = cnn_trainer.fit(save_path=save_model)
    plot_training_history(history, os.path.join(figures_dir, "vision_training_history.png"), title="Chest X-Ray Residual CNN")

    primary_metrics = evaluate_vision_model(
        model=cnn_model,
        test_loader=test_loader,
        class_names=class_names,
        figures_dir=figures_dir,
        device=device,
        save_figures=True,
        model_label="ResidualChestCNN",
    )
    real_cnn_metrics = (
        evaluate_vision_model(
            model=cnn_model,
            test_loader=real_test_loader,
            class_names=class_names,
            figures_dir=figures_dir,
            device=device,
            save_figures=False,
            model_label="ResidualChestCNN [Real PneumoniaMNIST]",
        )
        if real_test_loader is not None
        else None
    )
    models_comparison["ResidualChestCNN"] = {
        "architecture": "4-Stage Residual CNN + SpatialAttention + MC Dropout",
        "trainable_parameters": total_params,
        "checkpoint": save_model,
        "unified_test_metrics": primary_metrics,
        "real_pneumoniamnist_metrics": real_cnn_metrics,
    }

    # 2. Transfer Learning: ResNet18
    resnet_model = TransferChestClassifier(arch="resnet18", num_classes=len(class_names), pretrained=True)
    resnet_params = sum(p.numel() for p in resnet_model.parameters() if p.requires_grad)
    logger.info(f"2/4 Fine-tuning TransferResNet18 ({resnet_params:,} params)...")
    resnet_trainer = VisionTrainer(
        model=resnet_model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=3e-4,
        num_epochs=transfer_epochs,
        patience=3,
        device=device,
    )
    resnet_trainer.fit(save_path="saved_models/vision_resnet18.pt")
    resnet_metrics = evaluate_vision_model(
        model=resnet_model,
        test_loader=test_loader,
        class_names=class_names,
        figures_dir=figures_dir,
        device=device,
        save_figures=False,
        model_label="TransferResNet18",
    )
    real_resnet_metrics = (
        evaluate_vision_model(
            model=resnet_model,
            test_loader=real_test_loader,
            class_names=class_names,
            figures_dir=figures_dir,
            device=device,
            save_figures=False,
            model_label="TransferResNet18 [Real PneumoniaMNIST]",
        )
        if real_test_loader is not None
        else None
    )
    models_comparison["TransferResNet18"] = {
        "architecture": "ImageNet-Pretrained ResNet18 Backbone + MC-Dropout Head",
        "trainable_parameters": resnet_params,
        "checkpoint": "saved_models/vision_resnet18.pt",
        "unified_test_metrics": resnet_metrics,
        "real_pneumoniamnist_metrics": real_resnet_metrics,
    }

    # 3. Transfer Learning: MobileNetV2
    mobilenet_model = TransferChestClassifier(arch="mobilenet_v2", num_classes=len(class_names), pretrained=True)
    mobilenet_params = sum(p.numel() for p in mobilenet_model.parameters() if p.requires_grad)
    logger.info(f"3/4 Fine-tuning TransferMobileNetV2 ({mobilenet_params:,} params)...")
    mobilenet_trainer = VisionTrainer(
        model=mobilenet_model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=4e-4,
        num_epochs=transfer_epochs,
        patience=3,
        device=device,
    )
    mobilenet_trainer.fit(save_path="saved_models/vision_mobilenetv2.pt")
    mobilenet_metrics = evaluate_vision_model(
        model=mobilenet_model,
        test_loader=test_loader,
        class_names=class_names,
        figures_dir=figures_dir,
        device=device,
        save_figures=False,
        model_label="TransferMobileNetV2",
    )
    real_mobilenet_metrics = (
        evaluate_vision_model(
            model=mobilenet_model,
            test_loader=real_test_loader,
            class_names=class_names,
            figures_dir=figures_dir,
            device=device,
            save_figures=False,
            model_label="TransferMobileNetV2 [Real PneumoniaMNIST]",
        )
        if real_test_loader is not None
        else None
    )
    models_comparison["TransferMobileNetV2"] = {
        "architecture": "ImageNet-Pretrained MobileNetV2 Backbone + MC-Dropout Head",
        "trainable_parameters": mobilenet_params,
        "checkpoint": "saved_models/vision_mobilenetv2.pt",
        "unified_test_metrics": mobilenet_metrics,
        "real_pneumoniamnist_metrics": real_mobilenet_metrics,
    }

    # 4. Classical Tabular Radiomics-14 + LightGBM
    real_files = real_test_loader.dataset.file_paths if real_test_loader is not None else None
    real_labels = real_test_loader.dataset.labels if real_test_loader is not None else None
    _, lgbm_metrics, real_lgbm_metrics = train_radiomic_lgbm(
        train_files=train_loader.dataset.file_paths,
        train_labels=train_loader.dataset.labels,
        test_files=test_loader.dataset.file_paths,
        test_labels=test_loader.dataset.labels,
        real_test_files=real_files,
        real_test_labels=real_labels,
        save_path="saved_models/vision_radiomic_lgbm.joblib",
    )
    models_comparison["RadiomicLightGBM"] = {
        "architecture": "14 First-Order/Texture/Spatial Radiomic Features + LightGBM",
        "trainable_parameters": 14,
        "checkpoint": "saved_models/vision_radiomic_lgbm.joblib",
        "unified_test_metrics": lgbm_metrics,
        "real_pneumoniamnist_metrics": real_lgbm_metrics,
    }

    output_payload = {
        **primary_metrics,
        "dataset_summary": {
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "test_samples": len(test_loader.dataset),
            "total_samples": len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset),
            "real_pneumoniamnist_test_samples": len(real_test_loader.dataset) if real_test_loader else 0,
        },
        "models_comparison": models_comparison,
    }

    metrics_save_path = "reports/vision_metrics.json"
    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
    with open(metrics_save_path, "w") as f:
        json.dump(output_payload, f, indent=2)
    logger.info(f"Saved multi-model vision metrics to {metrics_save_path}")
    return output_payload


def main():
    parser = argparse.ArgumentParser(description="Train Multi-Architecture Chest X-Ray Classifiers")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs for ResidualChestCNN")
    parser.add_argument("--transfer-epochs", type=int, default=6, help="Fine-tuning epochs for ResNet18 & MobileNetV2")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for ResidualChestCNN")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--data-dir", type=str, default="data/vision", help="Data directory")
    parser.add_argument("--save-model", type=str, default="saved_models/best_chest_cnn.pt", help="Primary CNN checkpoint")
    parser.add_argument("--figures-dir", type=str, default="reports/figures", help="Figures output directory")
    args = parser.parse_args()

    train_and_benchmark_vision(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        save_model=args.save_model,
        figures_dir=args.figures_dir,
        transfer_epochs=args.transfer_epochs,
    )


if __name__ == "__main__":
    main()
