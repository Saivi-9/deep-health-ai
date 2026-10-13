"""
Chest X-Ray Dataset & Dataloaders with Preprocessing Pipelines
--------------------------------------------------------------
Applies strict train/validation/test splitting and clinical image augmentations:
- Normalization (Zero-mean, unit variance)
- Safe biomedical augmentations (subtle horizontal flip, affine rotation, affine scaling)
- Stratified sampling across pathology classes
"""

import os
import glob
from typing import Tuple, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split


class MedicalImageDataset(Dataset):
    """PyTorch Dataset for Medical Chest Radiographs."""

    def __init__(self, file_paths: List[str], labels: List[int], transform: Optional[transforms.Compose] = None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path = self.file_paths[idx]
        img = Image.open(path).convert("L")  # 1-channel grayscale
        label = self.labels[idx]

        if self.transform:
            tensor_img = self.transform(img)
        else:
            tensor_img = transforms.ToTensor()(img)

        return tensor_img, label, path


def get_vision_transforms(image_size: Tuple[int, int] = (128, 128)) -> Tuple[transforms.Compose, transforms.Compose]:
    """Returns training augmentations and deterministic validation/testing transforms."""
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    return train_transform, eval_transform


def get_vision_dataloaders(
    data_dir: str = "data/vision",
    batch_size: int = 32,
    image_size: Tuple[int, int] = (128, 128),
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Scans directory, performs stratified splitting into train, val, and test sets,
    and returns PyTorch DataLoader objects.
    """
    normal_files = sorted(glob.glob(os.path.join(data_dir, "normal", "*.png")))
    pneumonia_files = sorted(glob.glob(os.path.join(data_dir, "pneumonia", "*.png")))

    if not normal_files or not pneumonia_files:
        raise FileNotFoundError(
            f"Images not found in {data_dir}. Please run data/generate_datasets.py first."
        )

    all_files = normal_files + pneumonia_files
    all_labels = [0] * len(normal_files) + [1] * len(pneumonia_files)

    # 1. First split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        all_files, all_labels,
        test_size=temp_ratio,
        stratify=all_labels,
        random_state=seed
    )

    # 2. Second split: Val vs Test
    relative_test_ratio = test_ratio / temp_ratio
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels,
        test_size=relative_test_ratio,
        stratify=temp_labels,
        random_state=seed
    )

    train_tf, eval_tf = get_vision_transforms(image_size)

    train_dataset = MedicalImageDataset(train_files, train_labels, transform=train_tf)
    val_dataset = MedicalImageDataset(val_files, val_labels, transform=eval_tf)
    test_dataset = MedicalImageDataset(test_files, test_labels, transform=eval_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_names = ["Normal", "Pneumonia"]
    return train_loader, val_loader, test_loader, class_names
