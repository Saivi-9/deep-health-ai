"""
Chest X-Ray Dataset & Dataloaders with Unified Real + Domain Augmentation Cohorts
---------------------------------------------------------------------------------
Applies strict train/validation/test stratified splitting before augmentations:
- Supports synthetic high-variance chest radiographs (data/vision/)
- Seamlessly merges real hospital PneumoniaMNIST radiographs (data/external/pneumoniamnist/)
  when available, plus exposes isolated real-only evaluation loaders.
"""

import os
import glob
from typing import Tuple, List, Optional
import pandas as pd
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
        img = Image.open(path).convert("L")
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
        transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.95, 1.05)),
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
    seed: int = 42,
    include_real_external: bool = True,
    external_dir: str = "data/external/pneumoniamnist",
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Scans primary vision directory (and optional real PneumoniaMNIST folder),
    performs strict stratified splitting into train, val, and test sets,
    and returns PyTorch DataLoader objects.
    """
    normal_files = sorted(glob.glob(os.path.join(data_dir, "normal", "*.png")))
    pneumonia_files = sorted(glob.glob(os.path.join(data_dir, "pneumonia", "*.png")))

    if not normal_files or not pneumonia_files:
        raise FileNotFoundError(
            f"Images not found in {data_dir}. Please run data/generate_datasets.py first."
        )

    all_files = list(normal_files) + list(pneumonia_files)
    all_labels = [0] * len(normal_files) + [1] * len(pneumonia_files)

    if include_real_external and os.path.isdir(external_dir):
        ext_norm = sorted(glob.glob(os.path.join(external_dir, "normal", "*.png")))
        ext_pneu = sorted(glob.glob(os.path.join(external_dir, "pneumonia", "*.png")))
        if ext_norm and ext_pneu:
            all_files.extend(ext_norm + ext_pneu)
            all_labels.extend([0] * len(ext_norm) + [1] * len(ext_pneu))

    temp_ratio = val_ratio + test_ratio
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        all_files, all_labels,
        test_size=temp_ratio,
        stratify=all_labels,
        random_state=seed
    )

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


def get_real_pneumoniamnist_test_loader(
    manifest_path: str = "data/external/pneumoniamnist_manifest.csv",
    base_dir: str = ".",
    image_size: Tuple[int, int] = (128, 128),
    batch_size: int = 32,
) -> Optional[DataLoader]:
    """Returns a DataLoader specifically for the isolated real PneumoniaMNIST test split."""
    if not os.path.exists(manifest_path):
        return None
    df = pd.read_csv(manifest_path)
    df_test = df[df["source_split"] == "test_real"] if "source_split" in df.columns else df
    if df_test.empty:
        return None
    paths = [os.path.join(base_dir, p) for p in df_test["relative_path"].tolist()]
    labels = df_test["label"].astype(int).tolist()
    _, eval_tf = get_vision_transforms(image_size)
    return DataLoader(MedicalImageDataset(paths, labels, transform=eval_tf), batch_size=batch_size, shuffle=False)
