"""
ECG Bio-Signal Dataset & Dataloaders
------------------------------------
Supports:
1. Unsupervised Normal Sinus Rhythm DataLoaders for Conv1dAutoencoder
2. 5-Class Supervised Lead-II ECG DataLoaders for ECGResNet1D / GBDT
3. Real PhysioNet MIT-BIH 10,000-beat 5-class DataLoaders (data/external/mitbih_10k.npz)
"""

import os
from typing import Tuple, Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class ECGDataset(Dataset):
    """PyTorch Dataset for 1D ECG signals."""

    def __init__(self, data: np.ndarray, labels: Optional[np.ndarray] = None):
        self.data = torch.from_numpy(data.astype(np.float32))
        if self.data.ndim == 2:
            self.data = self.data.unsqueeze(1)

        if labels is not None:
            self.labels = torch.from_numpy(labels.astype(np.int64))
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        signal = self.data[idx]
        label = self.labels[idx].item() if self.labels is not None else 0
        return signal, label


def load_ecg_dataloaders(
    data_dir: str = "data/ecg",
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """Loads train normal, validation normal, and binary anomaly test sets."""
    train_path = os.path.join(data_dir, "ecg_train_normal.npy")
    val_path = os.path.join(data_dir, "ecg_val_normal.npy")
    test_path = os.path.join(data_dir, "ecg_test.npy")
    test_labels_path = os.path.join(data_dir, "ecg_test_labels.npy")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"ECG files not found in {data_dir}. Please run data/generate_datasets.py first."
        )

    train_normal = np.load(train_path)
    val_normal = np.load(val_path)
    test_signals = np.load(test_path)
    test_labels = np.load(test_labels_path)

    train_loader = DataLoader(ECGDataset(train_normal), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ECGDataset(val_normal), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ECGDataset(test_signals, test_labels), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, test_signals, test_labels


def load_supervised_ecg_dataloaders(
    data_dir: str = "data/ecg",
    batch_size: int = 32,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads 5-class supervised Lead-II datasets (splitting `ecg_supervised_train.npz` cleanly
    into train/val and evaluating against isolated `ecg_test.npy` + `ecg_test_multiclass.npy`).
    """
    sup_path = os.path.join(data_dir, "ecg_supervised_train.npz")
    test_path = os.path.join(data_dir, "ecg_test.npy")
    test_multi_path = os.path.join(data_dir, "ecg_test_multiclass.npy")
    test_bin_path = os.path.join(data_dir, "ecg_test_labels.npy")

    loaded = np.load(sup_path, allow_pickle=True)
    X_sup, y_sup = loaded["X"], loaded["y"]
    classes = [str(c) for c in loaded["classes"].tolist()]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_sup, y_sup, test_size=0.15, stratify=y_sup, random_state=seed
    )
    X_test = np.load(test_path)
    y_test_multi = np.load(test_multi_path)
    y_test_bin = np.load(test_bin_path)

    train_loader = DataLoader(ECGDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ECGDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ECGDataset(X_test, y_test_multi), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, classes, X_sup, y_sup, X_test, y_test_multi, y_test_bin


def load_mitbih_dataloaders(
    npz_path: str = "data/external/mitbih_10k.npz",
    batch_size: int = 64,
    seed: int = 42,
) -> Optional[Tuple[DataLoader, DataLoader, DataLoader, List[str]]]:
    """Loads stratified train/val/test DataLoaders for the 10k real PhysioNet MIT-BIH cohort."""
    if not os.path.exists(npz_path):
        return None
    loaded = np.load(npz_path, allow_pickle=True)
    X, y = loaded["X"], loaded["y"]
    classes = [str(c) for c in loaded["classes"].tolist()]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed)

    return (
        DataLoader(ECGDataset(X_train, y_train), batch_size=batch_size, shuffle=True),
        DataLoader(ECGDataset(X_val, y_val), batch_size=batch_size, shuffle=False),
        DataLoader(ECGDataset(X_test, y_test), batch_size=batch_size, shuffle=False),
        classes,
    )
