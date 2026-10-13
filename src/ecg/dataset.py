"""
ECG Bio-Signal Dataset & Dataloaders
------------------------------------
Loads normalized 1D ECG lead segments for unsupervised autoencoder modeling:
- Channel dimension expansion (1, L)
- PyTorch DataLoader wrapping
"""

import os
from typing import Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ECGDataset(Dataset):
    """PyTorch Dataset for 1D ECG signals."""

    def __init__(self, data: np.ndarray, labels: Optional[np.ndarray] = None):
        self.data = torch.from_numpy(data.astype(np.float32))
        if self.data.ndim == 2:
            # (N, L) -> (N, 1, L)
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
    """
    Loads train normal, validation normal, and evaluation test sets.
    """
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

    train_dataset = ECGDataset(train_normal)
    val_dataset = ECGDataset(val_normal)
    test_dataset = ECGDataset(test_signals, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, test_signals, test_labels
