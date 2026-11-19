"""
1D Deep Learning Architectures for Cardiac Bio-Signal Analysis
-------------------------------------------------------------
1. Conv1dAutoencoder: Unsupervised 1D CNN Autoencoder trained strictly on healthy
   Normal Sinus Rhythm (NSR) beats for out-of-distribution residual anomaly detection.
2. ECGResNet1D: Supervised multi-stage 1D Residual Neural Network with skip connections
   and AdaptiveAvgPool1d supporting arbitrary beat lengths (e.g., 256 Lead-II or 180 MIT-BIH)
   for 5-class rhythm differential classification.
"""

import torch
import torch.nn as nn


class Conv1dAutoencoder(nn.Module):
    """
    1D Convolutional Autoencoder with True Compressed Latent Bottleneck (16-D)
    for unsupervised ECG anomaly detection (16:1 compression over 256 samples).
    """

    def __init__(self, in_channels: int = 1, latent_dim: int = 12):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool1d(2, stride=2),  # 128

            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool1d(2, stride=2),  # 64

            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool1d(2, stride=2),  # 32

            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool1d(16),   # 16 -> 256 flat features
        )
        self.encoder_fc = nn.Linear(16 * 16, latent_dim)

        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 16 * 16),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(16, 32, kernel_size=4, stride=2, padding=1, bias=False),  # 32
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1, inplace=True),

            nn.ConvTranspose1d(32, 32, kernel_size=4, stride=2, padding=1, bias=False),  # 64
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1, inplace=True),

            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),  # 128
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1, inplace=True),

            nn.ConvTranspose1d(16, 16, kernel_size=4, stride=2, padding=1, bias=False),  # 256
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv1d(16, in_channels, kernel_size=7, stride=1, padding=3),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_conv(x)
        return self.encoder_fc(h.flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z).view(z.size(0), 16, 16)
        return self.decoder_conv(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


class ResBlock1D(nn.Module):
    """1D Residual Convolutional Block with skip projection."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.15)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class ECGResNet1D(nn.Module):
    """
    Supervised 1D Residual Network for multi-class cardiac rhythm differential diagnosis.
    Works across Lead-II (256 samples) and PhysioNet MIT-BIH (180 samples) via AdaptiveAvgPool1d.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = ResBlock1D(32, 32, stride=1, kernel_size=5)
        self.stage2 = ResBlock1D(32, 64, stride=2, kernel_size=5)
        self.stage3 = ResBlock1D(64, 128, stride=2, kernel_size=3)
        self.stage4 = ResBlock1D(128, 128, stride=2, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.pool(out).flatten(1)
        return self.head(out)
