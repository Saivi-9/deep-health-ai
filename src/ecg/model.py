"""
1D Convolutional Deep Autoencoder for Temporal Bio-Signal Reconstruction
------------------------------------------------------------------------
Architecture:
- Hierarchical 1D Convolutions with Batch Normalization
- Latent manifold bottleneck compression
- Transposed 1D Convolutions for morphology reconstruction
- Unsupervised learning of Normal Sinus Rhythm dynamics
"""

import torch
import torch.nn as nn


class Conv1dAutoencoder(nn.Module):
    """
    1D Convolutional Autoencoder for ECG signals.
    Compresses 256-sample time series into latent space and reconstructs normal dynamics.
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()
        # Encoder: 256 -> 128 -> 64 -> 32
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool1d(2, stride=2),  # 128

            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool1d(2, stride=2),  # 64

            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool1d(2, stride=2),  # 32

            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Decoder: 32 -> 64 -> 128 -> 256
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 64
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),

            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),  # 128
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1, inplace=True),

            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),  # 256
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv1d(16, in_channels, kernel_size=7, stride=1, padding=3)  # 256
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        out = self.decode(z)
        return out
