"""
Vision Architectures for Chest Radiograph Diagnostics
------------------------------------------------------
1. ResidualChestCNN: Custom 4-stage Residual CNN with CBAM Spatial Attention + Dropout
2. TransferChestClassifier: ImageNet-pretrained ResNet18 / MobileNetV2 backbones
   adapted cleanly for 1-channel radiographic inputs with MC-Dropout uncertainty heads.
"""

import torch
import torch.nn as nn
from torchvision import models


class SpatialAttention(nn.Module):
    """Spatial Attention Module (CBAM-inspired) to highlight salient pathological regions."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv1(scale)
        return x * self.sigmoid(scale)


class ResBlock(nn.Module):
    """Residual building block with Batch Normalization and skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return self.relu(out)


class ResidualChestCNN(nn.Module):
    """
    End-to-End Deep Residual Convolutional Neural Network for Chest Radiographs.
    Designed for high sensitivity, specificity, and transparent interpretability.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = ResBlock(32, 32, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64x64

        self.layer2 = ResBlock(32, 64, stride=2)           # 32x32
        self.layer3 = ResBlock(64, 128, stride=2)          # 16x16
        self.spatial_attention = SpatialAttention(kernel_size=7)

        # Final convolutional layer for Grad-CAM localization
        self.layer4 = ResBlock(128, 256, stride=2)         # 8x8

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.layer1(out)
        out = self.pool1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.spatial_attention(out)
        out = self.layer4(out)
        out = self.gap(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class TransferChestClassifier(nn.Module):
    """
    Transfer Learning Chest Radiograph Classifier wrapping pretrained
    TorchVision backbones (ResNet18, MobileNetV2) with MC-Dropout classification heads
    and exposed Grad-CAM target layer (`gradcam_layer`).
    """

    def __init__(self, arch: str = "resnet18", num_classes: int = 2, pretrained: bool = True, freeze_stem: bool = True):
        super().__init__()
        self.arch = arch
        if arch == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            if freeze_stem and pretrained:
                for name, param in backbone.named_parameters():
                    if not (name.startswith("layer4") or name.startswith("fc")):
                        param.requires_grad = False
            in_features = backbone.fc.in_features
            backbone.fc = nn.Sequential(
                nn.Dropout(p=0.30),
                nn.Linear(in_features, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.20),
                nn.Linear(128, num_classes),
            )
            self.backbone = backbone
            self.gradcam_layer = self.backbone.layer4[-1]
        elif arch == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            backbone = models.mobilenet_v2(weights=weights)
            if freeze_stem and pretrained:
                for i, child in enumerate(backbone.features):
                    if i < 14:
                        for p in child.parameters():
                            p.requires_grad = False
            in_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.30),
                nn.Linear(in_features, num_classes),
            )
            self.backbone = backbone
            self.gradcam_layer = self.backbone.features[-1]
        else:
            raise ValueError(f"Unsupported transfer architecture: {arch}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)
