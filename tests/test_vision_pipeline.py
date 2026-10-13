"""Unit tests for Chest X-Ray Vision Pipeline."""

import pytest
import numpy as np
import torch
from src.vision.model import ResidualChestCNN, SpatialAttention
from src.vision.gradcam import GradCAM
from src.vision.dataset import MedicalImageDataset, get_vision_transforms


def test_spatial_attention_shape():
    module = SpatialAttention(kernel_size=7)
    x = torch.randn(2, 64, 16, 16)
    out = module(x)
    assert out.shape == x.shape, f"Expected shape {x.shape}, got {out.shape}"


def test_residual_chest_cnn_forward():
    model = ResidualChestCNN(in_channels=1, num_classes=2)
    model.eval()
    x = torch.randn(4, 1, 128, 128)
    out = model(x)
    assert out.shape == (4, 2), f"Expected logits shape (4, 2), got {out.shape}"


def test_gradcam_generation():
    model = ResidualChestCNN(in_channels=1, num_classes=2)
    gradcam = GradCAM(model, model.layer4)

    x = torch.randn(1, 1, 128, 128)
    heatmap, pred_c, conf = gradcam.generate_heatmap(x, target_class=1)

    assert heatmap.shape == (128, 128), f"Expected heatmap (128, 128), got {heatmap.shape}"
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0, "Heatmap values must be normalized in [0, 1]"
    assert 0.0 <= conf <= 1.0, "Confidence must be a valid probability in [0, 1]"

    gradcam.remove_hooks()


def test_vision_transforms():
    train_tf, eval_tf = get_vision_transforms(image_size=(128, 128))
    from PIL import Image
    dummy_img = Image.fromarray(np.uint8(np.zeros((100, 100))))

    train_tensor = train_tf(dummy_img)
    eval_tensor = eval_tf(dummy_img)

    assert train_tensor.shape == (1, 128, 128)
    assert eval_tensor.shape == (1, 128, 128)
