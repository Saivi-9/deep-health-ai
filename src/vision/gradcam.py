"""
Gradient-Weighted Class Activation Mapping (Grad-CAM)
-----------------------------------------------------
Produces visual explainability heatmaps for clinical decision support by
computing gradients of target class score with respect to convolutional feature maps.
"""

from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM implementation for PyTorch vision architectures.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Register hooks
        self.hook_handles = [
            self.target_layer.register_forward_hook(self._forward_hook),
            self.target_layer.register_full_backward_hook(self._backward_hook)
        ]

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generates 2D normalized Grad-CAM heatmap for the specified class.
        If target_class is None, uses the model's highest scoring predicted class.
        """
        self.model.eval()
        # Ensure gradients are enabled for the input pass
        input_tensor = input_tensor.clone().requires_grad_(True)
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        confidence = float(probs[0, target_class].item())

        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # Global average pooling of gradients: alpha_k = 1/Z * sum(grad)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weighted combination of activation maps
        activation_maps = self.activations[0]
        for i in range(activation_maps.shape[0]):
            activation_maps[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activation_maps, dim=0).cpu()
        # Pass through ReLU (only features that contribute positively to the class)
        heatmap = F.relu(heatmap)

        # Normalize to [0, 1]
        heatmap_np = heatmap.numpy()
        denom = np.max(heatmap_np) - np.min(heatmap_np)
        if denom > 1e-8:
            heatmap_np = (heatmap_np - np.min(heatmap_np)) / denom
        else:
            heatmap_np = np.zeros_like(heatmap_np)

        # Upsample heatmap to input image dimensions
        H, W = input_tensor.shape[2], input_tensor.shape[3]
        heatmap_tensor = torch.from_numpy(heatmap_np).unsqueeze(0).unsqueeze(0)
        upsampled = F.interpolate(heatmap_tensor, size=(H, W), mode="bilinear", align_corners=False)
        upsampled_np = upsampled.squeeze().numpy()

        return upsampled_np, target_class, confidence

    def remove_hooks(self):
        """Clean up hooks to prevent memory leaks."""
        for handle in self.hook_handles:
            handle.remove()
