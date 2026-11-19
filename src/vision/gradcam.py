"""
Gradient-Weighted Class Activation Mapping (Grad-CAM)
-----------------------------------------------------
Produces visual explainability heatmaps for clinical decision support by
computing gradients of target class score with respect to convolutional feature maps,
plus base64 RGBA thermal map & composited overlay encoders for REST/UI rendering.
"""

import base64
from io import BytesIO
from typing import Tuple, Optional, Dict, Any

import numpy as np
from PIL import Image
import matplotlib
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
        input_tensor = input_tensor.clone().requires_grad_(True)
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        confidence = float(probs[0, target_class].item())

        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activation_maps = self.activations[0].clone()
        for i in range(activation_maps.shape[0]):
            activation_maps[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activation_maps, dim=0).cpu()
        heatmap = F.relu(heatmap)

        heatmap_np = heatmap.numpy()
        denom = np.max(heatmap_np) - np.min(heatmap_np)
        if denom > 1e-8:
            heatmap_np = (heatmap_np - np.min(heatmap_np)) / denom
        else:
            heatmap_np = np.zeros_like(heatmap_np)

        h, w = input_tensor.shape[2], input_tensor.shape[3]
        heatmap_tensor = torch.from_numpy(heatmap_np).unsqueeze(0).unsqueeze(0)
        upsampled = F.interpolate(heatmap_tensor, size=(h, w), mode="bilinear", align_corners=False)
        upsampled_np = upsampled.squeeze().numpy()

        return upsampled_np, target_class, confidence

    def remove_hooks(self):
        """Clean up hooks to prevent memory leaks."""
        for handle in self.hook_handles:
            handle.remove()


def encode_gradcam_artifacts(
    pil_img: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.55,
    colormap_name: str = "turbo",
) -> Dict[str, Any]:
    """
    Renders a transparent thermal heatmap layer (`heatmap_base64`) and a
    blended diagnostic overlay (`overlay_base64`), plus normalized peak coordinates.
    """
    hm = np.clip(np.asarray(heatmap, dtype=np.float32), 0.0, 1.0)
    h, w = hm.shape
    cmap = matplotlib.colormaps.get_cmap(colormap_name)
    rgba = cmap(hm)  # (H, W, 4) float in [0, 1]

    # Transparent PNG heatmap scaled by activation strength so low background stays clean
    alpha_mask = np.clip((hm - 0.12) / 0.88, 0.0, 1.0) * 235.0
    hm_rgba_u8 = (rgba * 255).astype(np.uint8)
    hm_rgba_u8[..., 3] = alpha_mask.astype(np.uint8)
    hm_pil = Image.fromarray(hm_rgba_u8, mode="RGBA")
    buf_hm = BytesIO()
    hm_pil.save(buf_hm, format="PNG")
    hm_b64 = "data:image/png;base64," + base64.b64encode(buf_hm.getvalue()).decode("ascii")

    # Blended overlay PNG
    base_rgb = pil_img.convert("RGB").resize((w, h), Image.Resampling.BILINEAR)
    blend_alpha = (alpha_mask / 255.0 * alpha)[..., None]
    overlay_arr = (
        np.asarray(base_rgb, dtype=np.float32) * (1.0 - blend_alpha)
        + rgba[..., :3] * 255.0 * blend_alpha
    ).clip(0, 255).astype(np.uint8)
    buf_ov = BytesIO()
    Image.fromarray(overlay_arr, mode="RGB").save(buf_ov, format="PNG")
    ov_b64 = "data:image/png;base64," + base64.b64encode(buf_ov.getvalue()).decode("ascii")

    peak_idx = np.unravel_index(int(np.argmax(hm)), hm.shape)
    return {
        "heatmap_base64": hm_b64,
        "overlay_base64": ov_b64,
        "hotspot": {
            "x_norm": round(float(peak_idx[1] / max(1, w - 1)), 4),
            "y_norm": round(float(peak_idx[0] / max(1, h - 1)), 4),
            "peak_activation": round(float(hm[peak_idx]), 4),
        },
    }
