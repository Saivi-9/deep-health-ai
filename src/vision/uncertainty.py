"""
Epistemic Uncertainty Estimation via Monte Carlo Dropout
--------------------------------------------------------
Estimates model uncertainty in clinical chest radiograph diagnostics using
stochastic Monte Carlo forward passes through dropout layers:
- Mean class probability
- Predictive variance (epistemic uncertainty)
- Shannon predictive entropy
- Clinical actionability thresholds (Radiologist Referral flagging)
"""

from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MonteCarloDropoutEstimator:
    """Estimates prediction uncertainty using Monte Carlo Dropout."""

    def __init__(self, model: nn.Module, n_samples: int = 20, device: str = "cpu"):
        self.model = model
        self.n_samples = n_samples
        self.device = torch.device(device)
        self.model.to(self.device)

    def _enable_dropout(self):
        """Forces dropout layers into train mode while keeping BatchNorm in eval mode."""
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                module.train()

    def estimate_uncertainty(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Runs N stochastic forward passes on a single input tensor of shape (1, C, H, W).
        Returns predictive statistics, variance, entropy, and clinical risk flag.
        """
        self._enable_dropout()
        input_tensor = input_tensor.to(self.device)

        probabilities = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                logits = self.model(input_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                probabilities.append(probs)

        probs_arr = np.array(probabilities)  # Shape (N, num_classes)
        mean_probs = np.mean(probs_arr, axis=0)
        variance = np.var(probs_arr, axis=0)

        # Predictive entropy: H(p) = - sum(p * log(p))
        eps = 1e-9
        entropy = -float(np.sum(mean_probs * np.log(mean_probs + eps)))
        norm_entropy = float(entropy / np.log(len(mean_probs)))  # [0, 1]

        pred_class = int(np.argmax(mean_probs))
        pred_var = float(variance[pred_class])

        # Clinical referral criteria based on predictive variance
        if pred_var < 0.01:
            risk_level = "Low Uncertainty (Reliable Automated Diagnosis)"
            referral_required = False
        elif pred_var < 0.035:
            risk_level = "Moderate Uncertainty (Clinical Verification Advised)"
            referral_required = False
        else:
            risk_level = "High Uncertainty (Radiologist Secondary Review Mandatory)"
            referral_required = True

        return {
            "predicted_class": pred_class,
            "mean_probability": float(mean_probs[pred_class]),
            "class_probabilities": mean_probs.tolist(),
            "predictive_variance": pred_var,
            "normalized_entropy": norm_entropy,
            "uncertainty_risk_level": risk_level,
            "human_referral_mandatory": referral_required,
            "monte_carlo_samples": self.n_samples
        }
