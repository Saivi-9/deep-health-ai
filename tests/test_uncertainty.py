"""Unit tests for Monte Carlo Dropout Uncertainty Estimation."""

import pytest
import numpy as np
import torch
from src.vision.model import ResidualChestCNN
from src.vision.uncertainty import MonteCarloDropoutEstimator


def test_mc_dropout_output_structure():
    model = ResidualChestCNN(in_channels=1, num_classes=2)
    estimator = MonteCarloDropoutEstimator(model, n_samples=10, device="cpu")
    x = torch.randn(1, 1, 128, 128)
    result = estimator.estimate_uncertainty(x)

    assert "predicted_class" in result
    assert "mean_probability" in result
    assert "predictive_variance" in result
    assert "normalized_entropy" in result
    assert "uncertainty_risk_level" in result
    assert "human_referral_mandatory" in result


def test_mc_dropout_probability_sums_to_one():
    model = ResidualChestCNN(in_channels=1, num_classes=2)
    estimator = MonteCarloDropoutEstimator(model, n_samples=10, device="cpu")
    x = torch.randn(1, 1, 128, 128)
    result = estimator.estimate_uncertainty(x)

    probs_sum = sum(result["class_probabilities"])
    assert abs(probs_sum - 1.0) < 1e-5


def test_mc_dropout_entropy_bounded():
    model = ResidualChestCNN(in_channels=1, num_classes=2)
    estimator = MonteCarloDropoutEstimator(model, n_samples=10, device="cpu")
    x = torch.randn(1, 1, 128, 128)
    result = estimator.estimate_uncertainty(x)

    assert 0.0 <= result["normalized_entropy"] <= 1.0


def test_mc_dropout_referral_is_bool():
    model = ResidualChestCNN(in_channels=1, num_classes=2)
    estimator = MonteCarloDropoutEstimator(model, n_samples=5, device="cpu")
    x = torch.randn(1, 1, 128, 128)
    result = estimator.estimate_uncertainty(x)

    assert isinstance(result["human_referral_mandatory"], bool)
