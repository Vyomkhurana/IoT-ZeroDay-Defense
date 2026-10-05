"""Autoencoder, anomaly scoring and threshold selection."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ppfl.detection.anomaly_score import reconstruction_errors
from ppfl.detection.threshold import (
    ThresholdSelector,
    f1_optimal_threshold,
    histogram_counts,
    histogram_quantile,
    log_edges,
    percentile_threshold,
)
from ppfl.models.autoencoder import Autoencoder
from ppfl.training.trainer import train_autoencoder
from ppfl.utils.config import DetectionConfig, ModelConfig, TrainingConfig


def test_forward_encode_decode_shapes():
    model = Autoencoder.from_config(23, ModelConfig(hidden_dims=[12, 8], latent_dim=3))
    x = torch.randn(7, 23)
    assert model.encode(x).shape == (7, 3)
    assert model.decode(model.encode(x)).shape == (7, 23)
    assert model(x).shape == (7, 23)
    assert model.architecture()["num_parameters"] == model.num_parameters() > 0


def test_invalid_model_arguments():
    with pytest.raises(ValueError):
        Autoencoder(0)
    with pytest.raises(ValueError):
        Autoencoder(5, activation="swishy")


def test_training_reduces_reconstruction_error():
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(1500, 3))
    X = (latent @ rng.normal(size=(3, 12)) + 0.05 * rng.normal(size=(1500, 12))).astype(np.float32)
    torch.manual_seed(0)
    model = Autoencoder(12, [16], 3)
    before = reconstruction_errors(model, X).mean()
    result = train_autoencoder(model, X, epochs=15, cfg=TrainingConfig(batch_size=64, learning_rate=3e-3),
                               device=torch.device("cpu"), generator=torch.Generator().manual_seed(0))
    after = reconstruction_errors(model, X).mean()
    assert after < 0.25 * before
    assert result.epoch_losses[-1] < result.epoch_losses[0]
    assert result.num_steps == 15 * int(np.ceil(1500 / 64))


def test_anomalies_score_higher_after_training():
    rng = np.random.default_rng(1)
    basis = rng.normal(size=(2, 10))
    normal = (rng.normal(size=(2000, 2)) @ basis).astype(np.float32)
    anomalies = rng.normal(scale=2.0, size=(200, 10)).astype(np.float32)  # off the normal manifold
    torch.manual_seed(1)
    model = Autoencoder(10, [8], 2)
    train_autoencoder(model, normal, epochs=20, cfg=TrainingConfig(batch_size=64, learning_rate=3e-3), device=torch.device("cpu"))
    assert np.median(reconstruction_errors(model, anomalies)) > 10 * np.median(reconstruction_errors(model, normal))


def test_percentile_threshold():
    scores = np.arange(1, 101, dtype=float)
    assert percentile_threshold(scores, 95) == pytest.approx(np.percentile(scores, 95))
    assert (scores > percentile_threshold(scores, 95)).mean() == pytest.approx(0.05)


def test_f1_optimal_threshold_separable_and_ties():
    benign = np.array([0.1, 0.2, 0.3, 0.3])
    attack = np.array([0.8, 0.9, 1.0])
    thr, f1 = f1_optimal_threshold(benign, attack)
    assert 0.3 < thr < 0.8 and f1 == pytest.approx(1.0)
    # Tied scores (0.5) cannot be split: candidates are "flag 0.9 only" (F1 = 2/3) or
    # "flag everything" (F1 = 2/3); the higher threshold (fewer false positives) wins.
    thr2, f12 = f1_optimal_threshold(np.array([0.5, 0.5]), np.array([0.5, 0.9]))
    assert thr2 == pytest.approx(0.7) and f12 == pytest.approx(2 / 3)


def test_histogram_path_matches_exact():
    rng = np.random.default_rng(2)
    benign = rng.lognormal(mean=-3, sigma=0.8, size=20000)
    attack = rng.lognormal(mean=-1, sigma=0.8, size=3000)
    edges = log_edges(1e-8, 1e6, 4096)
    q = histogram_quantile(histogram_counts(benign, edges), edges, 95)
    assert q == pytest.approx(np.percentile(benign, 95), rel=0.01)
    for strategy in ("percentile", "validation_f1", "mean_std"):
        sel = ThresholdSelector(DetectionConfig(threshold_strategy=strategy))
        exact = sel.from_scores(benign, attack).value
        approx = sel.from_histogram_vector(sel.histograms(benign, attack)).value
        assert approx == pytest.approx(exact, rel=0.03), strategy


def test_histogram_sum_over_clients_equals_pooled():
    rng = np.random.default_rng(3)
    sel = ThresholdSelector(DetectionConfig())
    parts = [(rng.exponential(size=500), rng.exponential(5, size=50)) for _ in range(4)]
    summed = np.sum([sel.histograms(b, a) for b, a in parts], axis=0)
    pooled = sel.histograms(np.concatenate([b for b, _ in parts]), np.concatenate([a for _, a in parts]))
    assert np.array_equal(summed, pooled)


def test_validation_f1_falls_back_without_attacks():
    sel = ThresholdSelector(DetectionConfig(threshold_strategy="validation_f1"))
    res = sel.from_scores(np.linspace(0, 1, 100), np.array([]))
    assert res.strategy == "percentile"
