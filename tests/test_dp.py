"""DP-SGD mechanism and privacy accounting."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from ppfl.models.autoencoder import Autoencoder
from ppfl.privacy.accountant import PrivacyAccountant, PrivacyLedger, calibrate_noise_multiplier, compute_epsilon
from ppfl.privacy.differential_privacy import DPSGDEngine, laplace_mechanism, validate_dp_model
from ppfl.training.trainer import train_autoencoder
from ppfl.utils.config import PrivacyConfig, TrainingConfig
from ppfl.utils.serialization import flatten_state_dict

CPU = torch.device("cpu")


def _one_step_update(noise: float, clip: float, seed: int = 0, n: int = 64, scale: float = 100.0) -> np.ndarray:
    """Parameter change after exactly one DP-SGD step with SGD(lr=1) and fixed batches."""
    rng = np.random.default_rng(0)
    X = (scale * rng.normal(size=(n, 8))).astype(np.float32)  # huge gradients -> clipping binds
    torch.manual_seed(0)
    model = Autoencoder(8, [6], 2)
    before, _ = flatten_state_dict(model.state_dict())
    cfg = PrivacyConfig(enabled=True, noise_multiplier=noise, max_grad_norm=clip, poisson_sampling=False)
    engine = DPSGDEngine(cfg, num_samples=n, batch_size=n, planned_steps=1, seed=seed)
    train_autoencoder(model, X, epochs=1, cfg=TrainingConfig(batch_size=n, optimizer="sgd", learning_rate=1.0, momentum=0.0),
                      device=CPU, dp_engine=engine)
    after, _ = flatten_state_dict(model.state_dict())
    return after - before


def test_per_sample_clipping_bounds_the_update():
    for clip in (0.1, 1.0):
        update = _one_step_update(noise=0.0, clip=clip)
        # mean of per-sample gradients each clipped to norm <= C has norm <= C
        assert np.linalg.norm(update) <= clip * (1 + 1e-4)
        # A single example with a huge gradient is clipped to norm exactly C.
        assert np.linalg.norm(_one_step_update(noise=0.0, clip=clip, n=1)) == pytest.approx(clip, rel=1e-4)


def test_gaussian_noise_is_added_and_seeded():
    a, b = _one_step_update(noise=1.0, clip=1.0, seed=0), _one_step_update(noise=1.0, clip=1.0, seed=0)
    c = _one_step_update(noise=1.0, clip=1.0, seed=1)
    assert np.allclose(a, b)  # reproducible for a given seed
    assert not np.allclose(a, c)  # different noise for a different seed
    clean = _one_step_update(noise=0.0, clip=1.0)
    n_params = a.size
    # noise std on the averaged gradient: sigma * C / B per coordinate
    assert np.std(a - clean) == pytest.approx(1.0 * 1.0 / 64, rel=0.25)
    assert n_params > 0


def test_epsilon_grows_with_steps_and_shrinks_with_noise():
    q, delta = 0.01, 1e-5
    assert compute_epsilon(1.0, q, 100, delta) < compute_epsilon(1.0, q, 1000, delta)
    assert compute_epsilon(2.0, q, 1000, delta) < compute_epsilon(1.0, q, 1000, delta)
    assert compute_epsilon(1.0, q, 0, delta) == 0.0


def test_accountant_tracks_training_steps():
    X = np.random.default_rng(0).normal(size=(500, 6)).astype(np.float32)
    engine = DPSGDEngine(PrivacyConfig(enabled=True, noise_multiplier=1.1), num_samples=500, batch_size=50, planned_steps=20, seed=0)
    assert engine.epsilon() == 0.0
    train_autoencoder(Autoencoder(6, [4], 2), X, epochs=1, cfg=TrainingConfig(batch_size=50), device=CPU, dp_engine=engine)
    eps1 = engine.epsilon()
    train_autoencoder(Autoencoder(6, [4], 2), X, epochs=1, cfg=TrainingConfig(batch_size=50), device=CPU, dp_engine=engine)
    assert engine.steps == 20
    assert engine.epsilon() > eps1 > 0
    assert engine.sample_rate == pytest.approx(0.1)
    assert engine.epsilon() == pytest.approx(compute_epsilon(1.1, 0.1, 20, 1e-5), rel=1e-6)


def test_noise_calibration_reaches_target():
    sigma = calibrate_noise_multiplier(target_epsilon=2.0, delta=1e-5, sample_rate=0.02, steps=2000)
    assert compute_epsilon(sigma, 0.02, 2000, 1e-5) <= 2.0 + 1e-2
    engine = DPSGDEngine(PrivacyConfig(enabled=True, target_epsilon=2.0), num_samples=5000, batch_size=100, planned_steps=2000, seed=0)
    assert engine.noise_multiplier == pytest.approx(sigma, rel=1e-3)


def test_accountant_manual_record_and_ledger():
    acc = PrivacyAccountant("rdp")
    acc.record(1.0, 0.05, steps=10)
    assert acc.steps == 10 and acc.get_epsilon(1e-5) == pytest.approx(compute_epsilon(1.0, 0.05, 10, 1e-5))
    ledger = PrivacyLedger(delta=1e-5)
    ledger.record(1, 0, 0.5, 10, 1.0, 0.05)
    ledger.record(1, 1, 0.7, 10, 1.0, 0.05)
    ledger.record(2, 0, 0.9, 20, 1.0, 0.05)
    by_round = ledger.epsilon_by_round()
    assert by_round["epsilon_max"].tolist() == [0.7, 0.9]


def test_batchnorm_models_are_rejected():
    with pytest.raises(ValueError, match="not DP-SGD compatible"):
        validate_dp_model(nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4)))
    validate_dp_model(Autoencoder(4, [3], 2))


def test_laplace_mechanism_scale():
    rng = np.random.default_rng(0)
    noisy = laplace_mechanism(np.zeros(200000), epsilon=0.5, rng=rng)
    assert np.mean(np.abs(noisy)) == pytest.approx(2.0, rel=0.02)  # E|Lap(b)| = b = 1/eps
    with pytest.raises(ValueError):
        laplace_mechanism(np.zeros(3), 0.0, rng)
