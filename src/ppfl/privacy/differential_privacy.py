"""DP-SGD (Abadi et al., 2016) for local client training, built on Opacus.

Per optimizer step the mechanism

1. samples a batch by **Poisson sampling** with rate ``q`` (each record independently),
2. computes **per-sample gradients** (Opacus ``GradSampleModule``),
3. **clips** each per-sample gradient to L2 norm ``C`` (``max_grad_norm``),
4. sums them and adds **Gaussian noise** ``N(0, (sigma * C)^2 I)``,
5. divides by the expected batch size and applies the optimizer update,
6. records the step in the client's privacy accountant.

Everything a client sends afterwards (its model delta, possibly masked by secure
aggregation) is post-processing of these noisy updates, so the client's
``(epsilon, delta)`` covers it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from opacus import GradSampleModule
from opacus.data_loader import DPDataLoader
from opacus.optimizers import DPOptimizer
from opacus.validators import ModuleValidator
from torch import nn
from torch.utils.data import DataLoader

from ppfl.privacy.accountant import PrivacyAccountant, calibrate_noise_multiplier
from ppfl.utils.config import PrivacyConfig
from ppfl.utils.logging import get_logger
from ppfl.utils.seed import torch_generator

LOGGER = get_logger(__name__)


def validate_dp_model(model: nn.Module) -> None:
    """Raise if the model contains layers unsupported by per-sample gradients (e.g. BatchNorm)."""
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        raise ValueError(f"Model is not DP-SGD compatible: {errors}")


@dataclass
class PrivateTrainingObjects:
    module: GradSampleModule
    optimizer: DPOptimizer
    loader: DataLoader


class DPSGDEngine:
    """Per-client DP-SGD state that persists across federated rounds."""

    def __init__(
        self,
        cfg: PrivacyConfig,
        num_samples: int,
        batch_size: int,
        planned_steps: int,
        seed: int,
        client_id: int = 0,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("DP-SGD requires a non-empty local training set")
        self.cfg = cfg
        self.num_samples = num_samples
        self.batches_per_epoch = math.ceil(num_samples / batch_size)
        # Matches opacus.DPDataLoader: q = 1 / len(data_loader).
        self.sample_rate = 1.0 / self.batches_per_epoch
        self.expected_batch_size = max(1, int(num_samples * self.sample_rate))
        self.max_grad_norm = cfg.max_grad_norm
        self.accountant = PrivacyAccountant(cfg.accountant)
        if cfg.target_epsilon is not None:
            self.noise_multiplier = calibrate_noise_multiplier(
                cfg.target_epsilon, cfg.delta, self.sample_rate, planned_steps, cfg.accountant
            )
            LOGGER.info(
                "client_%d: calibrated noise multiplier %.3f for target epsilon %.2f (q=%.4f, %d planned steps)",
                client_id, self.noise_multiplier, cfg.target_epsilon, self.sample_rate, planned_steps,
            )
        else:
            self.noise_multiplier = cfg.noise_multiplier
        if cfg.delta >= 1.0 / num_samples:
            LOGGER.warning(
                "client_%d: delta=%.1e is not smaller than 1/n=%.1e; the guarantee is weak",
                client_id, cfg.delta, 1.0 / num_samples,
            )
        self._noise_gen = None if cfg.secure_mode else torch_generator(seed, "dp-noise", client_id)
        self._sample_gen = None if cfg.secure_mode else torch_generator(seed, "dp-sampling", client_id)

    def make_private(self, model: nn.Module, optimizer: torch.optim.Optimizer, loader: DataLoader) -> PrivateTrainingObjects:
        validate_dp_model(model)
        module = GradSampleModule(model, batch_first=True, loss_reduction="mean")
        dp_optimizer = DPOptimizer(
            optimizer,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            expected_batch_size=self.expected_batch_size,
            loss_reduction="mean",
            generator=self._noise_gen,
            secure_mode=self.cfg.secure_mode,
        )
        dp_optimizer.attach_step_hook(self.accountant.optimizer_hook(self.sample_rate))
        if self.cfg.poisson_sampling:
            loader = DPDataLoader.from_data_loader(loader, generator=self._sample_gen)
        return PrivateTrainingObjects(module, dp_optimizer, loader)

    @staticmethod
    def release(objects: PrivateTrainingObjects) -> None:
        """Remove Opacus hooks so the underlying model can be reused normally."""
        objects.module._close()

    def epsilon(self, delta: float | None = None) -> float:
        return self.accountant.get_epsilon(self.cfg.delta if delta is None else delta)

    @property
    def steps(self) -> int:
        return self.accountant.steps


def laplace_mechanism(values: np.ndarray, epsilon: float, rng: np.random.Generator, sensitivity: float = 1.0) -> np.ndarray:
    """Add Laplace(sensitivity / epsilon) noise (used for DP validation histograms)."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return values + rng.laplace(0.0, sensitivity / epsilon, size=np.shape(values))
