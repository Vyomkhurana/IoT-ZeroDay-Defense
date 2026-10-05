"""Autoencoder training loop shared by the centralized, local-only and federated modes.

One function, :func:`train_autoencoder`, handles plain training, FedProx's proximal
term and DP-SGD so that the modes differ only in configuration, never in code paths
that could make comparisons unfair.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ppfl.federated.fedprox import ProximalTerm
from ppfl.models.autoencoder import Autoencoder
from ppfl.privacy.differential_privacy import DPSGDEngine
from ppfl.utils.config import TrainingConfig


@dataclass
class TrainingResult:
    epoch_losses: list[float] = field(default_factory=list)
    num_samples: int = 0
    num_steps: int = 0
    wall_time_s: float = 0.0

    @property
    def final_loss(self) -> float:
        return self.epoch_losses[-1] if self.epoch_losses else float("nan")


def build_optimizer(params, cfg: TrainingConfig) -> torch.optim.Optimizer:
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer {cfg.optimizer!r}")


def make_loader(X: np.ndarray, batch_size: int, generator: torch.Generator | None) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator, drop_last=False)


def train_autoencoder(
    model: nn.Module,
    X: np.ndarray,
    *,
    epochs: int,
    cfg: TrainingConfig,
    device: torch.device,
    generator: torch.Generator | None = None,
    proximal_mu: float = 0.0,
    dp_engine: DPSGDEngine | None = None,
    epoch_callback: Callable[[int, float], None] | None = None,
) -> TrainingResult:
    """Train ``model`` in place to reconstruct ``X`` (MSE).

    Args:
        proximal_mu: FedProx coefficient; the anchor is the model state on entry.
        dp_engine: if given, training uses DP-SGD (per-sample clipping + Gaussian noise).
        epoch_callback: called with ``(epoch, mean_train_loss)`` after every epoch.
    """
    if len(X) == 0:
        raise ValueError("Cannot train on an empty dataset")
    model.to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    proximal = ProximalTerm(params, proximal_mu) if proximal_mu > 0 else None
    optimizer = build_optimizer(params, cfg)
    loader = make_loader(X, cfg.batch_size, generator)
    private = dp_engine.make_private(model, optimizer, loader) if dp_engine is not None else None
    forward = private.module if private else model
    if private:
        loader = private.loader

    result = TrainingResult(num_samples=len(X))
    start = time.perf_counter()
    try:
        for epoch in range(epochs):
            loss_sum, seen = 0.0, 0
            for (xb,) in loader:
                xb = xb.to(device)
                per_sample = Autoencoder.per_sample_error(xb, forward(xb))
                # An empty Poisson batch still performs a (noise-only) DP step.
                loss = per_sample.mean() if len(xb) else per_sample.sum()
                if private:
                    private.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if private.optimizer.pre_step():  # clip -> noise -> scale -> account
                        if proximal is not None:
                            proximal.add_gradient_(params)  # data-independent: post-processing
                        private.optimizer.original_optimizer.step()
                else:
                    optimizer.zero_grad(set_to_none=True)
                    if proximal is not None:
                        loss = loss + proximal.penalty(params)
                    loss.backward()
                    if cfg.grad_clip_norm is not None:
                        nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm)
                    optimizer.step()
                result.num_steps += 1
                loss_sum += float(per_sample.detach().sum())
                seen += len(xb)
            epoch_loss = loss_sum / max(seen, 1)
            result.epoch_losses.append(epoch_loss)
            if epoch_callback is not None:
                epoch_callback(epoch, epoch_loss)
    finally:
        if private:
            DPSGDEngine.release(private)
    result.wall_time_s = time.perf_counter() - start
    return result
