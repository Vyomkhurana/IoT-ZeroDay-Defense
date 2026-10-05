"""Anomaly scoring: per-sample reconstruction error of the autoencoder."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ppfl.models.autoencoder import Autoencoder


@torch.no_grad()
def reconstruction_errors(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int = 4096,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Return ``MSE(x, model(x))`` for every row of ``X`` (float64)."""
    if len(X) == 0:
        return np.zeros(0, dtype=np.float64)
    was_training = model.training
    model.eval()
    out = []
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(X[start : start + batch_size])).to(device)
        out.append(Autoencoder.per_sample_error(xb, model(xb)).double().cpu().numpy())
    model.train(was_training)
    return np.concatenate(out)


class AutoencoderScorer:
    """Callable wrapper turning an autoencoder into an anomaly scorer."""

    def __init__(self, model: nn.Module, batch_size: int = 4096, device: torch.device | str = "cpu") -> None:
        self.model = model
        self.batch_size = batch_size
        self.device = device

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return reconstruction_errors(self.model, X, self.batch_size, self.device)
