"""Fully-connected autoencoder for unsupervised network-traffic anomaly detection.

Architecture (symmetric, all sizes configurable)::

    x (input_dim) -> [Linear -> act -> (dropout)] x len(hidden_dims) -> Linear -> z (latent_dim)
    z             -> [Linear -> act -> (dropout)] x len(hidden_dims) -> Linear -> x_hat (input_dim)

The bottleneck and output layers are linear (inputs are standardised, so the
reconstruction target is unbounded). Only ``Linear``/element-wise layers are used,
which keeps the model compatible with Opacus per-sample gradient computation
(no BatchNorm).
"""

from __future__ import annotations

import torch
from torch import nn

from ppfl.utils.config import ModelConfig

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}


def _mlp(dims: list[int], activation: str, dropout: float) -> nn.Sequential:
    """Linear stack with activations between layers and a linear final layer."""
    layers: list[nn.Module] = []
    for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(d_in, d_out))
        if i < len(dims) - 2:
            layers.append(_ACTIVATIONS[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class Autoencoder(nn.Module):
    """Symmetric MLP autoencoder exposing ``encode``, ``decode`` and ``forward``."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | tuple[int, ...] = (64, 32),
        latent_dim: int = 16,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation {activation!r}; choose from {sorted(_ACTIVATIONS)}")
        self.input_dim = int(input_dim)
        self.hidden_dims = [int(h) for h in hidden_dims]
        self.latent_dim = int(latent_dim)
        self.encoder = _mlp([self.input_dim, *self.hidden_dims, self.latent_dim], activation, dropout)
        self.decoder = _mlp([self.latent_dim, *reversed(self.hidden_dims), self.input_dim], activation, dropout)

    @classmethod
    def from_config(cls, input_dim: int, cfg: ModelConfig) -> Autoencoder:
        if cfg.type != "autoencoder":
            raise ValueError(f"Unsupported model type {cfg.type!r}")
        return cls(input_dim, cfg.hidden_dims, cfg.latent_dim, cfg.activation, cfg.dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    @staticmethod
    def per_sample_error(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """Mean squared reconstruction error of each sample (the anomaly score)."""
        return ((x_hat - x) ** 2).mean(dim=1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def architecture(self) -> dict[str, object]:
        return {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "latent_dim": self.latent_dim,
            "num_parameters": self.num_parameters(),
        }
