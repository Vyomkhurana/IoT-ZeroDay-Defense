"""FedProx (Li et al., 2020): FedAvg with a proximal term in the local objective.

Each client minimises

    h_k(w) = F_k(w) + (mu / 2) * ||w - w_t||^2

where ``w_t`` is the global model received at the start of the round. The proximal
term limits client drift on heterogeneous (non-IID) data. Server-side aggregation is
identical to FedAvg.

Interaction with DP-SGD: the proximal gradient ``mu * (w - w_t)`` does not depend on
any training record, so it is added *after* the per-sample clipping and noising
(see :mod:`ppfl.training.trainer`). Adding a data-independent term is
post-processing and leaves the DP guarantee unchanged, while including it in the
loss would wrongly subject it to per-sample clipping.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ppfl.federated.fedavg import FedAvg


class ProximalTerm:
    """``(mu / 2) * ||w - w_global||^2`` and its gradient, for a fixed global model."""

    def __init__(self, global_params: list[torch.Tensor], mu: float) -> None:
        if mu < 0:
            raise ValueError("mu must be non-negative")
        self.mu = mu
        self.global_params = [p.detach().clone() for p in global_params]

    @classmethod
    def from_model(cls, model: nn.Module, mu: float) -> ProximalTerm:
        return cls(list(model.parameters()), mu)

    def penalty(self, params: list[torch.Tensor]) -> torch.Tensor:
        total = sum(((p - g) ** 2).sum() for p, g in zip(params, self.global_params))
        return 0.5 * self.mu * total

    @torch.no_grad()
    def add_gradient_(self, params: list[torch.Tensor]) -> None:
        """In-place ``p.grad += mu * (p - g)``."""
        for p, g in zip(params, self.global_params):
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            p.grad.add_(p.detach() - g, alpha=self.mu)


class FedProx(FedAvg):
    """FedProx server strategy: FedAvg aggregation + proximal coefficient for clients."""

    name = "fedprox"

    def __init__(self, mu: float = 0.01, server_learning_rate: float = 1.0) -> None:
        super().__init__(server_learning_rate)
        if mu < 0:
            raise ValueError("FedProx mu must be non-negative")
        self.mu = mu

    def client_instructions(self) -> dict[str, Any]:
        return {"proximal_mu": self.mu}


def build_strategy(name: str, mu: float, server_learning_rate: float) -> FedAvg:
    if name == "fedavg":
        return FedAvg(server_learning_rate)
    if name == "fedprox":
        return FedProx(mu, server_learning_rate)
    raise ValueError(f"Unknown federated strategy {name!r}")
