"""Federated Averaging (McMahan et al., 2017).

Clients train locally starting from the current global model ``w_t`` and report a
sample-count-weighted update. The server applies

    w_{t+1} = w_t + eta_s * (sum_k n_k * (w_k - w_t)) / (sum_k n_k)

which with the default server learning rate ``eta_s = 1`` is exactly the weighted
average of the client models. Clients transmit the *pre-weighted* vector
``[n_k * delta_k, n_k, n_k * loss_k]`` so that the identical aggregation code works
whether the server sees individual vectors (plain FL) or only their sum (secure
aggregation).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class AggregateUpdate:
    """Decoded sum of client contribution vectors."""

    mean_delta: np.ndarray
    total_weight: float
    mean_loss: float


def build_contribution(delta: np.ndarray, num_samples: int, loss: float) -> np.ndarray:
    """Client-side: ``[n * delta, n, n * loss]`` (float64)."""
    n = float(num_samples)
    return np.concatenate([n * np.asarray(delta, dtype=np.float64), [n, n * float(loss)]])


def decode_aggregate(summed: np.ndarray) -> AggregateUpdate:
    """Server-side: turn ``sum_k [n_k * delta_k, n_k, n_k * loss_k]`` into averages."""
    total = float(summed[-2])
    if total <= 0:
        raise ValueError("Aggregate carries no training samples (all clients dropped?)")
    return AggregateUpdate(summed[:-2] / total, total, float(summed[-1]) / total)


def weighted_average(vectors: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """Reference weighted average (used in tests and by the local-only baseline)."""
    w = np.asarray(weights, dtype=np.float64)
    if w.sum() <= 0:
        raise ValueError("Weights must sum to a positive value")
    return np.tensordot(w / w.sum(), np.stack(vectors).astype(np.float64), axes=1)


class FedAvg:
    """FedAvg server strategy."""

    name = "fedavg"

    def __init__(self, server_learning_rate: float = 1.0) -> None:
        self.server_learning_rate = server_learning_rate

    def client_instructions(self) -> dict[str, Any]:
        """Hyper-parameters broadcast to clients with the global model."""
        return {"proximal_mu": 0.0}

    def apply(self, global_vector: np.ndarray, aggregate: AggregateUpdate) -> np.ndarray:
        return global_vector + self.server_learning_rate * aggregate.mean_delta
