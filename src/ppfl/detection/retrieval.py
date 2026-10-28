"""Retrieval-based alert triage: compare an alert with the gateway's own labelled history.

A gateway indexes the latent codes of its own training and validation records (normal
traffic and known attacks; held-out zero-day classes are never among them). For an
alert it retrieves the ``k`` nearest records. The majority label says what the alert
resembles. The mean neighbour distance, divided by the distance a known record usually
has to its own neighbours (95th percentile), says whether the alert is a variant of
known traffic (ratio <= 1) or a new pattern (ratio > 1). It all runs on the gateway, so
no traffic leaves it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import nn


def latent_embedder(model: nn.Module, batch_size: int = 4096) -> Callable[[np.ndarray], np.ndarray]:
    """``X -> model.encode(X)`` as a NumPy function (the autoencoder's latent space)."""

    @torch.no_grad()
    def embed(X: np.ndarray) -> np.ndarray:
        model.eval()
        return np.concatenate([model.encode(torch.from_numpy(np.ascontiguousarray(X[s : s + batch_size]))).numpy()
                               for s in range(0, len(X), batch_size)])

    return embed


class GatewayMemory:
    """Nearest-neighbour index over one gateway's labelled records in an embedding space."""

    def __init__(self, embed: Callable[[np.ndarray], np.ndarray], X: np.ndarray, labels: np.ndarray, k: int = 10, seed: int = 0) -> None:
        self.embed, self.labels, self.k = embed, np.asarray(labels), k
        z = embed(X)
        self.index = NearestNeighbors(n_neighbors=k).fit(z)
        sample = np.random.default_rng(seed).choice(len(z), min(2000, len(z)), replace=False)
        d, _ = self.index.kneighbors(z[sample], k + 1)  # +1: the record itself
        self.typical = float(np.percentile(d[:, 1:].mean(axis=1), 95))

    def query(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(distance_ratio, closest_label, neighbour_labels)`` for every row of ``X``."""
        d, i = self.index.kneighbors(self.embed(X))
        neighbours = self.labels[i]
        closest = np.array([Counter(row).most_common(1)[0][0] for row in neighbours])
        return d.mean(axis=1) / self.typical, closest, neighbours
