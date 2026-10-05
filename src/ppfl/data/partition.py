"""Non-IID partitioning of traffic across simulated IoT gateways.

Strategies
----------
``dirichlet``
    For every stratum (by default every ``(device, label)`` pair) client proportions
    are drawn from ``Dir(alpha * 1_K)`` and the stratum's rows are split accordingly
    (Hsu et al., 2019). Small ``alpha`` concentrates each stratum on few clients,
    yielding highly heterogeneous clients (different devices, different attack
    exposure, different data volumes); large ``alpha`` approaches an IID split.
``device``
    Natural partition: each client is a gateway serving one or more physical devices
    (devices are assigned round-robin; if there are more clients than devices a
    device's traffic is split between several gateways).
``iid``
    Uniform random split (reference point for heterogeneity experiments).

Because the autoencoder learns *benign* behaviour, every client is guaranteed at
least ``min_benign_per_client`` benign rows (re-sampling, then a deterministic repair
step that moves benign rows from the largest holders).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from ppfl.utils.logging import get_logger

LOGGER = get_logger(__name__)


def make_strata(labels: np.ndarray, devices: np.ndarray, stratify_by: list[str]) -> np.ndarray:
    keys = []
    if "device" in stratify_by:
        keys.append(np.asarray(devices).astype(str))
    if "label" in stratify_by:
        keys.append(np.asarray(labels).astype(str))
    if not keys:
        return np.zeros(len(labels), dtype=np.int64)
    combined = keys[0] if len(keys) == 1 else np.char.add(np.char.add(keys[0], "||"), keys[1])
    _, ids = np.unique(combined, return_inverse=True)
    return ids.astype(np.int64)


def iid_partition(n: int, num_clients: int, rng: np.random.Generator) -> list[np.ndarray]:
    return [np.sort(p) for p in np.array_split(rng.permutation(n), num_clients)]


def dirichlet_partition(
    strata: np.ndarray, num_clients: int, alpha: float, rng: np.random.Generator
) -> list[np.ndarray]:
    buckets: list[list[np.ndarray]] = [[] for _ in range(num_clients)]
    for s in np.unique(strata):
        idx = rng.permutation(np.flatnonzero(strata == s))
        props = rng.dirichlet(np.full(num_clients, alpha))
        cuts = (np.cumsum(props)[:-1] * len(idx)).astype(int)
        for k, part in enumerate(np.split(idx, cuts)):
            buckets[k].append(part)
    return [np.sort(np.concatenate(b)) if b else np.zeros(0, dtype=np.int64) for b in buckets]


def device_partition(devices: np.ndarray, num_clients: int, rng: np.random.Generator) -> list[np.ndarray]:
    devices = np.asarray(devices).astype(str)
    unique = sorted(set(devices.tolist()))
    order = [unique[i] for i in rng.permutation(len(unique))]
    parts: list[list[np.ndarray]] = [[] for _ in range(num_clients)]
    if num_clients <= len(order):
        for i, dev in enumerate(order):
            parts[i % num_clients].append(np.flatnonzero(devices == dev))
    else:
        owners: dict[str, list[int]] = {d: [] for d in order}
        for k in range(num_clients):
            owners[order[k % len(order)]].append(k)
        for dev, clients in owners.items():
            idx = rng.permutation(np.flatnonzero(devices == dev))
            for k, chunk in zip(clients, np.array_split(idx, len(clients))):
                parts[k].append(chunk)
    return [np.sort(np.concatenate(p)) if p else np.zeros(0, dtype=np.int64) for p in parts]


def _repair_min_benign(
    parts: list[np.ndarray], is_benign: np.ndarray, minimum: int, rng: np.random.Generator
) -> list[np.ndarray]:
    parts = [p.copy() for p in parts]
    for k in range(len(parts)):
        deficit = minimum - int(is_benign[parts[k]].sum())
        while deficit > 0:
            donor = int(np.argmax([is_benign[p].sum() if j != k else -1 for j, p in enumerate(parts)]))
            donor_benign = parts[donor][is_benign[parts[donor]]]
            spare = len(donor_benign) - minimum
            if spare <= 0:
                raise ValueError(
                    f"Not enough benign data to give every client {minimum} benign rows; "
                    "reduce partition.num_clients or partition.min_benign_per_client"
                )
            moved = rng.choice(donor_benign, size=min(deficit, spare), replace=False)
            parts[donor] = np.setdiff1d(parts[donor], moved)
            parts[k] = np.sort(np.concatenate([parts[k], moved]))
            deficit -= len(moved)
    return parts


def partition_indices(
    labels: np.ndarray,
    devices: np.ndarray,
    *,
    num_clients: int,
    strategy: str,
    alpha: float,
    stratify_by: list[str],
    rng: np.random.Generator,
    benign_label: str | None = None,
    min_benign: int = 0,
    max_retries: int = 200,
) -> list[np.ndarray]:
    """Split row indices across clients. ``benign_label=None`` disables the benign floor."""
    n = len(labels)
    if strategy == "iid":
        parts = iid_partition(n, num_clients, rng)
    elif strategy == "device":
        parts = device_partition(devices, num_clients, rng)
    elif strategy == "dirichlet":
        strata = make_strata(labels, devices, stratify_by)
        is_benign = None if benign_label is None else np.asarray(labels) == benign_label
        for attempt in range(max_retries):
            parts = dirichlet_partition(strata, num_clients, alpha, rng)
            if is_benign is None or min(int(is_benign[p].sum()) for p in parts) >= min_benign:
                break
        else:
            LOGGER.warning(
                "Dirichlet partition (alpha=%.3g) did not satisfy min_benign=%d after %d attempts; "
                "applying deterministic repair", alpha, min_benign, max_retries,
            )
    else:
        raise ValueError(f"Unknown partition strategy {strategy!r}")
    if benign_label is not None and min_benign > 0:
        parts = _repair_min_benign(parts, np.asarray(labels) == benign_label, min_benign, rng)
    covered = np.concatenate(parts)
    assert len(covered) == len(np.unique(covered)), "partition overlap"
    return parts


def partition_summary(labels: np.ndarray, parts: list[np.ndarray]) -> pd.DataFrame:
    """Client x class count table."""
    labels = np.asarray(labels).astype(str)
    classes = sorted(set(labels.tolist()))
    rows = []
    for k, idx in enumerate(parts):
        vals, counts = np.unique(labels[idx], return_counts=True)
        row = dict.fromkeys(classes, 0)
        row.update(dict(zip(vals.tolist(), counts.tolist())))
        rows.append(row)
    df = pd.DataFrame(rows, index=pd.Index([f"client_{k}" for k in range(len(parts))], name="client"))
    return df[classes]


def heterogeneity_index(summary: pd.DataFrame) -> float:
    """Mean Jensen-Shannon distance between each client's label mix and the global mix."""
    counts = summary.to_numpy(dtype=np.float64)
    totals = counts.sum(axis=1, keepdims=True)
    valid = totals[:, 0] > 0
    if valid.sum() == 0:
        return 0.0
    client_dist = counts[valid] / totals[valid]
    global_dist = counts.sum(axis=0) / counts.sum()
    return float(np.mean([jensenshannon(c, global_dist, base=2) for c in client_dist]))
