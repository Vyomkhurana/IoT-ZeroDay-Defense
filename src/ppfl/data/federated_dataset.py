"""End-to-end data pipeline: preparation and construction of the federated dataset.

``prepare_dataset``         raw files -> cleaned table on disk (run once per dataset)
``build_federated_dataset`` cleaned table -> zero-day holdout -> non-IID clients ->
                            local train/val/test splits -> federated scaler

Everything after preparation is a deterministic function of the configuration and
the experiment seed, so partitions never have to be stored to be reproduced.

Data flow per client ``k``::

    known traffic (benign + known attacks) --partition--> client k --local split--> train / val / test
    unseen traffic (held-out attacks)      --partition--> client k ----------------> zero_day (test only)

* ``train``     benign rows train the autoencoder (optionally with a small fraction of
                known-attack rows as unlabelled contamination, ``data.train_contamination``)
* ``val``       benign (+ known attacks) for threshold calibration / validation loss
* ``test``      benign + known attacks, evaluation only
* ``zero_day``  held-out attack classes, evaluation only
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ppfl.data.loader import (
    DEVICE_COL,
    LABEL_COL,
    META_COLS,
    canonical_dataset_name,
    load_processed,
    load_raw_dataset,
    save_processed,
)
from ppfl.data.partition import heterogeneity_index, partition_indices, partition_summary
from ppfl.data.preprocessing import FeatureScaler, FeatureStatistics, clean_dataset
from ppfl.data.zero_day import ZeroDaySplit, make_zero_day_split
from ppfl.utils.config import Config
from ppfl.utils.logging import get_logger
from ppfl.utils.seed import numpy_rng
from ppfl.utils.serialization import to_jsonable

LOGGER = get_logger(__name__)


# --------------------------------------------------------------------------------------
# Containers
# --------------------------------------------------------------------------------------


@dataclass
class Split:
    """Feature matrix with per-row class labels and device names."""

    X: np.ndarray
    labels: np.ndarray
    devices: np.ndarray

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def subset(self, mask: np.ndarray) -> Split:
        return Split(self.X[mask], self.labels[mask], self.devices[mask])

    @staticmethod
    def empty(num_features: int) -> Split:
        return Split(np.zeros((0, num_features), np.float32), np.zeros(0, dtype=str), np.zeros(0, dtype=str))

    @staticmethod
    def concat(splits: list[Split]) -> Split:
        return Split(
            np.concatenate([s.X for s in splits]),
            np.concatenate([s.labels for s in splits]),
            np.concatenate([s.devices for s in splits]),
        )


@dataclass
class ClientData:
    """All data held locally by one simulated IoT gateway."""

    client_id: int
    train: Split
    val: Split
    test: Split
    zero_day: Split

    def training_matrix(self, benign_label: str, contamination: float, rng: np.random.Generator) -> np.ndarray:
        """Autoencoder training data: benign rows (+ optional unlabelled attack contamination)."""
        benign = self.train.labels == benign_label
        X = self.train.X[benign]
        if contamination > 0:
            attacks = np.flatnonzero(~benign)
            n_extra = min(len(attacks), int(round(contamination * benign.sum())))
            if n_extra:
                X = np.concatenate([X, self.train.X[rng.choice(attacks, n_extra, replace=False)]])
        return X

    def local_test(self) -> Split:
        """Client-local evaluation set: test split plus the client's held-out attacks."""
        return Split.concat([self.test, self.zero_day])


@dataclass
class FederatedDataset:
    clients: list[ClientData]
    feature_names: list[str]
    zero_day: ZeroDaySplit
    scaler: FeatureScaler
    partition_table: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def input_dim(self) -> int:
        return len(self.feature_names)

    @property
    def benign_label(self) -> str:
        return self.zero_day.benign_label

    def pooled(self, name: str) -> Split:
        """Concatenate one split (``train``/``val``/``test``/``zero_day``) over all clients."""
        return Split.concat([getattr(c, name) for c in self.clients])

    def global_test(self) -> tuple[Split, np.ndarray]:
        """Global evaluation set (all clients' test + zero-day rows) and the owning client ids."""
        parts, owners = [], []
        for c in self.clients:
            local = c.local_test()
            parts.append(local)
            owners.append(np.full(len(local), c.client_id))
        return Split.concat(parts), np.concatenate(owners)


# --------------------------------------------------------------------------------------
# Preparation (raw -> cleaned parquet)
# --------------------------------------------------------------------------------------


def prepare_dataset(cfg: Config, raw_dir: Any | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load, clean and store a dataset under ``data/processed/<dataset>``."""
    dataset = canonical_dataset_name(cfg.data.dataset)
    cfg = cfg.copy()
    cfg.data.dataset = dataset
    start = time.perf_counter()
    raw = load_raw_dataset(cfg, raw_dir)
    frame, features, report = clean_dataset(raw, cfg.data.benign_label, cfg.data.max_missing_fraction)
    metadata = {
        "dataset": dataset,
        "feature_names": features,
        "num_rows": len(frame),
        "class_counts": frame[LABEL_COL].value_counts().sort_index().to_dict(),
        "device_counts": frame[DEVICE_COL].value_counts().sort_index().to_dict(),
        "cleaning": report.as_dict(),
        "source_files": raw.source_files,
        "caps": {
            "max_samples_per_group": cfg.data.max_samples_per_group,
            "max_benign_samples_per_group": cfg.data.max_benign_samples_per_group,
        },
        "seed": cfg.experiment.seed,
        "synthetic_params": to_jsonable(cfg.data.synthetic) if dataset == "synthetic" else None,
        "prepared_in_seconds": round(time.perf_counter() - start, 2),
    }
    save_processed(frame, metadata, cfg.processed_dir)
    LOGGER.info("Saved processed dataset (%d rows, %d features) to %s", len(frame), len(features), cfg.processed_dir)
    return frame, metadata


def load_or_prepare(cfg: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the processed table; the synthetic dataset is (re)generated on demand."""
    if canonical_dataset_name(cfg.data.dataset) == "synthetic":
        try:
            frame, meta = load_processed(cfg.processed_dir)
            if meta.get("synthetic_params") == to_jsonable(cfg.data.synthetic) and meta.get("seed") == cfg.experiment.seed:
                return frame, meta
            LOGGER.info("Synthetic data parameters changed; regenerating")
        except FileNotFoundError:
            LOGGER.info("Synthetic dataset not prepared yet; generating it now")
        return prepare_dataset(cfg)
    return load_processed(cfg.processed_dir)


# --------------------------------------------------------------------------------------
# Federated construction
# --------------------------------------------------------------------------------------


def _local_split(
    labels: np.ndarray, val_frac: float, test_frac: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified (per label) split of local row positions into train/val/test."""
    train, val, test = [], [], []
    for label in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == label))
        n_test = int(round(len(idx) * test_frac))
        n_val = int(round(len(idx) * val_frac))
        if len(idx) >= 3:
            n_test, n_val = max(n_test, 1), max(n_val, 1)
        test.append(idx[:n_test])
        val.append(idx[n_test : n_test + n_val])
        train.append(idx[n_test + n_val :])
    return tuple(np.sort(np.concatenate(p)) if p else np.zeros(0, np.int64) for p in (train, val, test))


def build_federated_dataset(cfg: Config, frame: pd.DataFrame | None = None) -> FederatedDataset:
    """Construct client datasets according to ``cfg.data`` and ``cfg.partition``."""
    meta: dict[str, Any] = {}
    if frame is None:
        frame, meta = load_or_prepare(cfg)
    seed = cfg.experiment.seed
    features = [c for c in frame.columns if c not in META_COLS]
    labels = frame[LABEL_COL].to_numpy().astype(str)
    devices = frame[DEVICE_COL].to_numpy().astype(str)
    values = frame[features].to_numpy(dtype=np.float32)
    zd = make_zero_day_split(labels, cfg.data.holdout_classes, cfg.data.benign_label)

    unseen_mask = zd.is_unseen_attack(labels)
    known_idx, unseen_idx = np.flatnonzero(~unseen_mask), np.flatnonzero(unseen_mask)
    pc = cfg.partition
    known_parts = partition_indices(
        labels[known_idx], devices[known_idx],
        num_clients=pc.num_clients, strategy=pc.strategy, alpha=pc.dirichlet_alpha,
        stratify_by=pc.stratify_by, rng=numpy_rng(seed, "partition", "known"),
        benign_label=cfg.data.benign_label, min_benign=pc.min_benign_per_client,
        max_retries=pc.max_retries,
    )
    unseen_parts = (
        partition_indices(
            labels[unseen_idx], devices[unseen_idx],
            num_clients=pc.num_clients, strategy=pc.strategy, alpha=pc.dirichlet_alpha,
            stratify_by=pc.stratify_by, rng=numpy_rng(seed, "partition", "unseen"),
        )
        if len(unseen_idx)
        else [np.zeros(0, np.int64)] * pc.num_clients
    )

    raw_clients: list[dict[str, np.ndarray]] = []
    for k in range(pc.num_clients):
        rows = known_idx[known_parts[k]]
        tr, va, te = _local_split(labels[rows], cfg.data.val_fraction, cfg.data.test_fraction, numpy_rng(seed, "split", k))
        raw_clients.append(
            {"train": rows[tr], "val": rows[va], "test": rows[te], "zero_day": unseen_idx[unseen_parts[k]]}
        )

    # Federated scaler: each client contributes sufficient statistics of its training rows only.
    stats = [
        FeatureStatistics.from_array(values[rc["train"]], cfg.data.log_transform)
        for rc in raw_clients
        if len(rc["train"])
    ]
    scaler = FeatureScaler(cfg.data.scaler, cfg.data.log_transform, cfg.data.clip_value)
    scaler.fit_from_statistics(FeatureStatistics.merge_all(stats))

    def make_split(rows: np.ndarray) -> Split:
        if len(rows) == 0:
            return Split.empty(len(features))
        return Split(scaler.transform(values[rows]), labels[rows], devices[rows])

    clients = [
        ClientData(k, *(make_split(rc[name]) for name in ("train", "val", "test", "zero_day")))
        for k, rc in enumerate(raw_clients)
    ]
    all_rows = [np.concatenate([rc[n] for n in ("train", "val", "test", "zero_day")]) for rc in raw_clients]
    table = partition_summary(labels, all_rows)
    hetero = heterogeneity_index(table)
    for c in clients:
        n_benign = int((c.train.labels == cfg.data.benign_label).sum())
        LOGGER.info(
            "client_%d: train=%d (benign=%d) val=%d test=%d zero_day=%d devices=%d",
            c.client_id, len(c.train), n_benign, len(c.val), len(c.test), len(c.zero_day),
            len(set(c.train.devices.tolist())),
        )
    LOGGER.info("Partition strategy=%s alpha=%.3g heterogeneity (mean JS distance)=%.3f", pc.strategy, pc.dirichlet_alpha, hetero)
    metadata = {
        **{k: v for k, v in meta.items() if k in ("dataset", "num_rows", "class_counts", "cleaning")},
        "num_features": len(features),
        "heterogeneity_js": hetero,
        "known_attack_classes": list(zd.known_attack_classes),
        "unseen_attack_classes": list(zd.unseen_attack_classes),
    }
    return FederatedDataset(clients, features, zd, scaler, table, metadata)
