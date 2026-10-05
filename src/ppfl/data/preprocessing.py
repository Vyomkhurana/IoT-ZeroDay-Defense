"""Cleaning and federated-compatible feature scaling.

Cleaning (offline, at preparation time) only takes *schema-level* decisions that do
not depend on the train/test split: coercing to numeric, turning ``±inf`` into
missing values, dropping columns that are mostly missing or constant, and dropping
rows that are mostly missing.

Scaling and imputation are *fitted on training data only*. Because raw data never
leaves the clients, the scaler is fitted from per-client sufficient statistics
(count, sum, sum of squares, min, max per feature) that are merged at the server.
Merging sufficient statistics is mathematically identical to fitting on the pooled
training data, so the centralized baseline and the federated modes see exactly the
same feature transformation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ppfl.data.loader import DEVICE_COL, LABEL_COL, RawDataset
from ppfl.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class CleaningReport:
    rows_in: int
    rows_out: int
    columns_in: int
    columns_out: int
    dropped_missing_columns: list[str]
    dropped_constant_columns: list[str]
    infinite_cells: int
    missing_cells_after: int

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def clean_dataset(
    raw: RawDataset,
    benign_label: str,
    max_missing_fraction: float = 0.5,
    max_row_missing_fraction: float = 0.5,
) -> tuple[pd.DataFrame, list[str], CleaningReport]:
    """Clean a raw dataset; returns ``(frame, feature_columns, report)``."""
    frame = raw.frame
    features = list(raw.feature_columns)
    values = frame[features].apply(pd.to_numeric, errors="coerce").astype(np.float64)
    inf_mask = np.isinf(values.to_numpy())
    n_inf = int(inf_mask.sum())
    values = values.mask(inf_mask)

    missing_frac = values.isna().mean()
    drop_missing = sorted(missing_frac[missing_frac > max_missing_fraction].index)
    values = values.drop(columns=drop_missing)
    nunique = values.nunique(dropna=True)
    drop_constant = sorted(nunique[nunique <= 1].index)
    values = values.drop(columns=drop_constant)

    row_ok = values.isna().mean(axis=1) <= max_row_missing_fraction
    values = values.loc[row_ok]
    meta = frame.loc[row_ok, [LABEL_COL, DEVICE_COL]].astype(str)
    if benign_label not in set(meta[LABEL_COL]):
        raise ValueError(f"No rows labelled {benign_label!r} after cleaning; check data.benign_label")

    out = pd.concat([values.astype(np.float32), meta], axis=1).reset_index(drop=True)
    report = CleaningReport(
        rows_in=len(frame),
        rows_out=len(out),
        columns_in=len(features),
        columns_out=values.shape[1],
        dropped_missing_columns=drop_missing,
        dropped_constant_columns=drop_constant,
        infinite_cells=n_inf,
        missing_cells_after=int(values.isna().to_numpy().sum()),
    )
    LOGGER.info(
        "Cleaning: rows %d -> %d, features %d -> %d (inf cells=%d, constant=%d, mostly-missing=%d)",
        report.rows_in, report.rows_out, report.columns_in, report.columns_out,
        n_inf, len(drop_constant), len(drop_missing),
    )
    return out, list(values.columns), report


def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


@dataclass
class FeatureStatistics:
    """Per-feature sufficient statistics computed locally by one client."""

    count: np.ndarray
    total: np.ndarray
    total_sq: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray

    @classmethod
    def from_array(cls, x: np.ndarray, log_transform: bool = False) -> FeatureStatistics:
        x = np.asarray(x, dtype=np.float64)
        if log_transform:
            x = signed_log1p(x)
        finite = np.isfinite(x)
        safe = np.where(finite, x, 0.0)
        has_any = finite.any(axis=0)
        return cls(
            count=finite.sum(axis=0).astype(np.float64),
            total=safe.sum(axis=0),
            total_sq=(safe**2).sum(axis=0),
            minimum=np.where(has_any, np.where(finite, x, np.inf).min(axis=0, initial=np.inf), np.inf),
            maximum=np.where(has_any, np.where(finite, x, -np.inf).max(axis=0, initial=-np.inf), -np.inf),
        )

    def merge(self, other: FeatureStatistics) -> FeatureStatistics:
        return FeatureStatistics(
            count=self.count + other.count,
            total=self.total + other.total,
            total_sq=self.total_sq + other.total_sq,
            minimum=np.minimum(self.minimum, other.minimum),
            maximum=np.maximum(self.maximum, other.maximum),
        )

    @staticmethod
    def merge_all(stats: list[FeatureStatistics]) -> FeatureStatistics:
        if not stats:
            raise ValueError("Cannot merge an empty list of statistics")
        out = stats[0]
        for s in stats[1:]:
            out = out.merge(s)
        return out


class FeatureScaler:
    """Standard / min-max scaler with mean imputation, fitted from sufficient statistics."""

    def __init__(self, method: str = "standard", log_transform: bool = False, clip_value: float | None = None):
        if method not in ("standard", "minmax"):
            raise ValueError(f"Unknown scaler {method!r}")
        self.method = method
        self.log_transform = log_transform
        self.clip_value = clip_value
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.offset_: np.ndarray | None = None

    def fit_from_statistics(self, stats: FeatureStatistics) -> FeatureScaler:
        count = np.maximum(stats.count, 1.0)
        mean = stats.total / count
        if self.method == "standard":
            var = np.maximum(stats.total_sq / count - mean**2, 0.0)
            scale = np.sqrt(var)
            offset = mean
        else:
            lo = np.where(np.isfinite(stats.minimum), stats.minimum, 0.0)
            hi = np.where(np.isfinite(stats.maximum), stats.maximum, 1.0)
            scale = hi - lo
            offset = lo
        self.mean_ = mean
        self.offset_ = offset
        self.scale_ = np.where(scale > 1e-12, scale, 1.0)
        return self

    def fit(self, x: np.ndarray) -> FeatureScaler:
        return self.fit_from_statistics(FeatureStatistics.from_array(x, self.log_transform))

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.offset_ is None:
            raise RuntimeError("FeatureScaler must be fitted before transform()")
        x = np.asarray(x, dtype=np.float64)
        if self.log_transform:
            x = signed_log1p(x)
        x = np.where(np.isfinite(x), x, self.mean_)  # mean imputation (training statistics)
        z = (x - self.offset_) / self.scale_
        if self.clip_value is not None:
            z = np.clip(z, -self.clip_value, self.clip_value)
        return z.astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "log_transform": self.log_transform,
            "clip_value": self.clip_value,
            "mean": self.mean_,
            "offset": self.offset_,
            "scale": self.scale_,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FeatureScaler:
        obj = cls(d["method"], d["log_transform"], d["clip_value"])
        obj.mean_ = np.asarray(d["mean"], dtype=np.float64)
        obj.offset_ = np.asarray(d["offset"], dtype=np.float64)
        obj.scale_ = np.asarray(d["scale"], dtype=np.float64)
        return obj
