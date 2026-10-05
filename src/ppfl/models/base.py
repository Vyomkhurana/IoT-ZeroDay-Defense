"""Common interface for the classical (non-federated) anomaly detection baselines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np


class AnomalyDetector(ABC):
    """Detector trained on benign data; ``score`` returns higher-is-more-anomalous values."""

    name: str = "detector"

    @abstractmethod
    def fit(self, X_benign: np.ndarray) -> AnomalyDetector: ...

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray: ...

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> AnomalyDetector:
        return joblib.load(path)


def subsample(X: np.ndarray, max_rows: int | None, rng: np.random.Generator) -> np.ndarray:
    if max_rows is None or len(X) <= max_rows:
        return X
    return X[rng.choice(len(X), size=max_rows, replace=False)]
