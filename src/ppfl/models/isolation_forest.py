"""Isolation Forest baseline (Liu et al., 2008) via scikit-learn.

Trained centrally on pooled benign training data. Tree ensembles have no natural
FedAvg-style parameter averaging, so this baseline is intentionally *not* federated;
it serves as a lightweight centralized reference point.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from ppfl.models.base import AnomalyDetector, subsample
from ppfl.utils.config import IsolationForestConfig


class IsolationForestDetector(AnomalyDetector):
    name = "isolation_forest"

    def __init__(self, cfg: IsolationForestConfig, seed: int) -> None:
        self.cfg = cfg
        self.seed = seed
        self.model = IsolationForest(
            n_estimators=cfg.n_estimators,
            max_samples=cfg.max_samples,
            contamination="auto",
            random_state=seed % (2**32),
            n_jobs=-1,
        )

    def fit(self, X_benign: np.ndarray) -> IsolationForestDetector:
        rng = np.random.default_rng(self.seed)
        self.model.fit(subsample(X_benign, self.cfg.max_train_samples, rng))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        # score_samples is higher for normal points; negate so higher = more anomalous.
        return -self.model.score_samples(X)
