"""One-Class SVM baseline (Schölkopf et al., 2001) via scikit-learn.

Kernel OC-SVM training scales super-linearly with the number of samples, so it is
fitted on a random subsample (``baselines.one_class_svm.max_train_samples``) of the
pooled benign training data. Like Isolation Forest, it is a centralized baseline and
is not federated.
"""

from __future__ import annotations

import numpy as np
from sklearn.svm import OneClassSVM

from ppfl.models.base import AnomalyDetector, subsample
from ppfl.utils.config import OneClassSVMConfig


class OneClassSVMDetector(AnomalyDetector):
    name = "one_class_svm"

    def __init__(self, cfg: OneClassSVMConfig, seed: int) -> None:
        self.cfg = cfg
        self.seed = seed
        self.model = OneClassSVM(kernel=cfg.kernel, nu=cfg.nu, gamma=cfg.gamma)

    def fit(self, X_benign: np.ndarray) -> OneClassSVMDetector:
        rng = np.random.default_rng(self.seed)
        self.model.fit(subsample(X_benign, self.cfg.max_train_samples, rng))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        # decision_function is positive inside the learned support; negate for anomaly score.
        return -self.model.decision_function(X)
