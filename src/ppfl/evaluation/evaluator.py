"""Evaluate any scorer on the global test set and on every client's local test set."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ppfl.data.federated_dataset import FederatedDataset
from ppfl.evaluation.zero_day_metrics import headline, zero_day_report

Scorer = Callable[[np.ndarray], np.ndarray]


@dataclass
class DetectionEvaluation:
    threshold: float
    scores: np.ndarray
    labels: np.ndarray
    client_ids: np.ndarray
    report: dict[str, Any]
    per_client: list[dict[str, Any]] = field(default_factory=list)

    @property
    def summary(self) -> dict[str, float]:
        return headline(self.report)


def evaluate_detector(
    scorer: Scorer, dataset: FederatedDataset, threshold: float, per_client: bool = True
) -> DetectionEvaluation:
    """Score the global test set once; derive global and per-client reports from it."""
    test, owners = dataset.global_test()
    scores = scorer(test.X)
    report = zero_day_report(scores, test.labels, threshold, dataset.zero_day)
    rows = []
    if per_client:
        for c in dataset.clients:
            m = owners == c.client_id
            if not m.any():
                continue
            r = zero_day_report(scores[m], test.labels[m], threshold, dataset.zero_day)
            rows.append({"client_id": c.client_id, "n_test": int(m.sum()), **headline(r), "benign_fpr": r["benign_fpr"]})
    return DetectionEvaluation(threshold, scores, test.labels, owners, report, rows)


def quick_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float, dataset: FederatedDataset) -> dict[str, float]:
    """Headline metrics only (used for per-round monitoring)."""
    return headline(zero_day_report(scores, labels, threshold, dataset.zero_day))
