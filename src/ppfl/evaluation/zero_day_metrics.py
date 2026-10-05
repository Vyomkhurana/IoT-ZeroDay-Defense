"""Security-oriented metrics that separate known from unseen (held-out) attacks.

Headline metric — **unseen attack recall**::

    unseen_attack_recall = #(held-out attack samples flagged anomalous) / #(held-out attack samples)

i.e. the fraction of zero-day-like traffic the detector catches without ever having
seen that attack class.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ppfl.data.zero_day import ZeroDaySplit
from ppfl.evaluation.metrics import binary_metrics


def _recall(pred: np.ndarray, mask: np.ndarray) -> float:
    return float(pred[mask].mean()) if mask.any() else float("nan")


def _score_stats(scores: np.ndarray) -> dict[str, float]:
    if len(scores) == 0:
        return {"n": 0}
    return {
        "n": int(len(scores)),
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
    }


def zero_day_report(scores: np.ndarray, labels: np.ndarray, threshold: float, split: ZeroDaySplit) -> dict[str, Any]:
    """Full detection report for one model on one evaluation set.

    Sections: ``overall`` (benign vs all attacks), ``known`` (benign vs known attacks),
    ``unseen`` (benign vs held-out attacks), headline recalls, per-class rates and
    anomaly-score statistics per group.
    """
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels).astype(str)
    pred = scores > threshold
    benign = split.is_benign(labels)
    known = split.is_known_attack(labels)
    unseen = split.is_unseen_attack(labels)
    attack = ~benign

    def subset(mask: np.ndarray) -> dict[str, float]:
        return binary_metrics(attack[mask], pred[mask], scores[mask])

    per_class = {}
    for label in sorted(set(labels.tolist())):
        m = labels == label
        group = "benign" if label == split.benign_label else ("unseen_attack" if label in split.unseen_attack_classes else "known_attack")
        per_class[label] = {
            "group": group,
            "n": int(m.sum()),
            "flagged_rate": float(pred[m].mean()),  # detection rate for attacks, FPR for benign
            "mean_score": float(scores[m].mean()),
            "median_score": float(np.median(scores[m])),
        }
    return {
        "threshold": float(threshold),
        "overall": subset(np.ones_like(benign)),
        "known": subset(benign | known),
        "unseen": subset(benign | unseen),
        "unseen_attack_recall": _recall(pred, unseen),
        "known_attack_recall": _recall(pred, known),
        "benign_fpr": _recall(pred, benign),
        "per_class": per_class,
        "score_stats": {
            "benign": _score_stats(scores[benign]),
            "known_attack": _score_stats(scores[known]),
            "unseen_attack": _score_stats(scores[unseen]),
        },
    }


def headline(report: dict[str, Any]) -> dict[str, float]:
    """Flat summary used in tables and per-round logs."""
    o = report["overall"]
    return {
        "accuracy": o["accuracy"],
        "precision": o["precision"],
        "recall": o["recall"],
        "f1": o["f1"],
        "fpr": o["fpr"],
        "roc_auc": o.get("roc_auc", float("nan")),
        "pr_auc": o.get("pr_auc", float("nan")),
        "unseen_recall": report["unseen_attack_recall"],
        "known_recall": report["known_attack_recall"],
        "unseen_roc_auc": report["unseen"].get("roc_auc", float("nan")),
        "threshold": report["threshold"],
    }
