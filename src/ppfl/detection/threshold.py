"""Anomaly-threshold selection.

A sample is flagged as anomalous when ``score > threshold``. Thresholds are always
derived from *validation* data (benign rows, plus known-attack rows for the
``validation_f1`` strategy) — never from test data and never from held-out
(zero-day) classes.

Strategies
----------
``percentile``     q-th percentile of benign validation scores (default q = 95, i.e. a
                   target false-positive rate of about 5 %).
``mean_std``       mean + k * std of benign validation scores.
``validation_f1``  threshold maximising F1 on validation benign vs *known* attacks.

Two computation paths are provided:

* exact, from raw score arrays (centralized and local-only training);
* from **histograms** on a shared log-spaced grid (federated training). Each client
  only reports bin counts of its validation scores; the counts are summed (securely,
  when secure aggregation is enabled), so no per-sample score leaves a client. With
  ~4k bins over 14 decades, bin width is < 1 % of the score, which bounds the
  approximation error of the histogram path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ppfl.utils.config import DetectionConfig


@dataclass
class ThresholdResult:
    value: float
    strategy: str
    details: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------- exact


def percentile_threshold(benign_scores: np.ndarray, percentile: float) -> float:
    if len(benign_scores) == 0:
        raise ValueError("Cannot compute a percentile threshold without benign validation scores")
    return float(np.percentile(benign_scores, percentile))


def mean_std_threshold(benign_scores: np.ndarray, k: float) -> float:
    if len(benign_scores) == 0:
        raise ValueError("Cannot compute a mean/std threshold without benign validation scores")
    return float(np.mean(benign_scores) + k * np.std(benign_scores))


def f1_optimal_threshold(benign_scores: np.ndarray, attack_scores: np.ndarray) -> tuple[float, float]:
    """Exact F1-maximising threshold for the rule ``score > threshold``."""
    if len(benign_scores) == 0 or len(attack_scores) == 0:
        raise ValueError("validation_f1 requires both benign and attack validation scores")
    s = np.concatenate([benign_scores, attack_scores])
    y = np.concatenate([np.zeros(len(benign_scores)), np.ones(len(attack_scores))])
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    last_of_tie = np.r_[s[1:] != s[:-1], True]  # only cut between distinct scores
    f1 = np.where(last_of_tie, 2 * tp / (2 * tp + fp + (y.sum() - tp)), -1.0)
    i = int(np.argmax(f1))
    thr = 0.5 * (s[i] + s[i + 1]) if i + 1 < len(s) else float(np.nextafter(s[i], -np.inf))
    return float(thr), float(f1[i])


# ------------------------------------------------------------------------- histogram


def log_edges(lo: float, hi: float, bins: int) -> np.ndarray:
    return np.logspace(np.log10(lo), np.log10(hi), bins + 1)


def histogram_counts(scores: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin counts; out-of-range scores are clamped into the first/last bin."""
    idx = np.searchsorted(edges, np.asarray(scores, dtype=np.float64), side="right") - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    return np.bincount(idx, minlength=len(edges) - 1).astype(np.float64)


def histogram_quantile(counts: np.ndarray, edges: np.ndarray, percentile: float) -> float:
    counts = np.clip(counts, 0, None)
    total = counts.sum()
    if total <= 0:
        raise ValueError("Empty histogram")
    target = percentile / 100.0 * total
    cdf = np.cumsum(counts)
    b = int(np.searchsorted(cdf, target, side="left"))
    b = min(b, len(counts) - 1)
    below = cdf[b - 1] if b > 0 else 0.0
    frac = 0.0 if counts[b] == 0 else (target - below) / counts[b]
    lo, hi = np.log(edges[b]), np.log(edges[b + 1])
    return float(np.exp(lo + np.clip(frac, 0.0, 1.0) * (hi - lo)))


def histogram_mean_std(counts: np.ndarray, edges: np.ndarray) -> tuple[float, float]:
    counts = np.clip(counts, 0, None)
    centers = np.sqrt(edges[:-1] * edges[1:])
    total = counts.sum()
    if total <= 0:
        raise ValueError("Empty histogram")
    mean = float((counts * centers).sum() / total)
    var = float((counts * (centers - mean) ** 2).sum() / total)
    return mean, float(np.sqrt(var))


def histogram_f1_optimal(benign: np.ndarray, attack: np.ndarray, edges: np.ndarray) -> tuple[float, float]:
    benign, attack = np.clip(benign, 0, None), np.clip(attack, 0, None)
    if benign.sum() <= 0 or attack.sum() <= 0:
        raise ValueError("validation_f1 requires both benign and attack histogram mass")
    # Predicted positive at edge j: all bins >= j.
    tp = np.r_[np.cumsum(attack[::-1])[::-1], 0.0]
    fp = np.r_[np.cumsum(benign[::-1])[::-1], 0.0]
    fn = attack.sum() - tp
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    j = int(np.argmax(f1))
    return float(edges[min(j, len(edges) - 1)]), float(f1[j])


# -------------------------------------------------------------------------- selector


class ThresholdSelector:
    """Applies the configured strategy to either raw scores or aggregated histograms."""

    def __init__(self, cfg: DetectionConfig) -> None:
        self.cfg = cfg
        self.edges = log_edges(cfg.histogram_min, cfg.histogram_max, cfg.histogram_bins)

    @property
    def num_bins(self) -> int:
        return len(self.edges) - 1

    def _fallback(self, attack_available: bool) -> str:
        strategy = self.cfg.threshold_strategy
        if strategy == "validation_f1" and not attack_available:
            return "percentile"  # no known attacks in validation data
        return strategy

    def from_scores(self, benign_scores: np.ndarray, attack_scores: np.ndarray | None = None) -> ThresholdResult:
        attack_scores = np.zeros(0) if attack_scores is None else np.asarray(attack_scores)
        strategy = self._fallback(len(attack_scores) > 0)
        details: dict[str, Any] = {"n_benign": int(len(benign_scores)), "n_attack": int(len(attack_scores)), "method": "exact"}
        if strategy == "percentile":
            value = percentile_threshold(benign_scores, self.cfg.percentile)
            details["percentile"] = self.cfg.percentile
        elif strategy == "mean_std":
            value = mean_std_threshold(benign_scores, self.cfg.std_factor)
            details["std_factor"] = self.cfg.std_factor
        else:
            value, f1 = f1_optimal_threshold(benign_scores, attack_scores)
            details["validation_f1"] = f1
        return ThresholdResult(value, strategy, details)

    def histograms(self, benign_scores: np.ndarray, attack_scores: np.ndarray) -> np.ndarray:
        """Client-side: concatenated [benign | attack] histogram vector."""
        return np.concatenate([histogram_counts(benign_scores, self.edges), histogram_counts(attack_scores, self.edges)])

    def from_histogram_vector(self, vector: np.ndarray) -> ThresholdResult:
        """Server-side: threshold from the (aggregated) [benign | attack] histogram vector."""
        benign, attack = vector[: self.num_bins], vector[self.num_bins :]
        strategy = self._fallback(float(np.clip(attack, 0, None).sum()) >= 1.0)
        details: dict[str, Any] = {
            "n_benign": float(np.clip(benign, 0, None).sum()),
            "n_attack": float(np.clip(attack, 0, None).sum()),
            "method": "histogram",
            "bins": self.num_bins,
        }
        if strategy == "percentile":
            value = histogram_quantile(benign, self.edges, self.cfg.percentile)
            details["percentile"] = self.cfg.percentile
        elif strategy == "mean_std":
            mean, std = histogram_mean_std(benign, self.edges)
            value = mean + self.cfg.std_factor * std
            details["std_factor"] = self.cfg.std_factor
        else:
            value, f1 = histogram_f1_optimal(benign, attack, self.edges)
            details["validation_f1"] = f1
        return ThresholdResult(float(value), strategy, details)
