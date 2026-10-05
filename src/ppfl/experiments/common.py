"""Shared pieces of the experiment modes: model initialisation, monitoring, result containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch

from ppfl.data.federated_dataset import FederatedDataset
from ppfl.detection.anomaly_score import AutoencoderScorer, reconstruction_errors
from ppfl.detection.threshold import ThresholdResult, ThresholdSelector
from ppfl.evaluation.evaluator import DetectionEvaluation, quick_metrics
from ppfl.models.autoencoder import Autoencoder
from ppfl.utils.config import Config
from ppfl.utils.seed import derive_seed


def init_model(cfg: Config, input_dim: int) -> Autoencoder:
    """Deterministic initial weights, identical across modes for a given seed."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(derive_seed(cfg.experiment.seed, "model-init"))
        return Autoencoder.from_config(input_dim, cfg.model)


def threshold_to_dict(t: ThresholdResult) -> dict[str, Any]:
    return {"value": t.value, "strategy": t.strategy, **t.details}


@dataclass
class ModelResult:
    """Everything recorded for one detector in one experiment."""

    summary: dict[str, float]
    threshold: dict[str, Any]
    report: dict[str, Any] | None = None
    per_client: list[dict[str, Any]] = field(default_factory=list)
    summary_std: dict[str, float] | None = None

    @classmethod
    def from_evaluation(cls, ev: DetectionEvaluation, threshold: ThresholdResult) -> ModelResult:
        return cls(ev.summary, threshold_to_dict(threshold), ev.report, ev.per_client)


@dataclass
class ModeResult:
    models: dict[str, ModelResult]
    evaluations: dict[str, DetectionEvaluation] = field(default_factory=dict)
    final_model: Autoencoder | None = None
    federated: dict[str, Any] | None = None
    communication: dict[str, Any] = field(default_factory=dict)
    privacy: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)


class EvalContext:
    """Caches evaluation data and computes per-round monitoring metrics.

    Monitoring is an *experimenter-side* diagnostic (the simulation has access to every
    client's validation/test split). Nothing computed here feeds back into training or
    model selection: the final model is always the last one and its threshold is
    computed by the mode's own protocol (exact or federated histograms).
    """

    def __init__(self, cfg: Config, dataset: FederatedDataset, device: torch.device) -> None:
        self.cfg = cfg
        self.dataset = dataset
        self.device = device
        self.selector = ThresholdSelector(cfg.detection)
        benign = cfg.data.benign_label
        self.val_benign = [c.val.X[c.val.labels == benign] for c in dataset.clients]
        self.val_attack = [c.val.X[c.val.labels != benign] for c in dataset.clients]
        test, self.test_owners = dataset.global_test()
        self.test_X, self.test_labels = test.X, test.labels

    def scorer(self, model: torch.nn.Module) -> AutoencoderScorer:
        return AutoencoderScorer(model, self.cfg.evaluation.batch_size, self.device)

    def validation_scores(self, model: torch.nn.Module) -> tuple[list[np.ndarray], list[np.ndarray]]:
        score = self.scorer(model)
        return [score(x) for x in self.val_benign], [score(x) for x in self.val_attack]

    def exact_threshold(self, model: torch.nn.Module) -> ThresholdResult:
        benign, attack = self.validation_scores(model)
        return self.selector.from_scores(np.concatenate(benign), np.concatenate(attack))

    def histogram_threshold(self, model: torch.nn.Module) -> ThresholdResult:
        benign, attack = self.validation_scores(model)
        total = np.sum([self.selector.histograms(b, a) for b, a in zip(benign, attack)], axis=0)
        return self.selector.from_histogram_vector(total)

    def monitor(self, model: torch.nn.Module, method: str) -> dict[str, float]:
        benign, attack = self.validation_scores(model)
        pooled_benign = np.concatenate(benign)
        if method == "exact":
            thr = self.selector.from_scores(pooled_benign, np.concatenate(attack))
        else:
            thr = self.selector.from_histogram_vector(
                np.sum([self.selector.histograms(b, a) for b, a in zip(benign, attack)], axis=0)
            )
        scores = reconstruction_errors(model, self.test_X, self.cfg.evaluation.batch_size, self.device)
        metrics = quick_metrics(scores, self.test_labels, thr.value, self.dataset)
        return {"val_loss": float(pooled_benign.mean()), **metrics}
