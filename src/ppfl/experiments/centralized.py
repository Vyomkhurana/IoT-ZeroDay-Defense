"""Mode 1 — centralized reference: all clients' training data pooled at one server.

Also trains the classical centralized baselines (Isolation Forest, One-Class SVM).
No federated learning, no DP, no secure aggregation; this is the upper/reference
baseline that ignores the privacy constraint.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ppfl.data.federated_dataset import FederatedDataset
from ppfl.evaluation.evaluator import evaluate_detector
from ppfl.experiments.common import EvalContext, ModelResult, ModeResult, init_model
from ppfl.experiments.tracking import ExperimentTracker
from ppfl.models.base import AnomalyDetector
from ppfl.models.isolation_forest import IsolationForestDetector
from ppfl.models.one_class_svm import OneClassSVMDetector
from ppfl.training.trainer import train_autoencoder
from ppfl.utils.config import Config
from ppfl.utils.logging import get_logger
from ppfl.utils.seed import derive_seed, numpy_rng, torch_generator
from ppfl.utils.serialization import payload_nbytes

LOGGER = get_logger(__name__)


def pooled_training_matrix(cfg: Config, dataset: FederatedDataset) -> np.ndarray:
    """Union of exactly the rows each federated client would train on."""
    return np.concatenate(
        [
            c.training_matrix(cfg.data.benign_label, cfg.data.train_contamination, numpy_rng(cfg.experiment.seed, "contamination", c.client_id))
            for c in dataset.clients
        ]
    )


def _run_baseline(detector: AnomalyDetector, X_train: np.ndarray, ctx: EvalContext, dataset: FederatedDataset) -> tuple[ModelResult, object, float]:
    start = time.perf_counter()
    detector.fit(X_train)
    fit_time = time.perf_counter() - start
    threshold = ctx.selector.from_scores(
        detector.score(np.concatenate(ctx.val_benign)), detector.score(np.concatenate(ctx.val_attack))
    )
    ev = evaluate_detector(detector.score, dataset, threshold.value, ctx.cfg.evaluation.per_client)
    LOGGER.info("%s: F1=%.4f unseen-recall=%.4f FPR=%.4f (fit %.1fs)", detector.name, ev.summary["f1"], ev.summary["unseen_recall"], ev.summary["fpr"], fit_time)
    return ModelResult.from_evaluation(ev, threshold), ev, fit_time


def run_centralized(cfg: Config, dataset: FederatedDataset, tracker: ExperimentTracker, device: torch.device) -> ModeResult:
    ctx = EvalContext(cfg, dataset, device)
    X = pooled_training_matrix(cfg, dataset)
    LOGGER.info("Centralized training on %d pooled rows for %d epochs", len(X), cfg.training.epochs)
    model = init_model(cfg, dataset.input_dim).to(device)
    epoch_start = [time.perf_counter()]

    def on_epoch(epoch: int, loss: float) -> None:
        row = {"round": epoch + 1, "train_loss": loss, "epoch_time_s": time.perf_counter() - epoch_start[0]}
        if (epoch + 1) % cfg.federated.eval_every == 0 or epoch + 1 == cfg.training.epochs:
            row.update(ctx.monitor(model, "exact"))
            LOGGER.info("epoch %3d | train %.5f | val %.5f | F1 %.4f | unseen recall %.4f", epoch + 1, loss, row["val_loss"], row["f1"], row["unseen_recall"])
        model.train()
        tracker.log_round(row)
        epoch_start[0] = time.perf_counter()

    result = train_autoencoder(
        model, X, epochs=cfg.training.epochs, cfg=cfg.training, device=device,
        generator=torch_generator(cfg.experiment.seed, "central-shuffle"), epoch_callback=on_epoch,
    )
    threshold = ctx.exact_threshold(model)
    ev = evaluate_detector(ctx.scorer(model), dataset, threshold.value, cfg.evaluation.per_client)
    models = {"autoencoder": ModelResult.from_evaluation(ev, threshold)}
    evaluations = {"autoencoder": ev}
    timing = {"train_time_s": result.wall_time_s}

    b = cfg.baselines
    seed = derive_seed(cfg.experiment.seed, "baseline")
    X_benign = np.concatenate([c.train.X[c.train.labels == cfg.data.benign_label] for c in dataset.clients])
    for enabled, detector in (
        (b.isolation_forest.enabled, IsolationForestDetector(b.isolation_forest, seed)),
        (b.one_class_svm.enabled, OneClassSVMDetector(b.one_class_svm, seed)),
    ):
        if enabled:
            models[detector.name], evaluations[detector.name], timing[f"{detector.name}_fit_s"] = _run_baseline(detector, X_benign, ctx, dataset)
            detector.save(tracker.model_dir / f"{detector.name}.joblib")

    # What a centralized IDS must collect: every client's training and validation traffic.
    raw_upload = sum(payload_nbytes(c.train.X) + payload_nbytes(c.val.X) for c in dataset.clients)
    return ModeResult(
        models=models,
        evaluations=evaluations,
        final_model=model,
        communication={"raw_data_upload_bytes": raw_upload, "total_bytes": raw_upload},
        timing=timing,
    )
