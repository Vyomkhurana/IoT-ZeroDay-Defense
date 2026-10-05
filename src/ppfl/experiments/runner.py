"""Run one experiment end to end from a :class:`Config`."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import torch

from ppfl.data.federated_dataset import FederatedDataset, build_federated_dataset
from ppfl.experiments.centralized import run_centralized
from ppfl.experiments.common import ModeResult
from ppfl.experiments.federated import run_federated
from ppfl.experiments.local import run_local
from ppfl.experiments.tracking import ExperimentTracker
from ppfl.utils.config import Config
from ppfl.utils.environment import resolve_device
from ppfl.utils.logging import get_logger, setup_logging
from ppfl.utils.seed import set_global_seed

LOGGER = get_logger(__name__)

MODES = {"centralized": run_centralized, "federated": run_federated, "local": run_local}


def describe_mode(cfg: Config) -> str:
    """Human-readable label used in tables and plots."""
    mode = cfg.training.mode
    if mode == "centralized":
        return "Centralized"
    if mode == "local":
        return "Local-only"
    dp, sa = cfg.privacy.enabled, cfg.secure_aggregation.enabled
    base = "FedProx" if cfg.federated.strategy == "fedprox" else "FL"
    if dp and sa:
        return f"Full PPFL ({'FedProx' if base == 'FedProx' else 'FedAvg'})"
    if dp:
        return f"{base} + DP"
    if sa:
        return f"{base} + SecAgg"
    return "Plain FL (FedAvg)" if base == "FL" else "FedProx"


def experiment_setup(cfg: Config, dataset: FederatedDataset) -> dict[str, Any]:
    return {
        "dataset": cfg.data.dataset,
        "num_features": dataset.input_dim,
        "num_clients": cfg.partition.num_clients,
        "partition_strategy": cfg.partition.strategy,
        "dirichlet_alpha": cfg.partition.dirichlet_alpha,
        "heterogeneity_js": dataset.metadata.get("heterogeneity_js"),
        "known_attack_classes": list(dataset.zero_day.known_attack_classes),
        "unseen_attack_classes": list(dataset.zero_day.unseen_attack_classes),
        "train_contamination": cfg.data.train_contamination,
        "model": {"hidden_dims": cfg.model.hidden_dims, "latent_dim": cfg.model.latent_dim, "activation": cfg.model.activation},
        "optimizer": cfg.training.optimizer,
        "batch_size": cfg.training.batch_size,
        "learning_rate": cfg.training.learning_rate,
        "epochs": cfg.training.epochs if cfg.training.mode != "federated" else None,
        "rounds": cfg.federated.rounds if cfg.training.mode == "federated" else None,
        "local_epochs": cfg.federated.local_epochs if cfg.training.mode == "federated" else None,
        "strategy": cfg.federated.strategy if cfg.training.mode == "federated" else None,
        "fedprox_mu": cfg.federated.mu if cfg.federated.strategy == "fedprox" else None,
        "fraction_fit": cfg.federated.fraction_fit,
        "dp_enabled": cfg.privacy.enabled and cfg.training.mode == "federated",
        "noise_multiplier": cfg.privacy.noise_multiplier if cfg.privacy.enabled else None,
        "max_grad_norm": cfg.privacy.max_grad_norm if cfg.privacy.enabled else None,
        "delta": cfg.privacy.delta if cfg.privacy.enabled else None,
        "secure_aggregation": cfg.secure_aggregation.enabled and cfg.training.mode == "federated",
        "secagg_dropout_rate": cfg.secure_aggregation.dropout_rate if cfg.secure_aggregation.enabled else None,
        "threshold_strategy": cfg.detection.threshold_strategy,
        "threshold_percentile": cfg.detection.percentile,
        "seed": cfg.experiment.seed,
    }


def run_experiment(cfg: Config, frame: pd.DataFrame | None = None, dataset: FederatedDataset | None = None) -> dict[str, Any]:
    """Train, evaluate and persist one experiment; returns the metrics dictionary."""
    tracker = ExperimentTracker(cfg)
    setup_logging(cfg.logging.level, tracker.logs_dir)
    set_global_seed(cfg.experiment.seed, cfg.experiment.deterministic)
    if cfg.experiment.num_threads:
        torch.set_num_threads(cfg.experiment.num_threads)
    device = resolve_device(cfg.experiment.device)
    label = describe_mode(cfg)
    LOGGER.info("=== Experiment '%s' (%s) | dataset=%s | device=%s | seed=%d ===", cfg.experiment.name, label, cfg.data.dataset, device, cfg.experiment.seed)
    tracker.save_setup()

    start = time.perf_counter()
    if dataset is None:
        dataset = build_federated_dataset(cfg, frame)
    tracker.save_table(dataset.partition_table, "client_distribution", index=True)
    result: ModeResult = MODES[cfg.training.mode](cfg, dataset, tracker, device)
    total_time = time.perf_counter() - start

    for name, table in result.tables.items():
        tracker.save_table(table, name)
    ae = result.models["autoencoder"]
    if ae.per_client:
        tracker.save_table(pd.DataFrame(ae.per_client), "per_client")
    if result.final_model is not None:
        tracker.save_model(result.final_model, result.final_model.architecture(), ae.threshold, dataset.scaler.to_dict())
    if cfg.evaluation.save_scores:
        for name, ev in result.evaluations.items():
            tracker.save_scores(name, ev.scores, ev.labels, dataset.zero_day.group_of(ev.labels), ev.threshold)

    metrics: dict[str, Any] = {
        "experiment": {
            "name": cfg.experiment.name,
            "label": label,
            "mode": cfg.training.mode,
            "description": cfg.experiment.description,
            "config_fingerprint": cfg.fingerprint(),
            "started_utc": tracker.started.isoformat(),
            "synthetic_data": cfg.data.dataset == "synthetic",
        },
        "setup": experiment_setup(cfg, dataset),
        "models": {name: vars(m) for name, m in result.models.items()},
        "federated": result.federated,
        "communication": result.communication,
        "privacy": result.privacy or {"enabled": False},
        "timing": {**result.timing, "total_time_s": total_time},
        "data": dataset.metadata,
    }
    tracker.save_metrics(metrics)

    if cfg.evaluation.make_figures:
        from ppfl.visualization.plots import experiment_figures

        try:
            experiment_figures(tracker.dir)
        except Exception:  # figures must never invalidate a finished run
            LOGGER.exception("Figure generation failed for %s", tracker.dir)

    s = ae.summary
    LOGGER.info(
        "RESULT %s | F1 %.4f | precision %.4f | recall %.4f | FPR %.4f | UNSEEN-ATTACK RECALL %.4f | AUC %.4f%s | %.1fs",
        cfg.experiment.name, s["f1"], s["precision"], s["recall"], s["fpr"], s["unseen_recall"], s["roc_auc"],
        f" | eps {result.privacy['epsilon']:.3f}" if result.privacy.get("epsilon") is not None else "", total_time,
    )
    LOGGER.info("Artefacts written to %s (finished %s)", tracker.dir, datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return metrics
