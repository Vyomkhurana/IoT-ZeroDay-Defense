"""Experiment directory layout and artefact persistence.

::

    results/experiments/<name>/
        config.yaml            fully-resolved configuration
        environment.json       package versions, hardware, git commit, seed
        metrics.json           final metrics (all models) + setup metadata
        metrics.csv            per-round (federated) / per-epoch (centralized) history
        per_client.csv         per-client evaluation of the final model
        client_distribution.csv
        privacy_ledger.csv     per-client cumulative epsilon after each round (DP runs)
        scores.npz             test anomaly scores / labels (subsampled) for plotting
        model/                 autoencoder weights, threshold, scaler, baselines
        figures/               per-experiment plots
        logs/                  train.log, events.jsonl
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from ppfl.utils.config import Config, save_config
from ppfl.utils.environment import collect_environment
from ppfl.utils.logging import get_logger
from ppfl.utils.serialization import load_json, save_json

LOGGER = get_logger(__name__)


class ExperimentTracker:
    def __init__(self, cfg: Config, clean: bool = True) -> None:
        self.cfg = cfg
        self.dir = cfg.experiment_dir
        if clean and self.dir.exists():
            shutil.rmtree(self.dir)
        for sub in ("model", "figures", "logs"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, Any]] = []
        self.started = datetime.now(timezone.utc)
        self.on_round: Callable[[dict[str, Any]], None] | None = None  # live progress hook (used by the UI)

    @property
    def logs_dir(self) -> Path:
        return self.dir / "logs"

    @property
    def model_dir(self) -> Path:
        return self.dir / "model"

    @property
    def figures_dir(self) -> Path:
        return self.dir / "figures"

    def save_setup(self) -> None:
        save_config(self.cfg, self.dir / "config.yaml")
        env = collect_environment()
        env["seed"] = self.cfg.experiment.seed
        env["config_fingerprint"] = self.cfg.fingerprint()
        env["started_utc"] = self.started.isoformat()
        save_json(env, self.dir / "environment.json")

    def log_round(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.dir / "metrics.csv", index=False)
        if self.on_round is not None:
            self.on_round(row)

    def save_table(self, df: pd.DataFrame, name: str, index: bool = False) -> None:
        df.to_csv(self.dir / f"{name}.csv", index=index)

    def save_metrics(self, metrics: dict[str, Any]) -> None:
        metrics.setdefault("experiment", {})["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(metrics, self.dir / "metrics.json")

    def save_model(self, model: torch.nn.Module, architecture: dict[str, Any], threshold: dict[str, Any], scaler: dict[str, Any]) -> None:
        torch.save(model.state_dict(), self.model_dir / "autoencoder.pt")
        save_json({"architecture": architecture, "threshold": threshold}, self.model_dir / "model.json")
        save_json(scaler, self.model_dir / "scaler.json")

    def save_scores(self, name: str, scores: np.ndarray, labels: np.ndarray, groups: np.ndarray, threshold: float) -> None:
        cap = self.cfg.evaluation.max_saved_scores
        idx = np.arange(len(scores))
        if len(idx) > cap:
            idx = np.sort(np.random.default_rng(0).choice(len(idx), size=cap, replace=False))
        np.savez_compressed(
            self.dir / f"scores_{name}.npz",
            scores=scores[idx],
            labels=labels[idx].astype(str),
            groups=groups[idx].astype(str),
            threshold=np.array(threshold),
        )


def load_experiment(path: Path) -> dict[str, Any]:
    """Load ``metrics.json`` (+ history) of a finished experiment."""
    metrics = load_json(path / "metrics.json")
    history = path / "metrics.csv"
    metrics["_history"] = pd.read_csv(history) if history.exists() else pd.DataFrame()
    metrics["_dir"] = path
    return metrics
