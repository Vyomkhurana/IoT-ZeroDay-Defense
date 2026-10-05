"""Local-only baseline: every gateway trains its own autoencoder and never collaborates.

Quantifies the benefit of federation: a gateway that has seen little benign traffic
(or traffic from few device types) is expected to generalise worse than the
federated model. Each local model is evaluated on the global test set; reported
metrics are the mean (and std) over clients.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from ppfl.data.federated_dataset import FederatedDataset
from ppfl.detection.threshold import ThresholdSelector
from ppfl.evaluation.evaluator import evaluate_detector
from ppfl.evaluation.zero_day_metrics import headline, zero_day_report
from ppfl.experiments.common import EvalContext, ModelResult, ModeResult, init_model, threshold_to_dict
from ppfl.experiments.tracking import ExperimentTracker
from ppfl.training.trainer import train_autoencoder
from ppfl.utils.config import Config
from ppfl.utils.logging import get_logger
from ppfl.utils.seed import numpy_rng, torch_generator

LOGGER = get_logger(__name__)


def run_local(cfg: Config, dataset: FederatedDataset, tracker: ExperimentTracker, device: torch.device) -> ModeResult:
    ctx = EvalContext(cfg, dataset, device)
    selector = ThresholdSelector(cfg.detection)
    rows, summaries = [], []
    start = time.perf_counter()
    for c in dataset.clients:
        X = c.training_matrix(cfg.data.benign_label, cfg.data.train_contamination, numpy_rng(cfg.experiment.seed, "contamination", c.client_id))
        model = init_model(cfg, dataset.input_dim).to(device)
        result = train_autoencoder(
            model, X, epochs=cfg.training.epochs, cfg=cfg.training, device=device,
            generator=torch_generator(cfg.experiment.seed, "local-shuffle", c.client_id),
        )
        score = ctx.scorer(model)
        benign_val = c.val.X[c.val.labels == cfg.data.benign_label]
        attack_val = c.val.X[c.val.labels != cfg.data.benign_label]
        threshold = selector.from_scores(score(benign_val), score(attack_val))
        ev = evaluate_detector(score, dataset, threshold.value, per_client=False)
        local = c.local_test()
        own = headline(zero_day_report(score(local.X), local.labels, threshold.value, dataset.zero_day))
        summaries.append(ev.summary)
        rows.append(
            {"client_id": c.client_id, "n_train": len(X), "train_time_s": result.wall_time_s, "final_train_loss": result.final_loss,
             "threshold": threshold.value, **{f"global_{k}": v for k, v in ev.summary.items()}, **{f"own_{k}": v for k, v in own.items()}}
        )
        tracker.log_round({"round": c.client_id, **rows[-1]})
        LOGGER.info("client_%d local model: global F1 %.4f | unseen recall %.4f | own-test F1 %.4f", c.client_id, ev.summary["f1"], ev.summary["unseen_recall"], own["f1"])
    frame = pd.DataFrame(summaries)
    per_client = [{"client_id": r["client_id"], **{k[4:]: v for k, v in r.items() if k.startswith("own_")}} for r in rows]
    model_result = ModelResult(
        summary=frame.mean(numeric_only=True).to_dict(),
        summary_std=frame.std(numeric_only=True).to_dict(),
        threshold=threshold_to_dict(threshold) | {"note": "per-client thresholds; value shown is the last client's"},
        per_client=per_client,
    )
    return ModeResult(
        models={"autoencoder": model_result},
        communication={"total_bytes": 0},
        timing={"train_time_s": time.perf_counter() - start, "mean_client_compute_s": float(np.mean([r["train_time_s"] for r in rows]))},
        tables={"local_models": pd.DataFrame(rows)},
    )
