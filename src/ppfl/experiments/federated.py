"""Modes 2-5 — federated training (plain FL, FL+DP, FL+SecAgg, full PPFL).

The four federated modes share this single code path; they differ only in
``privacy.enabled`` and ``secure_aggregation.enabled`` (and the strategy).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from ppfl.data.federated_dataset import FederatedDataset
from ppfl.evaluation.evaluator import evaluate_detector
from ppfl.experiments.common import EvalContext, ModelResult, ModeResult, init_model
from ppfl.experiments.tracking import ExperimentTracker
from ppfl.federated.client import IoTClient
from ppfl.federated.fedprox import build_strategy
from ppfl.federated.server import FederatedServer
from ppfl.privacy.accountant import PrivacyLedger
from ppfl.utils.config import Config
from ppfl.utils.logging import get_logger

LOGGER = get_logger(__name__)


def run_federated(cfg: Config, dataset: FederatedDataset, tracker: ExperimentTracker, device: torch.device) -> ModeResult:
    f, pr = cfg.federated, cfg.privacy
    ctx = EvalContext(cfg, dataset, device)
    clients = [IoTClient(c, cfg, dataset.input_dim, device) for c in dataset.clients]
    strategy = build_strategy(f.strategy, f.mu, f.server_learning_rate)
    server = FederatedServer(init_model(cfg, dataset.input_dim), strategy, cfg)
    ledger = PrivacyLedger(pr.delta)
    LOGGER.info(
        "Federated training: %d clients, %d rounds x %d local epochs, strategy=%s%s, DP=%s, SecAgg=%s",
        len(clients), f.rounds, f.local_epochs, f.strategy,
        f"(mu={f.mu})" if f.strategy == "fedprox" else "", pr.enabled, cfg.secure_aggregation.enabled,
    )

    client_rows: list[dict] = []
    totals = {"upload": 0, "download": 0, "secagg": 0}
    train_start = time.perf_counter()
    last_comm = None
    for r in range(1, f.rounds + 1):
        summary = server.run_round(r, clients)
        last_comm = summary.comm
        for s in summary.client_stats:
            client_rows.append(
                {"round": r, "client_id": s.client_id, "num_samples": s.num_samples, "train_loss": s.train_loss,
                 "compute_time_s": s.compute_time_s, "steps": s.steps, "epsilon": s.epsilon,
                 "dropped": s.client_id in summary.dropped}
            )
            if pr.enabled:
                eng = clients[s.client_id].dp_engine
                ledger.record(r, s.client_id, float(s.epsilon), eng.steps, eng.noise_multiplier, eng.sample_rate)
        totals["upload"] += summary.comm.upload_bytes
        totals["download"] += summary.comm.download_bytes
        totals["secagg"] += summary.comm.secagg_overhead_bytes
        times = [s.compute_time_s for s in summary.client_stats]
        row = {
            "round": r,
            "train_loss": summary.aggregate.mean_loss,
            "participants": len(summary.participants),
            "dropped": len(summary.dropped),
            "upload_bytes": summary.comm.upload_bytes,
            "download_bytes": summary.comm.download_bytes,
            "secagg_overhead_bytes": summary.comm.secagg_overhead_bytes,
            "cumulative_bytes": totals["upload"] + totals["download"],
            "round_time_s": summary.wall_time_s,
            "client_time_mean_s": float(np.mean(times)),
            "client_time_max_s": float(np.max(times)),
        }
        if pr.enabled:
            eps = [c.epsilon() for c in clients if c.dp_engine and c.dp_engine.steps > 0]
            row.update({"epsilon_max": max(eps), "epsilon_mean": float(np.mean(eps))})
        if r % f.eval_every == 0 or r == f.rounds:
            row.update(ctx.monitor(server.global_model(dataset.input_dim).to(device), "histogram"))
            LOGGER.info(
                "round %3d | train %.5f | val %.5f | F1 %.4f | unseen recall %.4f | FPR %.4f%s | %.1f KB",
                r, row["train_loss"], row["val_loss"], row["f1"], row["unseen_recall"], row["fpr"],
                f" | eps {row['epsilon_max']:.3f}" if pr.enabled else "", (row["upload_bytes"] + row["download_bytes"]) / 1024,
            )
        tracker.log_round(row)
    train_time = time.perf_counter() - train_start

    threshold, calib_comm = server.calibrate_threshold(clients, ctx.selector, f.rounds + 1, cfg.detection.histogram_dp_epsilon)
    model = server.global_model(dataset.input_dim).to(device)
    ev = evaluate_detector(ctx.scorer(model), dataset, threshold.value, cfg.evaluation.per_client)
    LOGGER.info("Federated threshold (%s, histogram): %.6g", threshold.strategy, threshold.value)

    total_bytes = totals["upload"] + totals["download"]
    federated = {
        "strategy": f.strategy,
        "rounds": f.rounds,
        "local_epochs": f.local_epochs,
        "num_clients": len(clients),
        "clients_per_round": int(round(np.mean([len(set(g["client_id"])) for _, g in pd.DataFrame(client_rows).groupby("round")]))),
        "model_parameters": int(model.num_parameters()),
        "update_payload_bytes": last_comm.update_payload_bytes if last_comm else 0,
        "broadcast_payload_bytes": server.broadcast_nbytes(),
        "total_upload_bytes": totals["upload"],
        "total_download_bytes": totals["download"],
        "secagg_overhead_bytes": totals["secagg"],
        "threshold_calibration_bytes": calib_comm.total_bytes,
        "total_dropped_updates": int(sum(r["dropped"] for r in client_rows)),
    }
    client_df = pd.DataFrame(client_rows)
    privacy = {"enabled": pr.enabled}
    if pr.enabled:
        final_eps = {c.client_id: c.epsilon() for c in clients if c.dp_engine}
        privacy.update(
            {
                "epsilon": max(final_eps.values()),  # worst-case client guarantee
                "epsilon_mean": float(np.mean(list(final_eps.values()))),
                "epsilon_per_client": final_eps,
                "delta": pr.delta,
                "noise_multiplier": float(np.mean([c.dp_engine.noise_multiplier for c in clients if c.dp_engine])),
                "noise_multiplier_per_client": {c.client_id: c.dp_engine.noise_multiplier for c in clients if c.dp_engine},
                "max_grad_norm": pr.max_grad_norm,
                "accountant": pr.accountant,
                "target_epsilon": pr.target_epsilon,
                "sample_rate_per_client": {c.client_id: c.dp_engine.sample_rate for c in clients if c.dp_engine},
                "guarantee": "example-level (epsilon, delta)-DP per client local dataset; epsilon = max over clients",
            }
        )
    if cfg.detection.histogram_dp_epsilon is not None:
        privacy["threshold_histogram_epsilon"] = cfg.detection.histogram_dp_epsilon
    return ModeResult(
        models={"autoencoder": ModelResult.from_evaluation(ev, threshold)},
        evaluations={"autoencoder": ev},
        final_model=model,
        federated=federated,
        communication={"total_bytes": total_bytes + calib_comm.total_bytes, "training_bytes": total_bytes},
        privacy=privacy,
        timing={
            "train_time_s": train_time,
            "mean_client_compute_s": float(client_df["compute_time_s"].mean()),
            "max_client_compute_s": float(client_df["compute_time_s"].max()),
            "mean_round_time_s": float(pd.DataFrame(tracker.rows)["round_time_s"].mean()),
        },
        tables={"client_rounds": client_df, **({"privacy_ledger": ledger.to_frame()} if pr.enabled else {})},
    )
