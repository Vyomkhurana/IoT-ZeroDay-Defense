"""Collect finished experiments into one comparison table and a Markdown report."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ppfl.utils.serialization import load_json

_MODEL_NAMES = {"autoencoder": "Autoencoder", "isolation_forest": "Isolation Forest", "one_class_svm": "One-Class SVM"}


def _row(name: str, metrics: dict[str, Any], model: str, result: dict[str, Any]) -> dict[str, Any]:
    exp, setup = metrics["experiment"], metrics.get("setup", {})
    priv = metrics.get("privacy") or {}
    comm = metrics.get("communication") or {}
    fed = metrics.get("federated") or {}
    timing = metrics.get("timing") or {}
    is_ae = model == "autoencoder"
    label = exp["label"] if is_ae else f"{_MODEL_NAMES.get(model, model)} (centralized)"
    return {
        "experiment": name,
        "label": label,
        "mode": exp["mode"] if is_ae else "centralized",
        "model": model,
        "dataset": setup.get("dataset"),
        "synthetic": bool(exp.get("synthetic_data")),
        **{k: v for k, v in result["summary"].items()},
        "epsilon": priv.get("epsilon") if is_ae else None,
        "delta": priv.get("delta") if is_ae else None,
        "noise_multiplier": priv.get("noise_multiplier") if is_ae else None,
        "dp": bool(setup.get("dp_enabled")) and is_ae,
        "secagg": bool(setup.get("secure_aggregation")) and is_ae,
        "total_mb": (comm.get("total_bytes") or 0) / 2**20 if is_ae else None,
        "update_kb": (fed.get("update_payload_bytes") or 0) / 1024 if is_ae and fed else None,
        "secagg_overhead_mb": (fed.get("secagg_overhead_bytes") or 0) / 2**20 if is_ae and fed else None,
        "train_time_s": timing.get("train_time_s") if is_ae else timing.get(f"{model}_fit_s"),
        "mean_client_compute_s": timing.get("mean_client_compute_s") if is_ae else None,
        "rounds": setup.get("rounds"),
        "local_epochs": setup.get("local_epochs"),
        "num_clients": setup.get("num_clients"),
        "dirichlet_alpha": setup.get("dirichlet_alpha"),
        "strategy": setup.get("strategy"),
        "seed": setup.get("seed"),
    }


def collect_experiments(root: Path, names: list[str] | None = None) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """One row per (experiment, model); also returns each experiment's round history."""
    rows, histories = [], {}
    dirs = [root / n for n in names] if names else sorted(p for p in root.iterdir() if p.is_dir())
    for d in dirs:
        if not (d / "metrics.json").exists():
            continue
        metrics = load_json(d / "metrics.json")
        for model, result in metrics.get("models", {}).items():
            rows.append(_row(d.name, metrics, model, result))
        if (d / "metrics.csv").exists():
            histories[d.name] = pd.read_csv(d / "metrics.csv")
    df = pd.DataFrame(rows)
    if not df.empty:
        dup = df["label"].duplicated(keep=False)
        df["display"] = df["label"].where(~dup, df["label"] + " [" + df["experiment"] + "]")
    return df, histories


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def markdown_table(df: pd.DataFrame, columns: dict[str, str], digits: int = 4) -> str:
    cols = [c for c in columns if c in df.columns]
    lines = ["| " + " | ".join(columns[c] for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c], digits) for c in cols) + " |")
    return "\n".join(lines)


def write_report(df: pd.DataFrame, out_path: Path, figures: list[Path]) -> Path:
    """Write ``results/report.md`` from measured results only."""
    synthetic = bool(df["synthetic"].any()) if not df.empty else False
    parts = [
        "# PPFL-IoT-ZeroDay — experiment report",
        f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} from `results/experiments/*/metrics.json`._",
        "",
    ]
    if synthetic:
        parts += [
            "> **Warning:** some or all experiments below were run on the *synthetic* test dataset. "
            "Those numbers only demonstrate that the pipeline runs and must not be reported as research results.",
            "",
        ]
    if df.empty:
        parts.append("No finished experiments found. Results will appear after running the experiments.")
    else:
        parts += [
            "## Detection performance",
            "Unseen-attack recall = fraction of held-out (zero-day-like) attack samples flagged as anomalous.",
            "",
            markdown_table(
                df.sort_values(["mode", "experiment"]),
                {"display": "Experiment", "dataset": "Data", "unseen_recall": "**Unseen recall**", "known_recall": "Known recall",
                 "f1": "F1", "precision": "Precision", "recall": "Recall", "fpr": "FPR", "accuracy": "Accuracy", "roc_auc": "ROC-AUC"},
            ),
            "",
            "## Privacy and cost",
            "",
            markdown_table(
                df[df["model"] == "autoencoder"].sort_values("experiment"),
                {"display": "Experiment", "dp": "DP", "secagg": "SecAgg", "epsilon": "ε (max client)", "delta": "δ",
                 "noise_multiplier": "σ", "total_mb": "Total comm. (MiB)", "update_kb": "Update (KiB)",
                 "secagg_overhead_mb": "SecAgg overhead (MiB)", "train_time_s": "Train time (s)", "mean_client_compute_s": "Client compute / round (s)"},
                digits=3,
            ),
            "",
            "Centralized communication is the size of the raw training/validation traffic that would have to be uploaded.",
            "",
        ]
    if figures:
        parts += ["## Figures", ""] + [f"- `{p.as_posix()}`" for p in figures]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return out_path
