"""Re-evaluate a finished experiment from its saved model and configuration.

The data partition is rebuilt deterministically from the saved ``config.yaml``. By
default the saved threshold is reused (exact reproduction). Passing
``--threshold-strategy`` / ``--percentile`` recalibrates the threshold on the
*validation* data (never the test data) with the same protocol as training:
exact scores for centralized runs, aggregated histograms for federated runs.

Example::

    python scripts/evaluate.py --experiment full_ppfl
    python scripts/evaluate.py --experiment full_ppfl --threshold-strategy validation_f1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from ppfl.data.federated_dataset import build_federated_dataset
from ppfl.detection.threshold import ThresholdResult
from ppfl.evaluation.evaluator import evaluate_detector
from ppfl.experiments.common import EvalContext, threshold_to_dict
from ppfl.models.autoencoder import Autoencoder
from ppfl.models.base import AnomalyDetector
from ppfl.utils.config import load_config, resolve_path
from ppfl.utils.environment import resolve_device
from ppfl.utils.logging import setup_logging
from ppfl.utils.serialization import load_json, save_json
from ppfl.visualization.plots import plot_binary_confusion, plot_group_confusion, plot_roc, plot_score_distribution, plot_scores_by_class, set_style


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", "-e", required=True, help="experiment name (directory under results/experiments)")
    parser.add_argument("--results-dir", default="results/experiments")
    parser.add_argument("--threshold-strategy", choices=["percentile", "validation_f1", "mean_std"])
    parser.add_argument("--percentile", type=float)
    args = parser.parse_args(argv)

    exp_dir = resolve_path(args.results_dir) / args.experiment
    synthetic_dir = resolve_path("results/synthetic/experiments") / args.experiment
    if not (exp_dir / "config.yaml").exists() and (synthetic_dir / "config.yaml").exists():
        exp_dir = synthetic_dir
    if not (exp_dir / "config.yaml").exists():
        print(f"ERROR: no finished experiment at {exp_dir}", file=sys.stderr)
        return 2
    overrides = []
    if args.threshold_strategy:
        overrides.append(f"detection.threshold_strategy={args.threshold_strategy}")
    if args.percentile is not None:
        overrides.append(f"detection.percentile={args.percentile}")
    cfg = load_config(exp_dir / "config.yaml", overrides)
    setup_logging(cfg.logging.level)
    device = resolve_device(cfg.experiment.device)
    dataset = build_federated_dataset(cfg)
    ctx = EvalContext(cfg, dataset, device)
    out = exp_dir / "evaluation"
    set_style()
    synthetic = cfg.data.dataset == "synthetic"
    results = {}

    if (exp_dir / "model" / "autoencoder.pt").exists():
        model = Autoencoder.from_config(dataset.input_dim, cfg.model)
        model.load_state_dict(torch.load(exp_dir / "model" / "autoencoder.pt", map_location=device))
        model.to(device)
        saved = load_json(exp_dir / "model" / "model.json")["threshold"]
        if overrides:
            thr = ctx.exact_threshold(model) if cfg.training.mode == "centralized" else ctx.histogram_threshold(model)
        else:
            thr = ThresholdResult(saved["value"], saved["strategy"], {"source": "saved"})
        ev = evaluate_detector(ctx.scorer(model), dataset, thr.value, cfg.evaluation.per_client)
        results["autoencoder"] = {"threshold": threshold_to_dict(thr), "summary": ev.summary, "report": ev.report, "per_client": ev.per_client}
        groups = dataset.zero_day.group_of(ev.labels)
        plot_group_confusion(ev.report, out, synthetic)
        plot_binary_confusion(ev.report, out, synthetic)
        plot_score_distribution(ev.scores, groups, thr.value, out, synthetic)
        plot_scores_by_class(ev.scores, ev.labels, groups, thr.value, out, synthetic)
        plot_roc(ev.scores, groups, out, synthetic)
    else:
        print("No saved autoencoder (local-only experiments keep per-client models in memory only).")

    saved_metrics = load_json(exp_dir / "metrics.json")
    for name in ("isolation_forest", "one_class_svm"):
        path = exp_dir / "model" / f"{name}.joblib"
        if path.exists():
            det = AnomalyDetector.load(path)
            thr_value = saved_metrics["models"][name]["threshold"]["value"]
            if overrides:
                thr_value = ctx.selector.from_scores(det.score(np.concatenate(ctx.val_benign)), det.score(np.concatenate(ctx.val_attack))).value
            ev = evaluate_detector(det.score, dataset, thr_value, cfg.evaluation.per_client)
            results[name] = {"threshold": thr_value, "summary": ev.summary, "report": ev.report}
            plot_score_distribution(ev.scores, dataset.zero_day.group_of(ev.labels), thr_value, out, synthetic, name)

    save_json(results, out / "evaluation.json")
    print(f"\nEvaluation of '{args.experiment}'" + (" [SYNTHETIC DATA]" if synthetic else ""))
    print(f"{'model':<18}{'threshold':>12}{'unseen rec.':>12}{'known rec.':>12}{'F1':>9}{'FPR':>9}{'AUC':>9}")
    for name, r in results.items():
        s = r["summary"]
        print(f"{name:<18}{s['threshold']:>12.4g}{s['unseen_recall']:>12.4f}{s['known_recall']:>12.4f}{s['f1']:>9.4f}{s['fpr']:>9.4f}{s['roc_auc']:>9.4f}")
    if "autoencoder" in results:
        print("\nPer-class flagged rate (detection rate for attacks, false-positive rate for benign):")
        for label, info in results["autoencoder"]["report"]["per_class"].items():
            print(f"  {label:<28s} {info['group']:<14s} n={info['n']:>8,}  flagged={info['flagged_rate']:.4f}")
    print(f"\nWritten to {Path(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
