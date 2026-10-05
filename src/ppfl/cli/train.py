"""Train and evaluate one experiment.

Examples::

    python scripts/train.py --config configs/full_ppfl.yaml
    python scripts/train.py --config configs/smoke_test.yaml
    python scripts/train.py --config configs/dp.yaml --set privacy.noise_multiplier=2.0 --name dp_sigma2
"""

from __future__ import annotations

import argparse
import sys

from ppfl.cli.common import add_config_arguments, config_from_args
from ppfl.experiments.runner import run_experiment


def print_summary(metrics: dict) -> None:
    print("\n" + "=" * 78)
    print(f"Experiment: {metrics['experiment']['name']}  ({metrics['experiment']['label']})")
    if metrics["experiment"].get("synthetic_data"):
        print("NOTE: synthetic data - pipeline validation only, not a research result.")
    print("-" * 78)
    print(f"{'model':<18}{'unseen rec.':>12}{'known rec.':>12}{'F1':>9}{'prec.':>9}{'recall':>9}{'FPR':>9}{'AUC':>9}")
    for name, m in metrics["models"].items():
        s = m["summary"]
        print(f"{name:<18}{s['unseen_recall']:>12.4f}{s['known_recall']:>12.4f}{s['f1']:>9.4f}{s['precision']:>9.4f}{s['recall']:>9.4f}{s['fpr']:>9.4f}{s['roc_auc']:>9.4f}")
    priv = metrics.get("privacy") or {}
    if priv.get("epsilon") is not None:
        print(f"Privacy: epsilon = {priv['epsilon']:.3f} (max over clients), delta = {priv['delta']:g}, sigma = {priv['noise_multiplier']:.3f}, C = {priv['max_grad_norm']}")
    comm = metrics.get("communication") or {}
    if comm.get("total_bytes"):
        print(f"Communication: {comm['total_bytes'] / 2**20:.2f} MiB total")
    print(f"Time: {metrics['timing']['total_time_s']:.1f}s")
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arguments(parser, default_config=None)
    parser.add_argument("--name", help="experiment name (overrides experiment.name)")
    args = parser.parse_args(argv)
    if args.config is None:
        parser.error("--config is required (e.g. configs/full_ppfl.yaml)")
    cfg = config_from_args(args, [f"experiment.name={args.name}"] if args.name else None)
    try:
        metrics = run_experiment(cfg)
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2
    print_summary(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
