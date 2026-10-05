"""Aggregate all finished experiments into ``results/summary.csv``, ``results/report.md``
and comparison figures under ``results/figures/``.

Example::

    python scripts/generate_report.py
    python scripts/generate_report.py --experiments centralized fedavg dp_fl full_ppfl
"""

from __future__ import annotations

import argparse

from ppfl.evaluation.comparison import collect_experiments, write_report
from ppfl.utils.config import resolve_path
from ppfl.utils.logging import setup_logging
from ppfl.utils.serialization import load_json
from ppfl.visualization.plots import comparison_figures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--experiments", nargs="*", help="restrict to these experiment names")
    args = parser.parse_args(argv)
    setup_logging("INFO")
    root = resolve_path(args.results_dir)
    exp_root = root / "experiments"
    exp_root.mkdir(parents=True, exist_ok=True)
    summary, histories = collect_experiments(exp_root, args.experiments)
    groups = load_json(root / "experiment_index.json") if (root / "experiment_index.json").exists() else {}
    existing = set(summary["experiment"]) if not summary.empty else set()
    groups = {k: [n for n in v if n in existing] for k, v in groups.items()}
    if args.experiments:
        groups["main"] = list(args.experiments)
    figures = comparison_figures(summary, histories, root / "figures", groups) if not summary.empty else []
    if not summary.empty:
        summary.to_csv(root / "summary.csv", index=False)
    report = write_report(summary, root / "report.md", [p.relative_to(root.parent) for p in figures])
    if summary.empty:
        print("No finished experiments found. Results will appear after running the experiments.")
    else:
        cols = ["display", "unseen_recall", "known_recall", "f1", "fpr", "roc_auc", "epsilon", "total_mb"]
        print(summary[[c for c in cols if c in summary]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"\nSummary: {root / 'summary.csv'}\nReport:  {report}\nFigures: {root / 'figures'} ({len(figures)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
