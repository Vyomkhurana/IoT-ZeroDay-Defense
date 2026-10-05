"""Run experiment suites defined in ``configs/experiments.yaml`` and build the report.

Examples::

    python scripts/run_experiments.py                       # main suite on the configured dataset
    python scripts/run_experiments.py --suite all
    python scripts/run_experiments.py --suite main --suite privacy
    python scripts/run_experiments.py --suite all --overlay configs/synthetic.yaml   # pipeline check
    python scripts/run_experiments.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import yaml

from ppfl.data.federated_dataset import load_or_prepare
from ppfl.experiments.runner import run_experiment
from ppfl.utils.config import Config, load_config, resolve_path
from ppfl.utils.logging import setup_logging
from ppfl.utils.serialization import load_json, save_json


def expand_suite(spec: Any, base_dir: Path) -> list[tuple[Path, list[str]]]:
    """Return ``(config_path, overrides)`` pairs for one suite specification."""
    if isinstance(spec, list):
        return [(base_dir / item, []) for item in spec]
    if isinstance(spec, dict) and {"config", "parameter", "values"} <= set(spec):
        template = spec.get("name", Path(spec["config"]).stem + "_{value}")
        return [
            (base_dir / spec["config"], [f"{spec['parameter']}={json.dumps(v)}", f"experiment.name={template.format(value=v)}"])
            for v in spec["values"]
        ]
    raise ValueError(f"Invalid suite specification: {spec!r}")


def _data_key(cfg: Config) -> str:
    return json.dumps({"data": vars(cfg.data) | {"synthetic": vars(cfg.data.synthetic)}, "seed": cfg.experiment.seed}, sort_keys=True, default=str)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suites-file", default="configs/experiments.yaml")
    parser.add_argument("--suite", action="append", help="suite name (repeatable) or 'all'; default: main")
    parser.add_argument("--overlay", action="append", default=[], help="YAML merged into every experiment config")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="override applied to every experiment")
    parser.add_argument("--skip-existing", action="store_true", help="skip experiments that already have metrics.json")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--no-report", action="store_true", help="do not run generate_report afterwards")
    parser.add_argument("--list", action="store_true", help="list the experiments that would run and exit")
    args = parser.parse_args(argv)

    suites_file = resolve_path(args.suites_file)
    suites: dict[str, Any] = yaml.safe_load(suites_file.read_text(encoding="utf-8"))["suites"]
    chosen = args.suite or ["main"]
    if "all" in chosen:
        chosen = list(suites)
    unknown = [s for s in chosen if s not in suites]
    if unknown:
        parser.error(f"unknown suite(s) {unknown}; available: {list(suites)}")

    plan: list[tuple[str, Config]] = []
    index: dict[str, list[str]] = {}
    for suite in chosen:
        for path, extra in expand_suite(suites[suite], suites_file.parent):
            cfg = load_config(path, list(args.overrides) + extra, args.overlay)
            plan.append((suite, cfg))
            index.setdefault(suite, []).append(cfg.experiment.name)

    if args.list:
        for suite, cfg in plan:
            print(f"[{suite}] {cfg.experiment.name:<28s} mode={cfg.training.mode:<11s} dataset={cfg.data.dataset}")
        return 0

    setup_logging("INFO")
    results_root = resolve_path(plan[0][1].experiment.output_dir).parent
    index_path = results_root / "experiment_index.json"
    previous = load_json(index_path) if index_path.exists() else {}
    save_json({**previous, **index}, index_path)

    frames: dict[str, Any] = {}
    failures, start = [], time.perf_counter()
    for i, (suite, cfg) in enumerate(plan, 1):
        name = cfg.experiment.name
        if args.skip_existing and (cfg.experiment_dir / "metrics.json").exists():
            print(f"[{i}/{len(plan)}] skip {name} (exists)")
            continue
        print(f"\n[{i}/{len(plan)}] suite={suite} experiment={name}")
        try:
            key = _data_key(cfg)
            if key not in frames:
                frames = {key: load_or_prepare(cfg)[0]}  # keep only the current dataset in memory
            run_experiment(cfg, frame=frames[key])
        except Exception as exc:  # keep going: one failed run must not lose the others
            failures.append((name, repr(exc)))
            traceback.print_exc()
            if args.stop_on_error:
                break
    print(f"\nFinished {len(plan) - len(failures)}/{len(plan)} experiments in {(time.perf_counter() - start) / 60:.1f} min")
    for name, err in failures:
        print(f"  FAILED {name}: {err}", file=sys.stderr)

    if not args.no_report:
        from ppfl.cli.generate_report import main as report_main

        report_main(["--results-dir", str(results_root)])
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
