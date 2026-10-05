"""Prepare a dataset: load raw files, clean, and store ``data/processed/<dataset>/cleaned.parquet``.

Examples::

    python scripts/prepare_data.py --dataset nbaIoT
    python scripts/prepare_data.py --dataset nbaiot --raw-dir D:/datasets/N-BaIoT
    python scripts/prepare_data.py --synthetic
"""

from __future__ import annotations

import argparse
import sys

from ppfl.cli.common import add_config_arguments, config_from_args
from ppfl.data.federated_dataset import prepare_dataset
from ppfl.data.loader import canonical_dataset_name
from ppfl.utils.logging import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", help="nbaiot | ciciot2023 | ton_iot | bot_iot | synthetic (case-insensitive)")
    parser.add_argument("--synthetic", action="store_true", help="generate the small synthetic IoT-like test dataset")
    parser.add_argument("--raw-dir", help="directory containing the raw dataset files (default: data/raw/<dataset>)")
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    if not args.synthetic and not args.dataset:
        parser.error("specify --dataset NAME or --synthetic")
    dataset = "synthetic" if args.synthetic else canonical_dataset_name(args.dataset)
    cfg = config_from_args(args, [f"data.dataset={dataset}"] + ([f"data.raw_dir={args.raw_dir}"] if args.raw_dir else []))
    setup_logging(cfg.logging.level)
    try:
        frame, meta = prepare_dataset(cfg)
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2
    print(f"\nPrepared '{dataset}': {meta['num_rows']:,} rows x {len(meta['feature_names'])} features -> {cfg.processed_dir}")
    print("Class counts:")
    for label, count in meta["class_counts"].items():
        print(f"  {label:<28s} {count:>10,}")
    if dataset == "synthetic":
        print("\nNOTE: synthetic data is for pipeline testing only; do not report results obtained on it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
