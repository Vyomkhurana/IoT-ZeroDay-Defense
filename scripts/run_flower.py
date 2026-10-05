#!/usr/bin/env python
"""Run FedAvg / FedProx through the optional Flower adapter (localhost gRPC, no Ray needed).

Requires ``pip install "flwr>=1.10"``. Secure aggregation is only available in the
native engine (``scripts/train.py``); DP-SGD works in both.

Example::

    python scripts/run_flower.py --config configs/fedavg.yaml --overlay configs/synthetic.yaml --set federated.rounds=3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppfl.cli.common import add_config_arguments, config_from_args  # noqa: E402
from ppfl.utils.logging import setup_logging  # noqa: E402
from ppfl.utils.serialization import save_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arguments(parser)
    parser.add_argument("--address", default="127.0.0.1:8089")
    args = parser.parse_args()
    cfg = config_from_args(args)
    setup_logging(cfg.logging.level)
    from ppfl.federated.flower_adapter import run_flower

    result = run_flower(cfg, args.address)
    out = cfg.experiment_dir.parent / f"{cfg.experiment.name}_flower" / "flower_metrics.json"
    save_json(result, out)
    s = result["summary"]
    print(f"\n[{result['framework']}] {result['strategy']} x {result['rounds']} rounds in {result['wall_time_s']:.1f}s")
    print(f"F1 {s['f1']:.4f} | unseen-attack recall {s['unseen_recall']:.4f} | FPR {s['fpr']:.4f} | AUC {s['roc_auc']:.4f}")
    print(f"Written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
