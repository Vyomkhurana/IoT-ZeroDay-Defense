"""Build (and optionally export) the non-IID client partition for a configuration.

Training rebuilds the partition deterministically from the config and seed, so this
command is for *inspection*: it writes the client x class table, a heterogeneity
summary and a distribution figure, and with ``--export`` one ``.npz`` per client.

Example::

    python scripts/create_clients.py --config configs/fedavg.yaml --set partition.dirichlet_alpha=0.1
"""

from __future__ import annotations

import argparse

import numpy as np

from ppfl.cli.common import add_config_arguments, config_from_args
from ppfl.data.federated_dataset import build_federated_dataset
from ppfl.utils.logging import setup_logging
from ppfl.utils.serialization import save_json
from ppfl.visualization.plots import plot_client_distribution, set_style


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arguments(parser)
    parser.add_argument("--export", action="store_true", help="also write one .npz per client")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    setup_logging(cfg.logging.level)
    ds = build_federated_dataset(cfg)
    p = cfg.partition
    tag = f"{p.strategy}_k{p.num_clients}" + (f"_a{p.dirichlet_alpha:g}" if p.strategy == "dirichlet" else "") + f"_s{cfg.experiment.seed}"
    out = cfg.processed_dir / "partitions" / tag
    out.mkdir(parents=True, exist_ok=True)
    ds.partition_table.to_csv(out / "client_distribution.csv")
    save_json(
        {
            "partition": vars(p),
            "heterogeneity_js": ds.metadata["heterogeneity_js"],
            "known_attack_classes": list(ds.zero_day.known_attack_classes),
            "unseen_attack_classes": list(ds.zero_day.unseen_attack_classes),
            "clients": [
                {"client_id": c.client_id, "train": len(c.train), "val": len(c.val), "test": len(c.test), "zero_day": len(c.zero_day),
                 "benign_train": int((c.train.labels == cfg.data.benign_label).sum()), "devices": sorted(set(c.train.devices.tolist()))}
                for c in ds.clients
            ],
        },
        out / "summary.json",
    )
    set_style()
    plot_client_distribution(ds.partition_table.reset_index(), out, cfg.data.dataset == "synthetic", list(ds.zero_day.unseen_attack_classes))
    if args.export:
        for c in ds.clients:
            np.savez_compressed(
                out / f"client_{c.client_id:03d}.npz",
                **{f"X_{n}": getattr(c, n).X for n in ("train", "val", "test", "zero_day")},
                **{f"y_{n}": getattr(c, n).labels.astype(str) for n in ("train", "val", "test", "zero_day")},
            )
    print(ds.partition_table.to_string())
    print(f"\nHeterogeneity (mean Jensen-Shannon distance to global label mix): {ds.metadata['heterogeneity_js']:.3f}")
    print(f"Partition artefacts written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
