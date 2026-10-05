"""Shared fixtures: small, fast configurations on synthetic data in temporary directories."""

from __future__ import annotations

from pathlib import Path

import pytest

import ppfl  # noqa: F401  (Windows OpenMP import-order guard)
from ppfl.data.federated_dataset import build_federated_dataset, prepare_dataset
from ppfl.utils.config import Config, load_config
from ppfl.utils.logging import setup_logging

TINY_SYNTHETIC = [
    "data.dataset=synthetic",
    "data.holdout_classes=[udp_flood, c2_beacon]",
    "data.log_transform=true",
    "data.synthetic.num_devices=3",
    "data.synthetic.benign_per_device=600",
    "data.synthetic.attacks_per_device_class=80",
    "data.synthetic.attack_coverage=1.0",
    "partition.num_clients=3",
    "partition.min_benign_per_client=50",
    "model.hidden_dims=[16]",
    "model.latent_dim=4",
    "training.batch_size=64",
    "training.epochs=2",
    "federated.rounds=2",
    "federated.local_epochs=1",
    "detection.histogram_bins=1024",
    "evaluation.make_figures=false",
    "baselines.one_class_svm.max_train_samples=300",
    "baselines.isolation_forest.n_estimators=20",
    "logging.level=WARNING",
]


@pytest.fixture(scope="session", autouse=True)
def _quiet_logging() -> None:
    setup_logging("WARNING")


def tiny_config(tmp: Path, *extra: str) -> Config:
    return load_config(
        Path(__file__).resolve().parents[1] / "configs" / "default.yaml",
        TINY_SYNTHETIC + [f"data.processed_dir={tmp / 'processed'}", f"data.raw_dir={tmp / 'raw'}",
                          f"experiment.output_dir={tmp / 'experiments'}", *extra],
    )


@pytest.fixture(scope="session")
def tiny_cfg(tmp_path_factory: pytest.TempPathFactory) -> Config:
    tmp = tmp_path_factory.mktemp("ppfl")
    cfg = tiny_config(tmp)
    prepare_dataset(cfg)
    return cfg


@pytest.fixture(scope="session")
def tiny_dataset(tiny_cfg: Config):
    return build_federated_dataset(tiny_cfg)
