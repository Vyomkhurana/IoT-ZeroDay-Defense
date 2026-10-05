"""Configuration system and an end-to-end run of every training mode on tiny synthetic data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ppfl.experiments.runner import describe_mode, run_experiment
from ppfl.utils.config import ConfigError, load_config
from ppfl.utils.serialization import load_json

from conftest import tiny_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_all_shipped_configs_load():
    for path in CONFIGS.glob("*.yaml"):
        if path.name == "experiments.yaml":
            continue
        cfg = load_config(path)
        assert cfg.experiment.name


def test_inheritance_and_overrides():
    cfg = load_config(CONFIGS / "full_ppfl_fedprox.yaml", ["federated.rounds=3", "privacy.delta=1e-6"])
    assert cfg.privacy.enabled and cfg.secure_aggregation.enabled  # inherited from full_ppfl.yaml
    assert cfg.federated.strategy == "fedprox" and cfg.federated.rounds == 3
    assert cfg.privacy.delta == pytest.approx(1e-6)  # YAML 1.1 "1e-6" string coerced to float
    assert describe_mode(cfg) == "Full PPFL (FedProx)"


def test_overlay_is_applied_before_overrides():
    cfg = load_config(CONFIGS / "fedavg.yaml", ["partition.num_clients=4"], [CONFIGS / "synthetic.yaml"])
    assert cfg.data.dataset == "synthetic" and cfg.partition.num_clients == 4


def test_invalid_configs_are_rejected():
    with pytest.raises(ConfigError, match="Unknown key"):
        load_config(CONFIGS / "default.yaml", ["federated.roundz=3"])
    with pytest.raises(ConfigError, match="invalid"):
        load_config(CONFIGS / "default.yaml", ["federated.strategy=fedsgd"])
    with pytest.raises(ConfigError):
        load_config(CONFIGS / "default.yaml", ["partition.dirichlet_alpha=-1"])
    with pytest.raises(ConfigError):
        load_config(CONFIGS / "default.yaml", ["training.batch_size=abc"])


@pytest.mark.parametrize(
    "name,extra",
    [
        ("centralized", ["training.mode=centralized"]),
        ("local", ["training.mode=local"]),
        ("full_ppfl", ["training.mode=federated", "federated.strategy=fedprox", "privacy.enabled=true",
                       "secure_aggregation.enabled=true", "secure_aggregation.deterministic=true",
                       "detection.threshold_strategy=validation_f1"]),
    ],
)
def test_end_to_end_modes(tiny_cfg, tmp_path, name, extra):
    cfg = tiny_config(Path(tiny_cfg.data.processed_dir).parent, f"experiment.name={name}",
                      f"experiment.output_dir={tmp_path}", "evaluation.make_figures=true", *extra)
    metrics = run_experiment(cfg)
    exp = tmp_path / name
    assert (exp / "metrics.json").exists() and (exp / "config.yaml").exists() and (exp / "environment.json").exists()
    saved = load_json(exp / "metrics.json")
    ae = saved["models"]["autoencoder"]["summary"]
    for key in ("f1", "precision", "recall", "fpr", "unseen_recall", "known_recall"):
        assert 0.0 <= ae[key] <= 1.0
    history = pd.read_csv(exp / "metrics.csv")
    assert len(history) >= 2
    assert any((exp / "figures").glob("*.png"))
    if name == "centralized":
        assert {"isolation_forest", "one_class_svm"} <= set(saved["models"])
        assert saved["communication"]["raw_data_upload_bytes"] > 0
    if name == "full_ppfl":
        assert saved["privacy"]["epsilon"] > 0
        assert history["epsilon_max"].is_monotonic_increasing
        assert saved["federated"]["secagg_overhead_bytes"] > 0
        assert (exp / "privacy_ledger.csv").exists()
        assert metrics["setup"]["unseen_attack_classes"] == ["c2_beacon", "udp_flood"]


def test_runs_are_reproducible(tiny_cfg, tmp_path):
    results = []
    for i in range(2):
        cfg = tiny_config(Path(tiny_cfg.data.processed_dir).parent, f"experiment.name=repro{i}", f"experiment.output_dir={tmp_path}",
                          "privacy.enabled=true", "secure_aggregation.enabled=true", "secure_aggregation.deterministic=true")
        results.append(run_experiment(cfg)["models"]["autoencoder"]["summary"])
    assert results[0] == results[1]
