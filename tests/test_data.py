"""Dataset loading, cleaning, scaling and non-IID partitioning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ppfl.data.federated_dataset import build_federated_dataset
from ppfl.data.loader import RawDataset, canonical_dataset_name, load_nbaiot
from ppfl.data.partition import heterogeneity_index, partition_indices, partition_summary
from ppfl.data.preprocessing import FeatureScaler, FeatureStatistics, clean_dataset
from ppfl.data.synthetic import generate_synthetic_traffic
from ppfl.utils.config import SyntheticDataConfig

from conftest import tiny_config


# ------------------------------------------------------------------ preprocessing
def test_cleaning_handles_inf_nan_and_constant_columns():
    frame = pd.DataFrame(
        {
            "a": [1.0, np.inf, 3.0, 4.0],
            "b": [1.0, 2.0, np.nan, 4.0],
            "const": [7.0, 7.0, 7.0, 7.0],
            "mostly_missing": [np.nan, np.nan, np.nan, 1.0],
            "label": ["benign", "benign", "scan", "scan"],
            "device": ["d1"] * 4,
        }
    )
    out, features, report = clean_dataset(RawDataset(frame, ["a", "b", "const", "mostly_missing"]), "benign")
    assert features == ["a", "b"]
    assert report.infinite_cells == 1
    assert report.dropped_constant_columns == ["const"]
    assert report.dropped_missing_columns == ["mostly_missing"]
    assert np.isnan(out["a"].iloc[1])  # inf -> missing (imputed later from training statistics)
    assert list(out["label"]) == ["benign", "benign", "scan", "scan"]


def test_federated_scaler_equals_pooled_fit():
    rng = np.random.default_rng(0)
    parts = [rng.normal(loc=i, scale=i + 1, size=(50 + 10 * i, 5)) for i in range(4)]
    parts[1][3, 2] = np.nan
    merged = FeatureStatistics.merge_all([FeatureStatistics.from_array(p) for p in parts])
    fed = FeatureScaler("standard").fit_from_statistics(merged)
    pooled = np.concatenate(parts)
    assert np.allclose(fed.mean_, np.nanmean(pooled, axis=0))
    assert np.allclose(fed.scale_, np.nanstd(pooled, axis=0))
    z = fed.transform(pooled)
    assert np.isfinite(z).all()  # NaN imputed with the training mean
    assert np.allclose(np.delete(z, 3 + 60, axis=0).mean(axis=0), 0, atol=0.05)


def test_minmax_and_log_scaler():
    x = np.array([[0.0, 10.0], [5.0, 1000.0], [10.0, 100.0]])
    z = FeatureScaler("minmax").fit(x).transform(x)
    assert z.min() == pytest.approx(0) and z.max() == pytest.approx(1)
    zl = FeatureScaler("standard", log_transform=True).fit(x).transform(x)
    assert np.isfinite(zl).all()


def test_synthetic_generator_is_deterministic_and_complete():
    cfg = SyntheticDataConfig(num_devices=2, benign_per_device=100, attacks_per_device_class=20, attack_coverage=1.0)
    a, b = generate_synthetic_traffic(cfg, 1), generate_synthetic_traffic(cfg, 1)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["label"]) == {"benign", *cfg.attack_classes}
    assert a["device"].nunique() == 2
    assert not a.drop(columns=["label", "device"]).isna().all().any()


def test_dataset_name_aliases():
    assert canonical_dataset_name("nbaIoT") == "nbaiot"
    assert canonical_dataset_name("N-BaIoT") == "nbaiot"
    assert canonical_dataset_name("TON_IoT") == "ton_iot"
    with pytest.raises(ValueError):
        canonical_dataset_name("mnist")


@pytest.mark.parametrize("layout", ["flat", "nested"])
def test_nbaiot_loader_layouts(tmp_path, layout):
    cols = ["MI_dir_L5_weight", "MI_dir_L5_mean", "H_L0.1_variance"]
    rng = np.random.default_rng(0)

    def write(path, n):
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rng.normal(size=(n, 3)), columns=cols).to_csv(path, index=False)

    if layout == "flat":
        write(tmp_path / "1.benign.csv", 30)
        write(tmp_path / "1.gafgyt.scan.csv", 10)
        write(tmp_path / "2.mirai.udpplain.csv", 12)
        pd.DataFrame({"x": [1]}).to_csv(tmp_path / "features.csv", index=False)  # ignored
    else:
        write(tmp_path / "Danmini_Doorbell" / "benign_traffic.csv", 30)
        write(tmp_path / "Danmini_Doorbell" / "gafgyt_attacks" / "scan.csv", 10)
        write(tmp_path / "Ecobee_Thermostat" / "mirai_attacks" / "udpplain.csv", 12)
    cfg = tiny_config(tmp_path, "data.dataset=nbaiot", "data.max_samples_per_group=8")
    raw = load_nbaiot(tmp_path, cfg)
    assert raw.feature_columns == cols
    counts = raw.frame["label"].value_counts().to_dict()
    assert counts == {"benign": 30, "gafgyt_scan": 8, "mirai_udpplain": 8}  # attack groups capped
    assert set(raw.frame["device"]) == {"Danmini_Doorbell", "Ecobee_Thermostat"}


def test_nbaiot_loader_missing_files(tmp_path):
    cfg = tiny_config(tmp_path, "data.dataset=nbaiot")
    with pytest.raises(FileNotFoundError, match="N-BaIoT"):
        load_nbaiot(tmp_path, cfg)


# ------------------------------------------------------------------ partitioning
def _toy_labels(n_per=400):
    labels = np.array(["benign"] * (3 * n_per) + ["a"] * n_per + ["b"] * n_per + ["c"] * n_per)
    devices = np.array(["d1", "d2", "d3"] * (len(labels) // 3))
    return labels, devices


@pytest.mark.parametrize("strategy", ["dirichlet", "iid", "device"])
def test_partition_is_disjoint_and_complete(strategy):
    labels, devices = _toy_labels()
    parts = partition_indices(labels, devices, num_clients=5, strategy=strategy, alpha=0.5,
                              stratify_by=["device", "label"], rng=np.random.default_rng(0),
                              benign_label="benign", min_benign=20)
    allidx = np.concatenate(parts)
    assert len(allidx) == len(labels) == len(np.unique(allidx))
    assert all((labels[p] == "benign").sum() >= 20 for p in parts)


def test_lower_alpha_is_more_heterogeneous():
    labels, devices = _toy_labels(1000)

    def hetero(alpha):
        vals = []
        for seed in range(5):
            parts = partition_indices(labels, devices, num_clients=8, strategy="dirichlet", alpha=alpha,
                                      stratify_by=["label"], rng=np.random.default_rng(seed))
            vals.append(heterogeneity_index(partition_summary(labels, parts)))
        return np.mean(vals)

    assert hetero(0.1) > hetero(1.0) > hetero(100.0)


def test_min_benign_repair():
    labels, devices = _toy_labels(300)
    parts = partition_indices(labels, devices, num_clients=10, strategy="dirichlet", alpha=0.05,
                              stratify_by=["device", "label"], rng=np.random.default_rng(1),
                              benign_label="benign", min_benign=40, max_retries=2)
    assert min((labels[p] == "benign").sum() for p in parts) >= 40


def test_federated_dataset_is_deterministic_and_scaled(tiny_cfg, tiny_dataset):
    again = build_federated_dataset(tiny_cfg)
    for a, b in zip(tiny_dataset.clients, again.clients):
        assert np.array_equal(a.train.X, b.train.X) and np.array_equal(a.test.labels, b.test.labels)
    pooled = tiny_dataset.pooled("train").X
    assert np.isfinite(pooled).all()
    assert np.allclose(pooled.mean(axis=0), 0, atol=1e-3)  # scaler fitted on training rows
    total = sum(len(c.train) + len(c.val) + len(c.test) + len(c.zero_day) for c in tiny_dataset.clients)
    assert total == tiny_dataset.partition_table.to_numpy().sum()
