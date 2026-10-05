"""Zero-day holdout correctness and detection / zero-day metrics."""

from __future__ import annotations

import numpy as np
import pytest

from ppfl.data.zero_day import ZeroDaySplit, make_zero_day_split, resolve_holdout_classes
from ppfl.evaluation.metrics import binary_metrics, confusion_counts
from ppfl.evaluation.zero_day_metrics import headline, zero_day_report


def test_holdout_pattern_resolution():
    classes = ["benign", "gafgyt_scan", "gafgyt_udp", "mirai_ack", "mirai_scan"]
    assert resolve_holdout_classes(classes, ["mirai_*"], "benign") == ["mirai_ack", "mirai_scan"]
    assert resolve_holdout_classes(classes, ["gafgyt_udp", "mirai_ack"], "benign") == ["gafgyt_udp", "mirai_ack"]
    with pytest.raises(ValueError, match="matches no class"):
        resolve_holdout_classes(classes, ["torii_*"], "benign")
    with pytest.raises(ValueError, match="benign"):
        resolve_holdout_classes(classes, ["benign"], "benign")


def test_zero_day_split_groups():
    labels = np.array(["benign", "a", "b", "c", "benign"])
    split = make_zero_day_split(labels, ["c"], "benign")
    assert split.known_attack_classes == ("a", "b") and split.unseen_attack_classes == ("c",)
    assert split.group_of(labels).tolist() == ["benign", "known_attack", "known_attack", "unseen_attack", "benign"]


def test_unseen_classes_never_reach_training_or_validation(tiny_dataset):
    unseen = set(tiny_dataset.zero_day.unseen_attack_classes)
    assert unseen == {"udp_flood", "c2_beacon"}
    for c in tiny_dataset.clients:
        for split in (c.train, c.val, c.test):
            assert not unseen & set(split.labels.tolist())
        assert set(c.zero_day.labels.tolist()) <= unseen
    test, _ = tiny_dataset.global_test()
    assert unseen <= set(test.labels.tolist())  # but they are present at test time
    # The autoencoder only ever trains on benign rows (no contamination configured).
    for c in tiny_dataset.clients:
        X = c.training_matrix("benign", 0.0, np.random.default_rng(0))
        assert len(X) == int((c.train.labels == "benign").sum())


def test_contamination_adds_only_known_attacks(tiny_dataset):
    c = tiny_dataset.clients[0]
    n_benign = int((c.train.labels == "benign").sum())
    X = c.training_matrix("benign", 0.1, np.random.default_rng(0))
    assert len(X) == n_benign + min(round(0.1 * n_benign), int((c.train.labels != "benign").sum()))


def test_binary_metrics_hand_example():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    y_pred = np.array([1, 1, 0, 0, 0, 1, 0, 1])
    assert confusion_counts(y_true, y_pred) == {"tp": 3, "fp": 1, "tn": 3, "fn": 1}
    m = binary_metrics(y_true, y_pred, scores=np.array([0.9, 0.8, 0.3, 0.1, 0.2, 0.7, 0.1, 0.95]))
    assert m["precision"] == pytest.approx(0.75)
    assert m["recall"] == pytest.approx(0.75)
    assert m["f1"] == pytest.approx(0.75)
    assert m["fpr"] == pytest.approx(0.25)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["roc_auc"] == pytest.approx(15 / 16)


def test_undefined_metrics_are_nan():
    m = binary_metrics(np.array([0, 0]), np.array([0, 0]), scores=np.array([0.1, 0.2]))
    assert np.isnan(m["precision"]) and np.isnan(m["recall"]) and np.isnan(m["roc_auc"])
    assert m["fpr"] == 0.0


def test_unseen_attack_recall_definition():
    split = ZeroDaySplit("benign", ("known",), ("zd1", "zd2"))
    labels = np.array(["benign"] * 4 + ["known"] * 2 + ["zd1"] * 3 + ["zd2"] * 1)
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.1, 0.95, 0.85, 0.2, 0.99])
    report = zero_day_report(scores, labels, threshold=0.5, split=split)
    assert report["unseen_attack_recall"] == pytest.approx(3 / 4)  # zd1: 2/3 caught, zd2: 1/1
    assert report["known_attack_recall"] == pytest.approx(1 / 2)
    assert report["benign_fpr"] == pytest.approx(1 / 4)
    assert report["per_class"]["zd1"]["flagged_rate"] == pytest.approx(2 / 3)
    assert report["per_class"]["zd2"]["group"] == "unseen_attack"
    assert report["unseen"]["n"] == 4 + 4
    h = headline(report)
    assert h["unseen_recall"] == report["unseen_attack_recall"]
    assert h["recall"] == pytest.approx(4 / 6)
