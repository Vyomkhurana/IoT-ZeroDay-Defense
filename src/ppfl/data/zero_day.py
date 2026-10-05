"""Zero-day (unseen attack) simulation by class holdout.

A set of attack classes is removed from *every* training and validation set and is
only ever used at test time. These held-out classes play the role of
"zero-day-like" attacks: the models (and the threshold selection) never observe them.
This is an experimental proxy for novel attacks, not a reproduction of a real
zero-day vulnerability.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass

import numpy as np

from ppfl.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class ZeroDaySplit:
    """Partition of class labels into benign, known attacks and unseen attacks."""

    benign_label: str
    known_attack_classes: tuple[str, ...]
    unseen_attack_classes: tuple[str, ...]

    def is_benign(self, labels: np.ndarray) -> np.ndarray:
        return np.asarray(labels) == self.benign_label

    def is_attack(self, labels: np.ndarray) -> np.ndarray:
        return ~self.is_benign(labels)

    def is_known_attack(self, labels: np.ndarray) -> np.ndarray:
        return np.isin(np.asarray(labels), self.known_attack_classes)

    def is_unseen_attack(self, labels: np.ndarray) -> np.ndarray:
        return np.isin(np.asarray(labels), self.unseen_attack_classes)

    def group_of(self, labels: np.ndarray) -> np.ndarray:
        """Return ``benign`` / ``known_attack`` / ``unseen_attack`` for each label."""
        labels = np.asarray(labels)
        out = np.full(labels.shape, "known_attack", dtype=object)
        out[self.is_benign(labels)] = "benign"
        out[self.is_unseen_attack(labels)] = "unseen_attack"
        return out.astype(str)


def resolve_holdout_classes(classes: list[str], patterns: list[str], benign_label: str) -> list[str]:
    """Expand fnmatch patterns (e.g. ``mirai_*``) against the available class names."""
    resolved: list[str] = []
    for pattern in patterns:
        matched = [c for c in classes if fnmatch.fnmatchcase(c, pattern)]
        if not matched:
            raise ValueError(f"Holdout pattern {pattern!r} matches no class. Available: {sorted(classes)}")
        resolved.extend(m for m in matched if m not in resolved)
    if benign_label in resolved:
        raise ValueError("The benign class cannot be held out as a zero-day attack")
    return sorted(resolved)


def make_zero_day_split(labels: np.ndarray, patterns: list[str], benign_label: str) -> ZeroDaySplit:
    classes = sorted(set(np.asarray(labels).tolist()))
    if benign_label not in classes:
        raise ValueError(f"Benign label {benign_label!r} not present in data")
    unseen = resolve_holdout_classes(classes, patterns, benign_label)
    known = [c for c in classes if c != benign_label and c not in unseen]
    if not unseen:
        LOGGER.warning("No holdout classes configured: zero-day (unseen attack) metrics will be undefined")
    if not known:
        LOGGER.warning("All attack classes are held out; no known attacks remain for validation")
    LOGGER.info("Zero-day split: known attacks=%s | unseen (held-out) attacks=%s", known, unseen)
    return ZeroDaySplit(benign_label, tuple(known), tuple(unseen))
