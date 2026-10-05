"""Privacy accounting for DP-SGD.

Each client owns one :class:`PrivacyAccountant` for the whole experiment. Every
DP-SGD step it takes (across all federated rounds it participates in) is recorded
as one application of the Sampled Gaussian Mechanism with noise multiplier
``sigma`` and sampling rate ``q``; the accountant composes them and converts the
result to an ``(epsilon, delta)`` guarantee. The default is the Rényi-DP accountant
(Mironov, 2017) as implemented in Opacus; the PRV and GDP accountants are also
available.

The resulting guarantee is *example-level* DP with respect to the client's local
training set: changing one traffic record changes the distribution of everything
the client ever releases (its model updates) by at most ``(epsilon, delta)``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from opacus.accountants import create_accountant
from opacus.accountants.utils import get_noise_multiplier


class PrivacyAccountant:
    """Thin, mechanism-agnostic wrapper around an Opacus accountant."""

    def __init__(self, mechanism: str = "rdp") -> None:
        self.mechanism = mechanism
        self._accountant = create_accountant(mechanism)

    def optimizer_hook(self, sample_rate: float) -> Callable[[Any], None]:
        """Hook for ``DPOptimizer.attach_step_hook``: records one SGM step per optimizer step."""
        return self._accountant.get_optimizer_hook_fn(sample_rate=sample_rate)

    def record(self, noise_multiplier: float, sample_rate: float, steps: int = 1) -> None:
        for _ in range(steps):
            self._accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

    @property
    def steps(self) -> int:
        return int(sum(entry[2] for entry in self._accountant.history))

    def get_epsilon(self, delta: float) -> float:
        if self.steps == 0:
            return 0.0
        return float(self._accountant.get_epsilon(delta=delta))


def compute_epsilon(noise_multiplier: float, sample_rate: float, steps: int, delta: float, mechanism: str = "rdp") -> float:
    """Epsilon after ``steps`` SGM applications (no training required)."""
    if steps <= 0:
        return 0.0
    acc = create_accountant(mechanism)
    acc.history = [(noise_multiplier, sample_rate, steps)]
    return float(acc.get_epsilon(delta=delta))


def calibrate_noise_multiplier(
    target_epsilon: float, delta: float, sample_rate: float, steps: int, mechanism: str = "rdp"
) -> float:
    """Smallest noise multiplier achieving ``target_epsilon`` after ``steps`` steps."""
    return float(
        get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            steps=steps,
            accountant=mechanism,
        )
    )


@dataclass
class PrivacyLedger:
    """Round-by-round record of every client's cumulative privacy loss."""

    delta: float
    rows: list[dict[str, Any]] = field(default_factory=list)

    def record(self, round_idx: int, client_id: int, epsilon: float, steps: int, noise_multiplier: float, sample_rate: float) -> None:
        self.rows.append(
            {
                "round": round_idx,
                "client_id": client_id,
                "epsilon": epsilon,
                "delta": self.delta,
                "steps": steps,
                "noise_multiplier": noise_multiplier,
                "sample_rate": sample_rate,
            }
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def epsilon_by_round(self) -> pd.DataFrame:
        """Max / mean cumulative epsilon over clients after each round."""
        df = self.to_frame()
        if df.empty:
            return pd.DataFrame(columns=["round", "epsilon_max", "epsilon_mean"])
        latest = df.sort_values("round").groupby(["round", "client_id"]).last().reset_index()
        rounds = sorted(latest["round"].unique())
        out = []
        current: dict[int, float] = {}
        for r in rounds:
            for _, row in latest[latest["round"] == r].iterrows():
                current[int(row["client_id"])] = float(row["epsilon"])
            values = list(current.values())
            out.append({"round": r, "epsilon_max": max(values), "epsilon_mean": sum(values) / len(values)})
        return pd.DataFrame(out)
