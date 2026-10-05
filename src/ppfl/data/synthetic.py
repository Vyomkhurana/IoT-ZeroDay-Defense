"""Synthetic IoT-like network-traffic generator — FOR TESTING THE PIPELINE ONLY.

The generator mimics the *structure* of N-BaIoT-style features (the same behavioural
statistics computed over several damped time windows, hence strongly correlated
columns) so the full pipeline can be exercised without downloading a dataset.
Results obtained on synthetic data say nothing about real-world detection quality
and must never be reported as research results.

Model
-----
Each device type ``d`` has a latent behaviour profile ``z ~ N(mu_d, Sigma_d)`` over
eight behavioural factors (packet rate, packet size, size spread, jitter, destination
diversity, TCP ratio, burstiness, connection duration), with a two-mode (idle /
active) mixture on the packet rate. Attack classes are deterministic shifts in this
latent space plus a variance reduction (attack traffic is more regular). Observed
features are non-linear transforms of the latent factors, observed through
``num_time_windows`` windows with window-dependent noise. During an attack, short
windows are dominated by attack traffic while long windows still average over the
device's benign history, which (as in real damped-window statistics) makes attacks
violate the cross-window structure of normal traffic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ppfl.utils.config import SyntheticDataConfig

LATENT_FACTORS = (
    "rate",
    "size",
    "size_std",
    "jitter",
    "dst_diversity",
    "tcp_ratio",
    "burstiness",
    "duration",
)

DEVICE_TYPES = (
    "doorbell",
    "thermostat",
    "baby_monitor",
    "security_camera",
    "webcam",
    "smart_plug",
    "smart_speaker",
    "light_bulb",
)

# Latent-space shifts (in units of the benign standard deviation) per attack family.
ATTACK_PROFILES: dict[str, dict[str, float]] = {
    "scan": {"dst_diversity": 4.0, "size": -1.0, "rate": 1.5, "duration": -2.0, "tcp_ratio": 2.0},
    "syn_flood": {"rate": 4.0, "size": -1.5, "size_std": -1.5, "tcp_ratio": 4.0, "duration": -2.5},
    "ack_flood": {"rate": 3.5, "size": -1.2, "tcp_ratio": 4.0, "jitter": -2.0, "burstiness": 1.0},
    "udp_flood": {"rate": 4.5, "size": 1.5, "tcp_ratio": -5.0, "size_std": -1.0},
    "http_flood": {"rate": 2.5, "size": 0.8, "tcp_ratio": 3.0, "dst_diversity": -1.5, "duration": 1.0},
    "c2_beacon": {"jitter": -2.5, "rate": 0.6, "size_std": -1.2, "duration": 1.2, "burstiness": -1.0},
    "junk": {"size": 2.0, "size_std": 2.0, "rate": 2.0, "tcp_ratio": -2.0},
}

_ATTACK_VARIANCE_SCALE = 0.6


def _device_profiles(num_devices: int, rng: np.random.Generator) -> list[dict[str, np.ndarray]]:
    k = len(LATENT_FACTORS)
    base = np.array([1.0, 1.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0])
    profiles = []
    for _ in range(num_devices):
        mean = base + rng.normal(0.0, 0.8, size=k)
        stds = rng.uniform(0.35, 0.6, size=k)
        a = rng.normal(0.0, 1.0, size=(k, k))
        corr = a @ a.T + k * np.eye(k)
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
        cov = np.outer(stds, stds) * corr
        profiles.append(
            {
                "mean": mean,
                "chol": np.linalg.cholesky(cov),
                "std": stds,
                "active_shift": rng.uniform(0.8, 1.6),
                "active_prob": rng.uniform(0.2, 0.5),
            }
        )
    return profiles


def _sample_latent(
    profile: dict[str, np.ndarray],
    n: int,
    rng: np.random.Generator,
    attack: str | None,
    strength: float,
) -> np.ndarray:
    eps = rng.standard_normal((n, len(LATENT_FACTORS)))
    if attack is None:
        z = profile["mean"] + eps @ profile["chol"].T
        active = rng.random(n) < profile["active_prob"]
        z[active, 0] += profile["active_shift"]
        return z
    shift = np.zeros(len(LATENT_FACTORS))
    for factor, delta in ATTACK_PROFILES[attack].items():
        shift[LATENT_FACTORS.index(factor)] = delta
    z = profile["mean"] + _ATTACK_VARIANCE_SCALE * (eps @ profile["chol"].T)
    return z + strength * shift * profile["std"]


def _attack_weight(window: int, num_windows: int) -> float:
    """Share of attack traffic seen by a damped window (short windows react first)."""
    return float(np.exp(-1.2 * window / max(num_windows - 1, 1)))


def _observe(
    z: np.ndarray, num_windows: int, rng: np.random.Generator, z_background: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Map latent factors to window-level traffic statistics.

    For attack traffic ``z_background`` holds the device's benign behaviour: long
    windows still average over pre-attack history, so each window sees a different
    benign/attack mix. This breaks the cross-window consistency of benign traffic
    (as in real damped-window features) instead of merely shifting the data.
    """
    cols: dict[str, np.ndarray] = {}
    for w in range(num_windows):
        window_len = 10.0 ** (w - 1)  # 0.1 s ... 1000 s (damped-window analogue)
        noise = rng.normal(0.0, 0.35 / np.sqrt(w + 1.0), size=z.shape)
        if z_background is None:
            zw = z + noise
        else:
            m = _attack_weight(w, num_windows)
            zw = m * z + (1.0 - m) * z_background + noise
        cols[f"w{w}_pkt_count"] = np.exp(zw[:, 0]) * window_len
        cols[f"w{w}_size_mean"] = 60.0 * np.exp(zw[:, 1])
        cols[f"w{w}_size_std"] = 0.3 * np.exp(zw[:, 1] + zw[:, 2])
        cols[f"w{w}_jitter"] = np.exp(zw[:, 3])
        cols[f"w{w}_dst_hosts"] = np.exp(zw[:, 4])
        cols[f"w{w}_tcp_ratio"] = 1.0 / (1.0 + np.exp(-zw[:, 5]))
        cols[f"w{w}_burstiness"] = np.log1p(np.exp(zw[:, 6]))
        cols[f"w{w}_duration"] = np.exp(zw[:, 7])
    return cols


def generate_synthetic_traffic(cfg: SyntheticDataConfig, seed: int) -> pd.DataFrame:
    """Generate a labelled synthetic IoT traffic table.

    Returns a DataFrame with numeric feature columns plus ``label`` and ``device``.
    A constant column and a small fraction of NaN/inf cells are included on purpose
    so that the cleaning stage is exercised.
    """
    unknown = sorted(set(cfg.attack_classes) - set(ATTACK_PROFILES))
    if unknown:
        raise ValueError(f"Unknown synthetic attack classes {unknown}; known: {sorted(ATTACK_PROFILES)}")
    rng = np.random.default_rng(seed)
    profiles = _device_profiles(cfg.num_devices, rng)
    frames = []
    for d, profile in enumerate(profiles):
        device = f"{DEVICE_TYPES[d % len(DEVICE_TYPES)]}_{d + 1}"
        batches: list[tuple[str, np.ndarray, np.ndarray | None]] = [
            ("benign", _sample_latent(profile, cfg.benign_per_device, rng, None, cfg.attack_strength), None)
        ]
        exposed = [a for a in cfg.attack_classes if rng.random() < cfg.attack_coverage]
        if not exposed and cfg.attack_classes:
            exposed = [cfg.attack_classes[int(rng.integers(len(cfg.attack_classes)))]]
        for attack in exposed:
            n = int(rng.integers(cfg.attacks_per_device_class // 2, cfg.attacks_per_device_class + 1))
            background = _sample_latent(profile, n, rng, None, cfg.attack_strength)
            batches.append((attack, _sample_latent(profile, n, rng, attack, cfg.attack_strength), background))
        for label, z, background in batches:
            df = pd.DataFrame(_observe(z, cfg.num_time_windows, rng, background))
            df["proto_version"] = 4.0  # constant column -> removed by cleaning
            df["label"] = label
            df["device"] = device
            frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    feature_cols = [c for c in data.columns if c not in ("label", "device")]
    if cfg.corrupt_fraction > 0:
        values = data[feature_cols].to_numpy(dtype=np.float64)
        mask = rng.random(values.shape) < cfg.corrupt_fraction
        corrupt = np.where(rng.random(values.shape) < 0.5, np.nan, np.inf)
        values[mask] = corrupt[mask]
        data[feature_cols] = values
    return data.iloc[rng.permutation(len(data))].reset_index(drop=True)
