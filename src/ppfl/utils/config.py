"""Structured YAML configuration.

Configuration files are plain YAML that map onto the dataclasses below. A file may
inherit from one or more other files through a top-level ``base`` key (paths are
relative to the inheriting file); values are deep-merged, later files win. Values
can additionally be overridden from the command line with ``--set key.sub=value``.

All hyper-parameters live here so that no tunable value is hard-coded elsewhere.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import types
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or contains invalid values."""


def project_root() -> Path:
    """Return the repository root (directory holding ``pyproject.toml``).

    Resolution order: ``PPFL_ROOT`` environment variable, the source checkout that
    contains this file, then the current working directory.
    """
    env = os.environ.get("PPFL_ROOT")
    if env:
        return Path(env).resolve()
    candidate = Path(__file__).resolve().parents[3]
    if (candidate / "pyproject.toml").exists():
        return candidate
    return Path.cwd().resolve()


def resolve_path(path: str | Path) -> Path:
    """Resolve a (possibly relative) path against the project root."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else (project_root() / p).resolve()


# --------------------------------------------------------------------------------------
# Configuration schema
# --------------------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Experiment identity and runtime environment."""

    name: str = "default"
    description: str = ""
    seed: int = 42
    output_dir: str = "results/experiments"
    device: str = "auto"  # auto | cpu | cuda | cuda:N
    deterministic: bool = True
    num_threads: int | None = None


@dataclass
class SyntheticDataConfig:
    """Parameters of the synthetic IoT-like traffic generator (testing only)."""

    num_devices: int = 6
    benign_per_device: int = 4000
    attacks_per_device_class: int = 500
    attack_classes: list[str] = field(
        default_factory=lambda: [
            "scan",
            "syn_flood",
            "ack_flood",
            "udp_flood",
            "http_flood",
            "c2_beacon",
        ]
    )
    attack_coverage: float = 0.85  # probability a device was exposed to a given attack
    attack_strength: float = 1.0  # global multiplier on attack deviations
    num_time_windows: int = 5
    corrupt_fraction: float = 0.001  # fraction of cells set to NaN/inf to exercise cleaning


@dataclass
class DataConfig:
    """Dataset location, preprocessing and splitting."""

    dataset: str = "nbaiot"  # nbaiot | ciciot2023 | ton_iot | bot_iot | synthetic
    raw_dir: str | None = None  # default: data/raw/<dataset>
    processed_dir: str = "data/processed"
    benign_label: str = "benign"
    holdout_classes: list[str] = field(default_factory=list)  # fnmatch patterns
    val_fraction: float = 0.15
    test_fraction: float = 0.25
    scaler: str = "standard"  # standard | minmax
    log_transform: bool = False
    clip_value: float | None = None
    train_contamination: float = 0.0  # fraction of known-attack train rows mixed into AE training
    max_samples_per_group: int | None = None  # per (device, attack class) cap at preparation
    max_benign_samples_per_group: int | None = None  # per (device, benign) cap at preparation
    max_missing_fraction: float = 0.5  # drop feature columns with more missing values
    loader_options: dict[str, Any] = field(default_factory=dict)
    synthetic: SyntheticDataConfig = field(default_factory=SyntheticDataConfig)


@dataclass
class PartitionConfig:
    """Non-IID partitioning of data across simulated IoT gateways."""

    num_clients: int = 10
    strategy: str = "dirichlet"  # dirichlet | device | iid
    dirichlet_alpha: float = 0.5
    stratify_by: list[str] = field(default_factory=lambda: ["device", "label"])
    min_benign_per_client: int = 200
    max_retries: int = 200


@dataclass
class ModelConfig:
    """Autoencoder architecture (input dimension is inferred from the data)."""

    type: str = "autoencoder"
    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    latent_dim: int = 16
    activation: str = "relu"
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    """Optimisation settings shared by all training modes."""

    mode: str = "federated"  # centralized | federated | local
    epochs: int = 20  # centralized / local-only training epochs
    batch_size: int = 128
    learning_rate: float = 1.0e-3
    optimizer: str = "adam"  # adam | adamw | sgd
    momentum: float = 0.9
    weight_decay: float = 0.0
    grad_clip_norm: float | None = None  # non-private gradient clipping (not DP)


@dataclass
class FederatedConfig:
    """Federated optimisation."""

    strategy: str = "fedavg"  # fedavg | fedprox
    rounds: int = 20
    local_epochs: int = 2
    fraction_fit: float = 1.0
    min_fit_clients: int = 2
    mu: float = 0.01  # FedProx proximal coefficient (ignored by FedAvg)
    server_learning_rate: float = 1.0
    eval_every: int = 1


@dataclass
class PrivacyConfig:
    """Example-level differential privacy via DP-SGD (Opacus) on each client."""

    enabled: bool = False
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    delta: float = 1.0e-5
    target_epsilon: float | None = None  # if set, calibrates the noise multiplier per client
    accountant: str = "rdp"  # rdp | prv | gdp
    poisson_sampling: bool = True
    secure_mode: bool = False  # cryptographically secure noise RNG (slower, not reproducible)


@dataclass
class SecureAggregationConfig:
    """Pairwise-masking secure aggregation (Bonawitz et al. style, simulated)."""

    enabled: bool = False
    threshold_fraction: float = 0.6  # Shamir threshold t = ceil(fraction * |U1|)
    dropout_rate: float = 0.0  # probability a client drops after key sharing
    fractional_bits: int = 24  # fixed-point precision of encoded updates
    deterministic: bool = False  # derive key material from the seed (testing only)


@dataclass
class DetectionConfig:
    """Anomaly threshold selection."""

    threshold_strategy: str = "percentile"  # percentile | validation_f1 | mean_std
    percentile: float = 95.0
    std_factor: float = 3.0
    histogram_bins: int = 4096
    histogram_min: float = 1.0e-8
    histogram_max: float = 1.0e6
    histogram_dp_epsilon: float | None = None  # Laplace noise on federated histograms


@dataclass
class IsolationForestConfig:
    enabled: bool = True
    n_estimators: int = 200
    max_samples: int | str = "auto"
    max_train_samples: int | None = 50000


@dataclass
class OneClassSVMConfig:
    enabled: bool = True
    kernel: str = "rbf"
    nu: float = 0.05
    gamma: str | float = "scale"
    max_train_samples: int = 5000


@dataclass
class BaselinesConfig:
    """Classical centralized baselines (trained on pooled benign data)."""

    isolation_forest: IsolationForestConfig = field(default_factory=IsolationForestConfig)
    one_class_svm: OneClassSVMConfig = field(default_factory=OneClassSVMConfig)


@dataclass
class EvaluationConfig:
    batch_size: int = 4096
    per_client: bool = True
    save_scores: bool = True
    max_saved_scores: int = 50000
    make_figures: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class Config:
    """Root configuration object."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    partition: PartitionConfig = field(default_factory=PartitionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    secure_aggregation: SecureAggregationConfig = field(default_factory=SecureAggregationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    baselines: BaselinesConfig = field(default_factory=BaselinesConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def copy(self) -> Config:
        return copy.deepcopy(self)

    def fingerprint(self) -> str:
        """Short stable hash of the full configuration."""
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    @property
    def raw_dir(self) -> Path:
        return resolve_path(self.data.raw_dir or f"data/raw/{self.data.dataset}")

    @property
    def processed_dir(self) -> Path:
        return resolve_path(self.data.processed_dir) / self.data.dataset

    @property
    def experiment_dir(self) -> Path:
        return resolve_path(self.experiment.output_dir) / self.experiment.name


_CHOICES: dict[str, tuple[str, ...]] = {
    "data.scaler": ("standard", "minmax"),
    "partition.strategy": ("dirichlet", "device", "iid"),
    "training.mode": ("centralized", "federated", "local"),
    "training.optimizer": ("adam", "adamw", "sgd"),
    "federated.strategy": ("fedavg", "fedprox"),
    "privacy.accountant": ("rdp", "prv", "gdp"),
    "detection.threshold_strategy": ("percentile", "validation_f1", "mean_std"),
    "model.activation": ("relu", "leaky_relu", "elu", "tanh", "gelu"),
}


def validate_config(cfg: Config) -> Config:
    """Check value ranges and categorical choices; raises :class:`ConfigError`."""
    for dotted, choices in _CHOICES.items():
        section, key = dotted.split(".")
        value = getattr(getattr(cfg, section), key)
        if value not in choices:
            raise ConfigError(f"{dotted}={value!r} is invalid; expected one of {choices}")

    def check(cond: bool, msg: str) -> None:
        if not cond:
            raise ConfigError(msg)

    d, p, f = cfg.data, cfg.partition, cfg.federated
    check(0 < d.val_fraction < 1 and 0 < d.test_fraction < 1, "split fractions must be in (0, 1)")
    check(d.val_fraction + d.test_fraction < 0.9, "val_fraction + test_fraction must be < 0.9")
    check(0.0 <= d.train_contamination <= 1.0, "data.train_contamination must be in [0, 1]")
    check(p.num_clients >= 1, "partition.num_clients must be >= 1")
    check(p.dirichlet_alpha > 0, "partition.dirichlet_alpha must be > 0")
    check(all(k in ("device", "label") for k in p.stratify_by), "stratify_by keys: device, label")
    check(len(cfg.model.hidden_dims) >= 1, "model.hidden_dims needs at least one layer")
    check(cfg.model.latent_dim >= 1, "model.latent_dim must be >= 1")
    check(cfg.training.batch_size >= 1 and cfg.training.learning_rate > 0, "bad batch/lr")
    check(f.rounds >= 1 and f.local_epochs >= 1, "federated.rounds/local_epochs must be >= 1")
    check(0 < f.fraction_fit <= 1, "federated.fraction_fit must be in (0, 1]")
    check(f.mu >= 0, "federated.mu must be >= 0")
    check(f.eval_every >= 1, "federated.eval_every must be >= 1")
    pr = cfg.privacy
    check(pr.max_grad_norm > 0, "privacy.max_grad_norm must be > 0")
    check(0 < pr.delta < 1, "privacy.delta must be in (0, 1)")
    check(pr.noise_multiplier >= 0, "privacy.noise_multiplier must be >= 0")
    check(pr.target_epsilon is None or pr.target_epsilon > 0, "target_epsilon must be > 0")
    sa = cfg.secure_aggregation
    check(0 < sa.threshold_fraction <= 1, "secure_aggregation.threshold_fraction in (0, 1]")
    check(0 <= sa.dropout_rate < 1, "secure_aggregation.dropout_rate must be in [0, 1)")
    check(8 <= sa.fractional_bits <= 40, "secure_aggregation.fractional_bits in [8, 40]")
    det = cfg.detection
    check(0 < det.percentile < 100, "detection.percentile must be in (0, 100)")
    check(0 < det.histogram_min < det.histogram_max, "invalid histogram range")
    check(det.histogram_bins >= 16, "detection.histogram_bins must be >= 16")
    return cfg


# --------------------------------------------------------------------------------------
# Loading / merging / coercion
# --------------------------------------------------------------------------------------


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_yaml_tree(path: Path, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in _stack:
        raise ConfigError(f"Circular config inheritance: {' -> '.join(map(str, _stack + (path,)))}")
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Top level of {path} must be a mapping")
    bases = data.pop("base", None)
    merged: dict[str, Any] = {}
    for base in [bases] if isinstance(bases, str) else (bases or []):
        merged = deep_merge(merged, _load_yaml_tree(path.parent / base, _stack + (path,)))
    return deep_merge(merged, data)


def apply_overrides(raw: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``key.sub=value`` overrides (values parsed as YAML scalars/lists)."""
    out = copy.deepcopy(raw)
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"Override {item!r} must have the form key.sub=value")
        key, value = item.split("=", 1)
        node = out
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"Cannot override {key!r}: {part!r} is not a section")
        node[parts[-1]] = yaml.safe_load(value)
    return out


def _coerce(value: Any, tp: Any, where: str) -> Any:
    origin = get_origin(tp)
    if tp is Any:
        return value
    if origin in (Union, types.UnionType):
        args = get_args(tp)
        if value is None and type(None) in args:
            return None
        errors = []
        for arg in (a for a in args if a is not type(None)):
            try:
                return _coerce(value, arg, where)
            except ConfigError as exc:
                errors.append(str(exc))
        raise ConfigError(f"{where}: {value!r} does not match {tp} ({'; '.join(errors)})")
    if is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping, got {type(value).__name__}")
        return _from_dict(tp, value, where)
    if origin is list:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list, got {value!r}")
        (item_tp,) = get_args(tp) or (Any,)
        return [_coerce(v, item_tp, f"{where}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping, got {value!r}")
        return dict(value)
    if tp is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false", "yes", "no"):
            return value.lower() in ("true", "yes")
        raise ConfigError(f"{where}: expected a boolean, got {value!r}")
    if tp is int:
        if isinstance(value, bool):
            raise ConfigError(f"{where}: expected an integer, got a boolean")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                as_float = float(value)
            except ValueError as exc:
                raise ConfigError(f"{where}: expected an integer, got {value!r}") from exc
            if as_float.is_integer():
                return int(as_float)
        raise ConfigError(f"{where}: expected an integer, got {value!r}")
    if tp is float:
        if isinstance(value, bool):
            raise ConfigError(f"{where}: expected a number, got a boolean")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)  # handles YAML 1.1 strings such as "1e-5"
            except ValueError as exc:
                raise ConfigError(f"{where}: expected a number, got {value!r}") from exc
        raise ConfigError(f"{where}: expected a number, got {value!r}")
    if tp is str:
        if isinstance(value, str):
            return value
        raise ConfigError(f"{where}: expected a string, got {value!r}")
    return value


def _from_dict(cls: type, data: dict[str, Any], where: str = "") -> Any:
    hints = get_type_hints(cls)
    valid = {f.name for f in fields(cls)}
    unknown = set(data) - valid
    if unknown:
        raise ConfigError(
            f"Unknown key(s) {sorted(unknown)} in section '{where or 'root'}'. "
            f"Valid keys: {sorted(valid)}"
        )
    kwargs = {
        name: _coerce(value, hints[name], f"{where}.{name}" if where else name)
        for name, value in data.items()
    }
    return cls(**kwargs)


def config_from_dict(raw: dict[str, Any]) -> Config:
    """Build and validate a :class:`Config` from a plain dictionary."""
    return validate_config(_from_dict(Config, raw))


def _find(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() or p.exists() else resolve_path(p)


def load_raw_config(
    path: str | Path | None = None,
    overrides: list[str] | None = None,
    overlays: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Merged raw dictionary: ``path`` (+ bases), then each overlay file, then overrides."""
    raw: dict[str, Any] = _load_yaml_tree(_find(path)) if path is not None else {}
    for overlay in overlays or []:
        raw = deep_merge(raw, _load_yaml_tree(_find(overlay)))
    return apply_overrides(raw, overrides)


def load_config(
    path: str | Path | None = None,
    overrides: list[str] | None = None,
    overlays: list[str | Path] | None = None,
) -> Config:
    """Load a YAML config (with ``base`` inheritance), apply overlay files and CLI overrides."""
    return config_from_dict(load_raw_config(path, overrides, overlays))


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)
