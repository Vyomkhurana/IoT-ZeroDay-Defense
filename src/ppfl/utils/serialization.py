"""Serialization helpers: model state <-> flat vectors, payload sizing, JSON/YAML I/O."""

from __future__ import annotations

import dataclasses
import json
import math
import pickle
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclasses.dataclass(frozen=True)
class TensorSpec:
    """Name/shape/dtype of one tensor in a flattened state dictionary."""

    name: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 1


def state_dict_specs(state: Mapping[str, torch.Tensor]) -> list[TensorSpec]:
    return [TensorSpec(k, tuple(v.shape), str(v.dtype).replace("torch.", "")) for k, v in state.items()]


def flatten_state_dict(state: Mapping[str, torch.Tensor]) -> tuple[np.ndarray, list[TensorSpec]]:
    """Concatenate every tensor of a state dict into one float64 vector."""
    specs = state_dict_specs(state)
    if not specs:
        return np.zeros(0, dtype=np.float64), specs
    parts = [v.detach().cpu().to(torch.float64).reshape(-1).numpy() for v in state.values()]
    return np.concatenate(parts), specs


def unflatten_state_dict(vector: np.ndarray, specs: list[TensorSpec]) -> OrderedDict[str, torch.Tensor]:
    """Inverse of :func:`flatten_state_dict`."""
    expected = sum(s.numel for s in specs)
    if vector.size != expected:
        raise ValueError(f"Vector has {vector.size} elements, specs require {expected}")
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    offset = 0
    for spec in specs:
        chunk = vector[offset : offset + spec.numel].reshape(spec.shape)
        out[spec.name] = torch.as_tensor(chunk, dtype=getattr(torch, spec.dtype)).clone()
        offset += spec.numel
    return out


def payload_nbytes(obj: Any) -> int:
    """Size in bytes of an object serialized for transmission (pickle protocol 5)."""
    return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def to_jsonable(obj: Any) -> Any:
    """Convert numpy / torch / dataclass / Path objects to JSON-compatible types."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, torch.Tensor):
        return to_jsonable(obj.detach().cpu().tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(obj), fh, indent=2)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)
