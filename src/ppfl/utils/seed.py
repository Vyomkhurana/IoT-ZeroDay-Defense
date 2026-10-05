"""Reproducibility helpers.

Every stochastic component draws its randomness from a seed *derived* from the
experiment seed and a component-specific key (e.g. ``("client", 3, "round", 7)``).
This keeps components independent: changing the number of rounds does not change
the data partition, adding a client does not change another client's noise, etc.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import torch


def derive_seed(base_seed: int, *keys: object) -> int:
    """Derive a 63-bit seed from a base seed and arbitrary hashable keys."""
    material = "|".join([str(base_seed), *map(str, keys)]).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little") >> 1


def numpy_rng(base_seed: int, *keys: object) -> np.random.Generator:
    return np.random.default_rng(derive_seed(base_seed, *keys))


def torch_generator(base_seed: int, *keys: object, device: str | torch.device = "cpu") -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(derive_seed(base_seed, *keys))
    return gen


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA) global generators."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
