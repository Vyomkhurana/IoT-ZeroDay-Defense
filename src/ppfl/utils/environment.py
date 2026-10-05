"""Hardware/software environment capture and device resolution."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from typing import Any

import torch

from ppfl.utils.config import project_root

_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "torch",
    "opacus",
    "matplotlib",
    "seaborn",
    "PyYAML",
    "pyarrow",
    "flwr",
)


def resolve_device(name: str = "auto") -> torch.device:
    """Map ``auto`` to CUDA when available, otherwise CPU; validate explicit choices."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Device {name!r} requested but CUDA is unavailable "
            f"(torch {torch.__version__}). Use experiment.device=cpu or install a CUDA build."
        )
    return device


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for pkg in _PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root(),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def collect_environment() -> dict[str, Any]:
    """Record everything needed to reproduce / contextualise timing measurements."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "packages": package_versions(),
        "git_commit": git_commit(),
    }
    if torch.cuda.is_available():
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    try:
        import psutil  # optional

        info["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        pass
    return info
