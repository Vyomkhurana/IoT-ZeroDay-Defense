"""PPFL-IoT-ZeroDay: privacy-preserving federated learning for IoT zero-day threat detection.

The package is organised by concern:

* :mod:`ppfl.data`          dataset loading, cleaning, zero-day holdout and non-IID partitioning
* :mod:`ppfl.models`        autoencoder (primary model) and classical baselines
* :mod:`ppfl.training`      local training loop shared by centralized / federated / DP modes
* :mod:`ppfl.federated`     IoT clients, federated server, FedAvg / FedProx
* :mod:`ppfl.privacy`       DP-SGD (Opacus) and privacy accounting
* :mod:`ppfl.security`      pairwise-masking secure aggregation (research prototype)
* :mod:`ppfl.detection`     anomaly scoring and threshold selection
* :mod:`ppfl.evaluation`    metrics, zero-day metrics, cross-experiment comparison
* :mod:`ppfl.visualization` publication-style plots
* :mod:`ppfl.experiments`   experiment orchestration and tracking
"""

import sys

__version__ = "0.1.0"

if sys.platform == "win32":  # pragma: no cover - platform specific
    # Anaconda's MKL-linked NumPy/scikit-learn and pip-installed PyTorch each ship an
    # Intel OpenMP runtime (libiomp5md.dll). If PyTorch loads its copy first, the
    # second copy aborts the process ("OMP: Error #15"). Importing scikit-learn first
    # makes Windows load a single runtime that PyTorch then reuses.
    try:
        import sklearn  # noqa: F401
    except ImportError:
        pass
