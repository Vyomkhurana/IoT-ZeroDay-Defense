#!/usr/bin/env python
"""Thin wrapper around :mod:`ppfl.cli.run_experiments` (see ``python scripts/run_experiments.py --help``)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppfl.cli.run_experiments import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
