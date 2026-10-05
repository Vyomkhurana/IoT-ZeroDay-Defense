#!/usr/bin/env python
"""Thin wrapper around :mod:`ppfl.cli.train` (see ``python scripts/train.py --help``)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppfl.cli.train import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
