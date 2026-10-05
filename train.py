#!/usr/bin/env python
"""Convenience entry point: ``python train.py --config configs/full_ppfl.yaml``.

Equivalent to ``python scripts/train.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ppfl.cli.train import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
