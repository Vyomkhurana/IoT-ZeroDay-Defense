"""Structured logging: human-readable console/file logs plus a JSON-lines event log."""

from __future__ import annotations

import json
import logging
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_LOGGER = "ppfl"
_TEXT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_RESERVED = set(vars(logging.makeLogRecord({})))


class JsonFormatter(logging.Formatter):
    """Format records as one JSON object per line, including ``extra=`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, default=str)


def setup_logging(
    level: str = "INFO",
    log_dir: str | Path | None = None,
) -> logging.Logger:
    """Configure the ``ppfl`` logger. Safe to call repeatedly (handlers are replaced)."""
    logger = logging.getLogger(ROOT_LOGGER)
    logger.setLevel(level.upper())
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_TEXT_FORMAT, datefmt="%H:%M:%S"))
    logger.addHandler(console)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        text = logging.FileHandler(log_dir / "train.log", mode="w", encoding="utf-8")
        text.setFormatter(logging.Formatter(_TEXT_FORMAT))
        logger.addHandler(text)
        events = logging.FileHandler(log_dir / "events.jsonl", mode="w", encoding="utf-8")
        events.setFormatter(JsonFormatter())
        logger.addHandler(events)

    # Third-party noise: Opacus warns on every wrap when the secure RNG is off.
    warnings.filterwarnings("ignore", message="Secure RNG turned off")
    warnings.filterwarnings("ignore", message="Optimal order is the largest alpha")
    warnings.filterwarnings("ignore", message=".*non-full backward hook.*")
    warnings.filterwarnings("ignore", message="Full backward hook is firing")
    logging.getLogger("opacus").setLevel(logging.ERROR)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child of the ``ppfl`` logger."""
    if name.startswith(ROOT_LOGGER):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER}.{name}")
