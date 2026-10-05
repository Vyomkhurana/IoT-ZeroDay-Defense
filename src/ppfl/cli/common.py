"""Shared command-line helpers."""

from __future__ import annotations

import argparse

from ppfl.utils.config import Config, load_config


def add_config_arguments(parser: argparse.ArgumentParser, default_config: str | None = "configs/default.yaml") -> None:
    parser.add_argument("--config", "-c", default=default_config, help="YAML config file (default: %(default)s)")
    parser.add_argument("--overlay", action="append", default=[], help="extra YAML merged on top of --config (repeatable)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set federated.rounds=5 (repeatable)")


def config_from_args(args: argparse.Namespace, extra_overrides: list[str] | None = None) -> Config:
    return load_config(args.config, list(args.overrides) + list(extra_overrides or []), args.overlay)
