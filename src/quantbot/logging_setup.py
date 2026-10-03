"""Logging: UTC timestamps to stdout (captured by journald on EC2) plus a rotating file."""

from __future__ import annotations

import logging
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(log_dir: str | None = None, level: str = "INFO") -> None:
    formatter = logging.Formatter(FORMAT, datefmt="%Y-%m-%dT%H:%M:%SZ")
    formatter.converter = time.gmtime
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level.upper())
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(Path(log_dir) / "bot.log", maxBytes=20_000_000, backupCount=10)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
