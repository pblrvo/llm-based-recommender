"""Shared logging configuration for the embedding pipeline.

Usage:
    from logger import Logger
    logger = Logger.get_logger(__name__)
    logger.info("...")
"""

import logging
import sys
from pathlib import Path

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "pipeline.log"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class Logger:
    """Configures console + file handlers on the root logger exactly once,
    then hands out named child loggers (one per module) that share them."""

    _configured = False

    @classmethod
    def get_logger(cls, name: str, level: int = logging.INFO, log_file: Path = LOG_FILE) -> logging.Logger:
        cls._configure_root(level, log_file)
        return logging.getLogger(name)

    @classmethod
    def _configure_root(cls, level: int, log_file: Path) -> None:
        if cls._configured:
            return

        root = logging.getLogger()
        root.setLevel(level)

        formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

        cls._configured = True
