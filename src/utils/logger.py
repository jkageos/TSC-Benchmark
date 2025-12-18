"""
Unified logging configuration for TSC-Benchmark.

Provides consistent formatting and verbosity control.
"""

import logging
import sys
from pathlib import Path


def setup_logging(verbose: bool = True, log_file: Path | None = None) -> None:
    """
    Configure root logger for the project.

    Args:
        verbose: If True, show DEBUG level; else INFO level
        log_file: Optional file to write logs to
    """
    level = logging.DEBUG if verbose else logging.INFO

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # Root logger
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get logger for a specific module."""
    return logging.getLogger(name)
