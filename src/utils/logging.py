"""
Centralized logging configuration using loguru.

Call `setup_logging()` once at application entry points
(API startup, training scripts, etc.).
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """
    Configure loguru for the application.

    Parameters
    ----------
    level:
        Logging verbosity level (DEBUG, INFO, WARNING, ERROR).
    log_file:
        Optional path for file-based logging. Falls back to logs/app.log.
    rotation:
        File rotation threshold (e.g. "10 MB", "1 week").
    retention:
        How long to keep rotated files.
    """
    logger.remove()  # remove default handler

    # Console handler — human-readable colourised output
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = log_file or str(LOG_DIR / "app.log")
    logger.add(
        log_path,
        level=level,
        rotation=rotation,
        retention=retention,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
        encoding="utf-8",
    )

    logger.info(f"Logging initialised — level={level}, file={log_path}")
