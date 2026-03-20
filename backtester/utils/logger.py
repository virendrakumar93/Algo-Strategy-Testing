"""
Centralized logging setup for the backtesting framework.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "outputs",
    strategy_name: str = "default",
    project_root: Path | None = None,
) -> logging.Logger:
    """
    Configure the root logger with console and optional file handlers.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_to_file: Whether to write logs to a file.
        log_dir: Base directory for log files.
        strategy_name: Strategy name used for log file naming.
        project_root: Project root for resolving relative paths.

    Returns:
        Configured root logger instance.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear any existing handlers to avoid duplicates on re-init
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        root = project_root or Path.cwd()
        log_path = root / log_dir / strategy_name / "logs"
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_path / f"{strategy_name}_backtest.log",
            mode="w",
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger
