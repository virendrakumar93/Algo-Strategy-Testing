"""
Output management module.

Handles creation of structured, timestamped output directories and saving of
trade logs, metrics, config snapshots, and reports.

Output structure:
    outputs/{instrument_type}/{symbol}/{strategy_name}/{run_id}/
        plots/
        metrics/
        trades/
        logs/
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from backtester.analytics.metrics import metrics_to_flat_df
from backtester.utils.config import AppConfig

logger = logging.getLogger(__name__)


class OutputManager:
    """
    Manages structured output directory creation and file saving
    for backtest results with timestamped run isolation.
    """

    def __init__(self, config: AppConfig, run_id: str | None = None) -> None:
        """
        Initialize the output manager and create directory structure.

        Args:
            config: Application configuration.
            run_id: Optional run identifier. Defaults to timestamp-based ID.
        """
        self._config = config
        self._run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._strategy_name = config.strategy.name
        self._file_prefix = config.file_prefix

        # Hierarchical output dirs
        self._base_dir = config.strategy_output_dir(self._run_id)
        self._plots_dir = config.strategy_plots_dir(self._run_id)
        self._reports_dir = config.strategy_reports_dir(self._run_id)
        self._logs_dir = config.strategy_logs_dir(self._run_id)
        self._trades_dir = config.strategy_trades_dir(self._run_id)

        # Create all directories
        for d in [self._plots_dir, self._reports_dir, self._logs_dir, self._trades_dir]:
            d.mkdir(parents=True, exist_ok=True)

        logger.info("Output directories created under: %s", self._base_dir)

    @property
    def plots_dir(self) -> Path:
        """Return the plots directory path."""
        return self._plots_dir

    @property
    def run_id(self) -> str:
        """Return the unique run identifier."""
        return self._run_id

    @property
    def base_dir(self) -> Path:
        """Return the base output directory for this run."""
        return self._base_dir

    @property
    def file_prefix(self) -> str:
        """Return the standardized file naming prefix."""
        return self._file_prefix

    def save_config_snapshot(self) -> Path:
        """
        Save a copy of the current configuration as a YAML snapshot.

        Returns:
            Path to the saved config snapshot.
        """
        snapshot = {
            "run_id": self._run_id,
            "timestamp": datetime.now().isoformat(),
            "instrument": {
                "type": self._config.instrument.type,
                "symbol": self._config.instrument.symbol,
                "timeframe": self._config.instrument.timeframe,
                "expiry": self._config.instrument.expiry,
                "strike": self._config.instrument.strike,
                "option_type": self._config.instrument.option_type,
            },
            "strategy": {
                "name": self._config.strategy.name,
                "params": self._config.strategy.params,
            },
            "backtest": {
                "initial_capital": self._config.backtest.initial_capital,
                "slippage": self._config.backtest.slippage,
                "brokerage": self._config.backtest.brokerage,
                "lot_size": self._config.backtest.lot_size,
            },
            "data": {
                "years": self._config.data.years,
                "interval": self._config.data.interval,
            },
            "analysis_mode": self._config.analysis.mode,
        }

        path = self._logs_dir / f"{self._file_prefix}_config_snapshot.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(snapshot, f, default_flow_style=False, sort_keys=False)
        logger.info("Config snapshot saved: %s", path)
        return path

    def save_trade_log(self, trade_log: pd.DataFrame) -> Path:
        """
        Save the trade log as a CSV file.

        Args:
            trade_log: Per-trade results DataFrame.

        Returns:
            Path to the saved CSV file.
        """
        path = self._trades_dir / f"{self._file_prefix}_trades.csv"
        trade_log.to_csv(path, index=False)
        logger.info("Trade log saved: %s (%d trades)", path, len(trade_log))
        return path

    def save_metrics(self, metrics: Dict[str, Any]) -> tuple[Path, Path]:
        """
        Save metrics as both JSON and CSV files.

        Args:
            metrics: Full metrics dictionary.

        Returns:
            Tuple of (json_path, csv_path).
        """
        # JSON
        json_path = self._reports_dir / f"{self._file_prefix}_metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info("Metrics JSON saved: %s", json_path)

        # CSV (flattened)
        csv_path = self._reports_dir / f"{self._file_prefix}_metrics.csv"
        flat_df = metrics_to_flat_df(metrics)
        flat_df.to_csv(csv_path, index=False)
        logger.info("Metrics CSV saved: %s", csv_path)

        # Yearly breakdown CSV
        if metrics.get("yearly"):
            yearly_path = self._reports_dir / f"{self._file_prefix}_yearly.csv"
            yearly_df = pd.DataFrame(metrics["yearly"])
            yearly_df.to_csv(yearly_path, index=False)
            logger.info("Yearly metrics CSV saved: %s", yearly_path)

        return json_path, csv_path

    def save_equity_curve(self, equity_df: pd.DataFrame) -> Path:
        """
        Save the equity curve as a CSV file.

        Args:
            equity_df: Time-indexed equity curve.

        Returns:
            Path to the saved CSV file.
        """
        path = self._reports_dir / f"{self._file_prefix}_equity_curve.csv"
        equity_df.to_csv(path)
        logger.info("Equity curve saved: %s", path)
        return path

    def save_all(
        self,
        trade_log: pd.DataFrame,
        equity_df: pd.DataFrame,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Save all outputs (config snapshot, trade log, metrics, equity curve).

        Args:
            trade_log: Per-trade results.
            equity_df: Equity curve.
            metrics: Performance metrics.

        Returns:
            Dictionary of saved file paths.
        """
        paths: Dict[str, Any] = {}

        # Always save config snapshot
        paths["config_snapshot"] = str(self.save_config_snapshot())

        if self._config.output.save_trades_csv and not trade_log.empty:
            paths["trade_log"] = str(self.save_trade_log(trade_log))

        if self._config.output.save_metrics_json or self._config.output.save_metrics_csv:
            json_path, csv_path = self.save_metrics(metrics)
            paths["metrics_json"] = str(json_path)
            paths["metrics_csv"] = str(csv_path)

        if not equity_df.empty:
            paths["equity_curve"] = str(self.save_equity_curve(equity_df))

        logger.info("All outputs saved. Files: %s", list(paths.keys()))
        return paths
