"""
Configuration loader and validator for the backtesting framework.

Supports multi-instrument, multi-strategy, multi-mode configurations.
Backward-compatible with the original single-strategy config format.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class APIConfig:
    """Kite Connect API configuration."""
    api_key: str
    api_secret: str
    access_token: str
    instrument_token: int
    exchange: str
    tradingsymbol: str


@dataclass(frozen=True)
class InstrumentConfig:
    """
    Instrument specification.

    Supports equity, index, futures, options, and mutual_fund types.
    """
    type: str                          # equity | index | futures | options | mutual_fund
    symbol: str                        # e.g. RELIANCE, NIFTY, BANKNIFTY
    instrument_token: int = 256265     # Kite instrument token
    exchange: str = "NSE"
    expiry: Optional[str] = None       # For F&O: "2026-03-26"
    strike: Optional[float] = None     # For options: 22000
    option_type: Optional[str] = None  # For options: CE / PE
    timeframe: str = "minute"          # minute | day


@dataclass(frozen=True)
class DataConfig:
    """Data fetching and storage configuration."""
    years: int
    interval: str
    raw_dir: str
    processed_dir: str
    enable_cache: bool
    cache_expiry_hours: int


@dataclass(frozen=True)
class StrategyConfig:
    """Strategy selection and parameters."""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisConfig:
    """Analysis mode configuration."""
    mode: str = "backtest"   # backtest | analysis | portfolio


@dataclass(frozen=True)
class BacktestConfig:
    """Backtesting parameters."""
    initial_capital: float
    slippage: float
    brokerage: float
    lot_size: int


@dataclass(frozen=True)
class OutputConfig:
    """Output paths and flags."""
    base_dir: str
    save_trades_csv: bool
    save_metrics_json: bool
    save_metrics_csv: bool
    save_plots_html: bool


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration."""
    level: str
    log_to_file: bool
    log_dir: str


@dataclass(frozen=True)
class AppConfig:
    """
    Top-level application configuration container.

    Extended to support instrument abstraction, analysis modes, and
    hierarchical output paths while remaining backward-compatible.
    """
    api: APIConfig
    data: DataConfig
    strategy: StrategyConfig
    backtest: BacktestConfig
    output: OutputConfig
    logging: LoggingConfig
    instrument: InstrumentConfig
    analysis: AnalysisConfig = field(default_factory=lambda: AnalysisConfig())
    project_root: Path = field(default_factory=lambda: Path.cwd())

    # ---- run-level identifiers ----
    @property
    def run_id(self) -> str:
        """Unique run identifier based on timestamp."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- hierarchical output paths (NEW) ----
    def strategy_output_dir(self, run_id: Optional[str] = None) -> Path:
        """
        Return the output directory for the current instrument + strategy + run.

        Structure: outputs/{instrument_type}/{symbol}/{strategy}/{run_id}/
        Falls back to flat structure if run_id is None.
        """
        base = (
            self.project_root
            / self.output.base_dir
            / self.instrument.type
            / self.instrument.symbol
            / self.strategy.name
        )
        if run_id:
            return base / run_id
        return base

    def strategy_plots_dir(self, run_id: Optional[str] = None) -> Path:
        """Return the plots sub-directory."""
        return self.strategy_output_dir(run_id) / "plots"

    def strategy_reports_dir(self, run_id: Optional[str] = None) -> Path:
        """Return the reports / metrics sub-directory."""
        return self.strategy_output_dir(run_id) / "metrics"

    def strategy_logs_dir(self, run_id: Optional[str] = None) -> Path:
        """Return the logs sub-directory."""
        return self.strategy_output_dir(run_id) / "logs"

    def strategy_trades_dir(self, run_id: Optional[str] = None) -> Path:
        """Return the trades sub-directory."""
        return self.strategy_output_dir(run_id) / "trades"

    @property
    def file_prefix(self) -> str:
        """
        Standard file naming prefix encoding instrument, symbol, strategy, timeframe.

        Example: equity_RELIANCE_rsi_mean_reversion_minute
        """
        return (
            f"{self.instrument.type}_{self.instrument.symbol}"
            f"_{self.strategy.name}_{self.instrument.timeframe}"
        )


def load_config(config_path: str | Path, strategy_override: str | None = None) -> AppConfig:
    """
    Load and validate configuration from a YAML file.

    Backward-compatible: works with both the original flat config format
    and the new multi-instrument format.

    Args:
        config_path: Path to the config.yaml file.
        strategy_override: If provided, overrides the strategy name from config.

    Returns:
        Fully validated AppConfig instance.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If required configuration keys are missing.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    # Allow environment variable overrides for sensitive fields
    api_raw = raw.get("api", {})
    api_key = os.environ.get("KITE_API_KEY", api_raw.get("api_key", ""))
    api_secret = os.environ.get("KITE_API_SECRET", api_raw.get("api_secret", ""))
    access_token = os.environ.get("KITE_ACCESS_TOKEN", api_raw.get("access_token", ""))

    project_root = config_path.parent.resolve()

    # Strategy (support both flat and nested)
    strat_raw = raw.get("strategy", {})
    strategy_name = strategy_override or strat_raw.get("name", "")
    if not strategy_name:
        raise ValueError("Strategy name must be provided via config or --strategy flag.")
    strategy_params = strat_raw.get("params", {}) or {}

    data_raw = raw.get("data", {})
    bt_raw = raw.get("backtest", raw.get("execution", {}))
    out_raw = raw.get("output", {})
    log_raw = raw.get("logging", {})

    # Instrument config (NEW — backward-compatible)
    inst_raw = raw.get("instrument", {})
    inst_type = inst_raw.get("type", "index")
    inst_symbol = inst_raw.get("symbol", api_raw.get("tradingsymbol", "NIFTY 50"))
    inst_token = inst_raw.get("instrument_token", api_raw.get("instrument_token", 256265))

    # Analysis config (NEW)
    analysis_raw = raw.get("analysis", {})

    config = AppConfig(
        api=APIConfig(
            api_key=api_key,
            api_secret=api_secret,
            access_token=access_token,
            instrument_token=inst_token,
            exchange=inst_raw.get("exchange", api_raw.get("exchange", "NSE")),
            tradingsymbol=inst_symbol,
        ),
        data=DataConfig(
            years=data_raw.get("years", 5),
            interval=data_raw.get("interval", inst_raw.get("timeframe", "minute")),
            raw_dir=data_raw.get("raw_dir", "data/raw"),
            processed_dir=data_raw.get("processed_dir", "data/processed"),
            enable_cache=data_raw.get("enable_cache", True),
            cache_expiry_hours=data_raw.get("cache_expiry_hours", 24),
        ),
        strategy=StrategyConfig(
            name=strategy_name,
            params=strategy_params,
        ),
        backtest=BacktestConfig(
            initial_capital=float(bt_raw.get("initial_capital", bt_raw.get("capital", 1_000_000))),
            slippage=float(bt_raw.get("slippage", 0.5)),
            brokerage=float(bt_raw.get("brokerage", 20.0)),
            lot_size=int(bt_raw.get("lot_size", 50)),
        ),
        output=OutputConfig(
            base_dir=out_raw.get("base_dir", "outputs"),
            save_trades_csv=out_raw.get("save_trades_csv", True),
            save_metrics_json=out_raw.get("save_metrics_json", True),
            save_metrics_csv=out_raw.get("save_metrics_csv", True),
            save_plots_html=out_raw.get("save_plots_html", True),
        ),
        logging=LoggingConfig(
            level=log_raw.get("level", "INFO"),
            log_to_file=log_raw.get("log_to_file", True),
            log_dir=log_raw.get("log_dir", "outputs"),
        ),
        instrument=InstrumentConfig(
            type=inst_type,
            symbol=inst_symbol,
            instrument_token=inst_token,
            exchange=inst_raw.get("exchange", api_raw.get("exchange", "NSE")),
            expiry=inst_raw.get("expiry"),
            strike=inst_raw.get("strike"),
            option_type=inst_raw.get("option_type"),
            timeframe=inst_raw.get("timeframe", data_raw.get("interval", "minute")),
        ),
        analysis=AnalysisConfig(
            mode=analysis_raw.get("mode", "backtest"),
        ),
        project_root=project_root,
    )

    logger.info("Configuration loaded from %s", config_path)
    return config
