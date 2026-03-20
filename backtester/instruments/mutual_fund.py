"""
Mutual fund instrument handler.

Handles NAV-based data fetching, preprocessing, and analytics
for mutual fund analysis (non-backtest mode).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from backtester.instruments.base import BaseInstrument
from backtester.utils.config import AppConfig

logger = logging.getLogger(__name__)


class MutualFundInstrument(BaseInstrument):
    """
    Handler for mutual fund analysis.

    Characteristics:
        - NAV-based (not OHLCV)
        - Daily frequency only
        - No intraday trading
        - Analysis mode (not backtest): CAGR, rolling returns, drawdowns
        - XIRR for SIP/SWP analysis
    """

    instrument_type: str = "mutual_fund"
    description: str = "Mutual fund NAV-based analysis"

    def fetch_data(self, config: AppConfig, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch mutual fund NAV data.

        Attempts to load from local CSV/parquet cache. For production,
        integrate with AMFI API or Kite Connect MF module.

        Args:
            config: Application configuration.
            force_refresh: Bypass cache.

        Returns:
            DataFrame with columns: date (index), nav.
        """
        logger.info("Fetching mutual fund NAV data for %s", config.instrument.symbol)

        # Check for local data file
        data_dir = config.project_root / config.data.processed_dir
        nav_file = data_dir / f"mf_{config.instrument.symbol}_nav.csv"
        parquet_file = data_dir / f"mf_{config.instrument.symbol}_nav.parquet"

        if parquet_file.exists() and not force_refresh:
            df = pd.read_parquet(parquet_file)
            df.index = pd.to_datetime(df.index)
            logger.info("Loaded MF NAV from cache: %d rows", len(df))
            return df

        if nav_file.exists():
            df = pd.read_csv(nav_file, parse_dates=["date"], index_col="date")
            df.to_parquet(parquet_file)
            logger.info("Loaded MF NAV from CSV: %d rows", len(df))
            return df

        # If no local data, try fetching via Kite Connect MF API
        try:
            from backtester.data.fetcher import load_or_fetch_data
            df = load_or_fetch_data(config, force_refresh=force_refresh)
            # Convert OHLCV to NAV-like format (use close price)
            if "close" in df.columns:
                nav_df = pd.DataFrame({"nav": df["close"]}, index=df.index)
                # Resample to daily (take last value per day)
                nav_df = nav_df.resample("D").last().dropna()
                return nav_df
            return df
        except Exception as exc:
            logger.error("Failed to fetch MF data: %s", exc)
            raise RuntimeError(
                f"No NAV data found for {config.instrument.symbol}. "
                f"Place a CSV file at {nav_file} with columns: date, nav"
            ) from exc

    def preprocess(self, df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
        """
        Preprocess mutual fund NAV data.

        Handles:
            - Missing NAV dates (weekends, holidays) via forward-fill
            - Duplicate dates
            - Zero/negative NAV removal
        """
        df = df.copy()

        # Ensure we have a 'nav' column
        if "nav" not in df.columns:
            if "close" in df.columns:
                df["nav"] = df["close"]
            else:
                raise ValueError("NAV data must have a 'nav' or 'close' column.")

        # Remove duplicates
        df = df[~df.index.duplicated(keep="first")]

        # Sort chronologically
        df = df.sort_index()

        # Remove zero/negative NAV
        df = df[df["nav"] > 0]

        # Forward-fill missing dates (weekends/holidays)
        df = df.asfreq("B").ffill()  # Business day frequency
        df = df.dropna(subset=["nav"])

        logger.info("Preprocessed MF NAV: %d rows (%s to %s)", len(df), df.index.min(), df.index.max())
        return df

    def compute_features(self, df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
        """
        Compute mutual fund analysis features.

        Includes:
            - Daily returns
            - Rolling CAGR (1Y, 3Y, 5Y)
            - Rolling volatility
            - Drawdown series
            - Sharpe / Sortino ratios
        """
        df = df.copy()

        # Daily returns
        df["daily_return"] = df["nav"].pct_change()

        # Rolling returns (annualized)
        for years, label in [(1, "1y"), (3, "3y"), (5, "5y")]:
            window = years * 252  # trading days
            if len(df) >= window:
                df[f"rolling_return_{label}"] = (
                    (df["nav"] / df["nav"].shift(window)) ** (1 / years) - 1
                ) * 100
            else:
                df[f"rolling_return_{label}"] = np.nan

        # Rolling volatility (annualized, 1Y window)
        df["rolling_volatility_1y"] = df["daily_return"].rolling(252).std() * np.sqrt(252) * 100

        # Drawdown
        running_max = df["nav"].cummax()
        df["drawdown_pct"] = ((df["nav"] - running_max) / running_max) * 100

        # Rolling Sharpe (1Y, risk-free rate = 6% for India)
        risk_free_daily = 0.06 / 252
        excess_returns = df["daily_return"] - risk_free_daily
        df["rolling_sharpe_1y"] = (
            excess_returns.rolling(252).mean() / excess_returns.rolling(252).std()
        ) * np.sqrt(252)

        # Rolling Sortino (1Y)
        downside = df["daily_return"].copy()
        downside[downside > 0] = 0
        df["rolling_sortino_1y"] = (
            excess_returns.rolling(252).mean()
            / downside.rolling(252).std()
        ) * np.sqrt(252)

        df = df.dropna(subset=["daily_return"])

        logger.info("Computed MF features: %d rows, columns=%s", len(df), list(df.columns))
        return df
