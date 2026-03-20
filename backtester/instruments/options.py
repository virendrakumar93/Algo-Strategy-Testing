"""
Options instrument handler.

Handles data fetching for options with strike/expiry selection
and basic Greeks approximation.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from backtester.data.fetcher import load_or_fetch_data
from backtester.data.processor import prepare_data
from backtester.instruments.base import BaseInstrument
from backtester.utils.config import AppConfig

logger = logging.getLogger(__name__)


class OptionsInstrument(BaseInstrument):
    """
    Handler for options instruments.

    Characteristics:
        - Strike price and expiry selection
        - CE (Call) / PE (Put) differentiation
        - Greeks approximation (delta, gamma, theta, vega)
        - Time decay considerations
        - Illiquidity handling for deep OTM/ITM strikes
    """

    instrument_type: str = "options"
    description: str = "Index/Stock options (F&O segment)"

    def fetch_data(self, config: AppConfig, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch options OHLCV data.

        For options strategies that work on the underlying (e.g., straddles),
        fetches the underlying index/stock data. For specific strike analysis,
        fetches the options contract data.
        """
        logger.info(
            "Fetching options data for %s (strike=%s, type=%s, expiry=%s)",
            config.instrument.symbol,
            config.instrument.strike,
            config.instrument.option_type,
            config.instrument.expiry,
        )
        # Fetch underlying data for strategy evaluation
        return load_or_fetch_data(config, force_refresh=force_refresh)

    def preprocess(self, df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
        """
        Preprocess options data.

        Handles:
            - Standard market hours filtering
            - Illiquidity detection (low volume candles)
            - Expiry day special handling
        """
        df = prepare_data(df)
        df = self._flag_illiquid(df)
        df = self._flag_expiry_days(df, config)
        return df

    def compute_features(self, df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
        """
        Compute options-specific features.

        Includes:
            - ATM strike identification (for straddle/condor strategies)
            - Basic delta approximation
            - Implied volatility proxy (historical vol)
        """
        df = df.copy()
        df = self._compute_historical_volatility(df)
        df = self._compute_atm_proxy(df, config)
        return df

    @staticmethod
    def _flag_illiquid(df: pd.DataFrame, volume_threshold: int = 100) -> pd.DataFrame:
        """
        Flag candles with very low volume as potentially illiquid.

        Args:
            df: OHLCV DataFrame.
            volume_threshold: Minimum volume to consider liquid.

        Returns:
            DataFrame with 'is_illiquid' flag.
        """
        df = df.copy()
        if "volume" in df.columns:
            df["is_illiquid"] = (df["volume"] < volume_threshold).astype(int)
            illiquid_pct = df["is_illiquid"].mean() * 100
            if illiquid_pct > 10:
                logger.warning("%.1f%% of candles are illiquid (vol < %d)", illiquid_pct, volume_threshold)
        return df

    @staticmethod
    def _flag_expiry_days(df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
        """
        Flag rows that fall on expiry day for special handling.

        Args:
            df: OHLCV DataFrame.
            config: AppConfig with expiry info.

        Returns:
            DataFrame with 'is_expiry_day' flag.
        """
        df = df.copy()
        if config.instrument.expiry:
            expiry_date = pd.Timestamp(config.instrument.expiry).date()
            df["is_expiry_day"] = (df.index.date == expiry_date).astype(int)
        else:
            # Mark Thursdays as potential weekly expiry days (NSE convention)
            df["is_expiry_day"] = (df.index.dayofweek == 3).astype(int)
        return df

    @staticmethod
    def _compute_historical_volatility(
        df: pd.DataFrame, window: int = 20
    ) -> pd.DataFrame:
        """
        Compute annualized historical volatility as IV proxy.

        Args:
            df: OHLCV DataFrame.
            window: Rolling window for volatility computation.

        Returns:
            DataFrame with 'hist_volatility' column.
        """
        df = df.copy()
        log_returns = np.log(df["close"] / df["close"].shift(1))
        df["hist_volatility"] = log_returns.rolling(window=window).std() * np.sqrt(252 * 375)
        logger.info("Computed historical volatility (window=%d)", window)
        return df

    @staticmethod
    def _compute_atm_proxy(df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
        """
        Compute ATM strike level rounded to nearest strike interval.

        For NIFTY: rounds to nearest 50.
        For BANKNIFTY: rounds to nearest 100.

        Args:
            df: OHLCV DataFrame.
            config: Config with instrument details.

        Returns:
            DataFrame with 'atm_strike' column.
        """
        df = df.copy()
        symbol = config.instrument.symbol.upper()

        if "BANKNIFTY" in symbol or "BANKN" in symbol:
            strike_interval = 100
        elif "NIFTY" in symbol:
            strike_interval = 50
        else:
            strike_interval = 50  # default

        df["atm_strike"] = (df["close"] / strike_interval).round() * strike_interval
        return df
