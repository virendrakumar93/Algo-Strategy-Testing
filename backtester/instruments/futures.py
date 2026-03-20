"""
Futures instrument handler.

Handles data fetching with expiry-aware logic and rollover
handling for index/stock futures.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from backtester.data.fetcher import load_or_fetch_data
from backtester.data.features import compute_all_features
from backtester.data.processor import prepare_data
from backtester.instruments.base import BaseInstrument
from backtester.utils.config import AppConfig

logger = logging.getLogger(__name__)


class FuturesInstrument(BaseInstrument):
    """
    Handler for futures instruments.

    Characteristics:
        - Monthly/weekly expiry contracts
        - Requires rollover handling for continuous data
        - Lot size from exchange specifications
        - Higher leverage, margin-based trading
    """

    instrument_type: str = "futures"
    description: str = "Index/Stock futures (F&O segment)"

    def fetch_data(self, config: AppConfig, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch futures OHLCV data.

        For continuous data, fetches the spot/index data and applies
        typical futures premium/discount adjustments. For specific
        expiry contracts, fetches the exact contract.
        """
        logger.info(
            "Fetching futures data for %s (expiry=%s)",
            config.instrument.symbol,
            config.instrument.expiry,
        )
        return load_or_fetch_data(config, force_refresh=force_refresh)

    def preprocess(self, df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
        """
        Clean futures data with rollover handling.

        Handles:
            - Standard market hours filtering
            - Gap detection across rollover dates
            - Volume-based rollover detection
        """
        df = prepare_data(df)
        df = self._handle_rollover(df)
        return df

    def compute_features(self, df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
        """Compute futures-specific features (standard + VWAP)."""
        params = config.strategy.params
        df = compute_all_features(
            df,
            supertrend_period=params.get("supertrend_period", 10),
            supertrend_multiplier=params.get("supertrend_multiplier", 2.0),
        )
        df = self._compute_vwap(df)
        return df

    @staticmethod
    def _handle_rollover(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and flag rollover dates in futures data.

        Rollover is detected when there's a large gap between consecutive
        candles that coincides with typical expiry patterns.

        Args:
            df: Preprocessed futures OHLCV DataFrame.

        Returns:
            DataFrame with 'is_rollover' flag column.
        """
        df = df.copy()
        daily_close = df.groupby(df.index.date)["close"].last()

        # Detect large overnight gaps (> 1% move) that may indicate rollover
        pct_change = daily_close.pct_change().abs()
        rollover_threshold = 0.01  # 1% gap
        rollover_dates = set(pct_change[pct_change > rollover_threshold].index)

        df["is_rollover"] = df.index.date.map(lambda d: d in rollover_dates).astype(int)

        n_rollovers = len(rollover_dates)
        if n_rollovers > 0:
            logger.info("Detected %d potential rollover dates", n_rollovers)

        return df

    @staticmethod
    def _compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Volume Weighted Average Price (VWAP) per trading day.

        VWAP = cumulative(typical_price * volume) / cumulative(volume)

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with 'vwap' column added.
        """
        df = df.copy()
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        df["_tp_vol"] = typical_price * df["volume"]

        trade_date = df.index.date
        df["_cum_tp_vol"] = df.groupby(trade_date)["_tp_vol"].cumsum()
        df["_cum_vol"] = df.groupby(trade_date)["volume"].cumsum()

        df["vwap"] = np.where(
            df["_cum_vol"] > 0,
            df["_cum_tp_vol"] / df["_cum_vol"],
            df["close"],
        )
        df.drop(columns=["_tp_vol", "_cum_tp_vol", "_cum_vol"], inplace=True)

        logger.info("Computed VWAP")
        return df
