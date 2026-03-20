"""
Index instrument handler.

Handles data fetching and preprocessing for market indices
(NIFTY 50, BANKNIFTY, etc.).
"""

from __future__ import annotations

import logging

import pandas as pd

from backtester.data.fetcher import load_or_fetch_data
from backtester.data.features import compute_all_features
from backtester.data.processor import prepare_data
from backtester.instruments.base import BaseInstrument
from backtester.utils.config import AppConfig

logger = logging.getLogger(__name__)


class IndexInstrument(BaseInstrument):
    """
    Handler for index instruments (NIFTY 50, BANKNIFTY, etc.).

    Characteristics:
        - Not directly tradable (use futures/options for trading)
        - Used as signal reference for strategy evaluation
        - No lot size in cash; futures lot size applies when trading
        - Standard OHLCV data
    """

    instrument_type: str = "index"
    description: str = "Market index (NIFTY, BANKNIFTY, etc.)"

    def fetch_data(self, config: AppConfig, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch index OHLCV data via Kite Connect."""
        logger.info("Fetching index data for %s", config.instrument.symbol)
        return load_or_fetch_data(config, force_refresh=force_refresh)

    def preprocess(self, df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
        """Clean index data (standard market hours filter, gap handling)."""
        return prepare_data(df)

    def compute_features(self, df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
        """Compute technical indicators for index data."""
        params = config.strategy.params
        return compute_all_features(
            df,
            supertrend_period=params.get("supertrend_period", 10),
            supertrend_multiplier=params.get("supertrend_multiplier", 2.0),
        )
