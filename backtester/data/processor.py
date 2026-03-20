"""
Data processing and cleaning module.

Handles missing data, gaps, and produces a clean minute-level DataFrame
ready for feature engineering.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# NSE trading hours (IST)
MARKET_OPEN_HOUR: int = 9
MARKET_OPEN_MINUTE: int = 15
MARKET_CLOSE_HOUR: int = 15
MARKET_CLOSE_MINUTE: int = 30


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate raw OHLCV data.

    Operations:
        - Remove duplicates by index
        - Drop rows with NaN in critical columns
        - Filter to market hours only (09:15 - 15:30 IST)
        - Sort by datetime index
        - Validate OHLC consistency (high >= low, etc.)

    Args:
        df: Raw OHLCV DataFrame with datetime index.

    Returns:
        Cleaned DataFrame.
    """
    initial_len = len(df)
    logger.info("Cleaning OHLCV data: %d rows", initial_len)

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Remove exact duplicates
    df = df[~df.index.duplicated(keep="first")]

    # Drop rows with NaN in OHLCV
    required_cols = ["open", "high", "low", "close", "volume"]
    existing_cols = [c for c in required_cols if c in df.columns]
    df = df.dropna(subset=existing_cols)

    # Filter to market hours only
    df = _filter_market_hours(df)

    # Sort chronologically
    df = df.sort_index()

    # OHLC sanity checks: clamp high/low
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    # Remove zero or negative prices
    price_cols = ["open", "high", "low", "close"]
    valid_mask = (df[price_cols] > 0).all(axis=1)
    df = df[valid_mask]

    removed = initial_len - len(df)
    if removed > 0:
        logger.info("Removed %d invalid/out-of-hours rows", removed)

    logger.info("Clean data: %d rows, %s to %s", len(df), df.index.min(), df.index.max())
    return df


def _filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to NSE market hours (09:15 to 15:30).

    Args:
        df: DataFrame with DatetimeIndex.

    Returns:
        Filtered DataFrame containing only market-hour candles.
    """
    time_index = df.index.time
    market_open = pd.Timestamp("1900-01-01 09:15:00").time()
    market_close = pd.Timestamp("1900-01-01 15:30:00").time()
    mask = (time_index >= market_open) & (time_index <= market_close)
    return df[mask]


def detect_trading_days(df: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Extract unique trading dates from the data.

    Args:
        df: Cleaned OHLCV DataFrame.

    Returns:
        Sorted DatetimeIndex of unique trading dates.
    """
    dates = pd.DatetimeIndex(df.index.date).unique().sort_values()
    return dates


def detect_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify trading days with significantly fewer candles than expected.

    A normal trading day has ~375 minute candles (09:15 to 15:30).
    Days with less than 50% of expected candles are flagged as partial.

    Args:
        df: Cleaned OHLCV DataFrame.

    Returns:
        DataFrame with columns ['date', 'candle_count', 'is_partial'].
    """
    expected_candles = 375
    threshold = expected_candles * 0.5

    daily_counts = df.groupby(df.index.date).size().reset_index()
    daily_counts.columns = ["date", "candle_count"]
    daily_counts["is_partial"] = daily_counts["candle_count"] < threshold

    partial_days = daily_counts[daily_counts["is_partial"]]
    if len(partial_days) > 0:
        logger.warning(
            "Detected %d partial trading days (<%d candles): %s",
            len(partial_days),
            int(threshold),
            partial_days["date"].tolist()[:10],
        )

    return daily_counts


def forward_fill_gaps(df: pd.DataFrame, max_gap_minutes: int = 5) -> pd.DataFrame:
    """
    Forward-fill small intra-day gaps in minute data.

    Only fills gaps up to `max_gap_minutes` to avoid filling across
    lunch breaks or extended halts with stale data.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        max_gap_minutes: Maximum gap size (in minutes) to forward-fill.

    Returns:
        DataFrame with small gaps filled.
    """
    filled = df.asfreq("min")
    # Only fill within market hours
    filled = _filter_market_hours(filled)
    filled = filled.ffill(limit=max_gap_minutes)
    filled = filled.dropna(subset=["close"])
    logger.info("Forward-filled gaps (max %d min): %d rows", max_gap_minutes, len(filled))
    return filled


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full data preparation pipeline: clean, detect gaps, and return processed data.

    Args:
        df: Raw OHLCV DataFrame.

    Returns:
        Processed DataFrame ready for feature engineering.
    """
    df = clean_ohlcv(df)
    gap_report = detect_gaps(df)
    partial_count = gap_report["is_partial"].sum()
    if partial_count > 0:
        logger.info("Gap report: %d partial days detected", partial_count)
    return df
