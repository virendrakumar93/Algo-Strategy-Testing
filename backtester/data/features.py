"""
Feature engineering module.

Computes technical indicators using vectorized pandas/numpy operations.
All indicators are appended as columns to the existing OHLCV DataFrame.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 2.0,
) -> pd.DataFrame:
    """
    Compute Supertrend indicator using vectorized operations.

    The Supertrend is a trend-following indicator based on ATR (Average True Range).
    When price is above the Supertrend line, the trend is bullish; below it, bearish.

    Algorithm:
        1. Compute True Range and ATR(period).
        2. Compute basic upper/lower bands = HL2 ± multiplier * ATR.
        3. Iterate to compute final bands (bands only move in trend direction).
        4. Determine trend direction based on close vs. final band.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        period: ATR lookback period.
        multiplier: ATR multiplier for band width.

    Returns:
        DataFrame with added columns:
            - 'supertrend': The Supertrend value.
            - 'supertrend_direction': 1 for bullish (price > ST), -1 for bearish.
    """
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(df)

    # Step 1: True Range
    prev_close = np.empty(n)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]

    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ),
    )

    # Step 2: ATR using exponential moving average
    atr = np.empty(n)
    atr[:period] = np.nan
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    # Step 3: Basic bands
    hl2 = (high + low) / 2.0
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    # Step 4: Final bands (loop from first valid ATR index onward)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    supertrend = np.full(n, np.nan)
    direction = np.zeros(n, dtype=np.int8)

    # First valid index is period-1 (where ATR first becomes non-NaN)
    start = period - 1
    final_upper[start] = basic_upper[start]
    final_lower[start] = basic_lower[start]
    supertrend[start] = basic_upper[start]
    direction[start] = 1

    for i in range(start + 1, n):
        # Final upper band: only moves down (tightens) in downtrend
        if basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Final lower band: only moves up (tightens) in uptrend
        if basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Direction: 1 = bullish (use lower band), -1 = bearish (use upper band)
        if direction[i - 1] == 1:  # was bullish
            if close[i] < final_lower[i]:
                direction[i] = -1
                supertrend[i] = final_upper[i]
            else:
                direction[i] = 1
                supertrend[i] = final_lower[i]
        else:  # was bearish
            if close[i] > final_upper[i]:
                direction[i] = 1
                supertrend[i] = final_lower[i]
            else:
                direction[i] = -1
                supertrend[i] = final_upper[i]

    df = df.copy()
    df["supertrend"] = supertrend
    df["supertrend_direction"] = direction

    logger.info("Computed Supertrend(%d, %.1f)", period, multiplier)
    return df


def compute_previous_day_high_low(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute previous trading day's high and low for each row.

    Uses vectorized groupby operations to find daily high/low, then shifts
    by one trading day and merges back onto the minute-level DataFrame.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.

    Returns:
        DataFrame with added columns:
            - 'prev_day_high': Previous trading day's high.
            - 'prev_day_low': Previous trading day's low.
    """
    df = df.copy()
    df["_trade_date"] = df.index.date

    # Daily aggregation
    daily = df.groupby("_trade_date").agg(
        day_high=("high", "max"),
        day_low=("low", "min"),
    )

    # Shift by one trading day
    daily["prev_day_high"] = daily["day_high"].shift(1)
    daily["prev_day_low"] = daily["day_low"].shift(1)

    # Map back onto minute-level rows (preserves DatetimeIndex)
    df["prev_day_high"] = df["_trade_date"].map(daily["prev_day_high"])
    df["prev_day_low"] = df["_trade_date"].map(daily["prev_day_low"])
    df.drop(columns=["_trade_date"], inplace=True)

    logger.info("Computed previous day high/low")
    return df


def compute_all_features(
    df: pd.DataFrame,
    supertrend_period: int = 10,
    supertrend_multiplier: float = 2.0,
) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline.

    Args:
        df: Cleaned OHLCV DataFrame.
        supertrend_period: ATR period for Supertrend.
        supertrend_multiplier: ATR multiplier for Supertrend.

    Returns:
        DataFrame with all computed features appended.
    """
    logger.info("Starting feature engineering pipeline")

    df = compute_supertrend(df, period=supertrend_period, multiplier=supertrend_multiplier)
    df = compute_previous_day_high_low(df)

    # Drop rows where features are not yet available (warmup period)
    feature_cols = ["supertrend", "prev_day_high", "prev_day_low"]
    before = len(df)
    df = df.dropna(subset=feature_cols)
    dropped = before - len(df)
    if dropped > 0:
        logger.info("Dropped %d warmup rows with NaN features", dropped)

    logger.info(
        "Feature engineering complete: %d rows, columns=%s",
        len(df),
        list(df.columns),
    )
    return df
