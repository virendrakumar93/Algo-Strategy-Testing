"""
Equity / Index strategies.

1. Moving Average Crossover (50/200)
2. RSI Mean Reversion
3. Donchian Breakout (High-Low Channel)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from backtester.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


# =============================================================================
# Shared indicator helpers (vectorized)
# =============================================================================

def _ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (vectorized Wilder's method)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# =============================================================================
# 1. Moving Average Crossover (50/200)
# =============================================================================

class MACrossoverStrategy(BaseStrategy):
    """
    Golden Cross / Death Cross strategy.

    LONG:  SMA(50) crosses above SMA(200)  → buy at next open, hold until exit.
    SHORT: SMA(50) crosses below SMA(200)  → short at next open, hold until exit.
    EXIT:  Opposite crossover signal.

    Uses daily close for signal, next open for execution.
    """

    name: str = "ma_crossover_50_200"
    description: str = "Moving Average Crossover (SMA 50/200)"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate crossover signals using vectorized operations."""
        df = df.copy()

        fast_period = 50
        slow_period = 200

        # Compute daily close series for MA calculation
        daily = df.groupby(df.index.date).agg(
            daily_open=("open", "first"),
            daily_close=("close", "last"),
        )
        daily.index = pd.to_datetime(daily.index)

        daily["sma_fast"] = _sma(daily["daily_close"], fast_period)
        daily["sma_slow"] = _sma(daily["daily_close"], slow_period)
        daily = daily.dropna()

        # Detect crossovers
        daily["fast_above"] = (daily["sma_fast"] > daily["sma_slow"]).astype(int)
        daily["crossover"] = daily["fast_above"].diff()
        # crossover = 1 → golden cross (long), crossover = -1 → death cross (short)

        trades: List[Dict[str, Any]] = []
        position = 0  # 0=flat, 1=long, -1=short
        entry_time = None
        entry_price = 0.0

        dates = daily.index.tolist()
        for i in range(1, len(dates)):
            cross = daily["crossover"].iloc[i]
            current_date = dates[i]

            if cross == 1 and position <= 0:
                # Close short if open
                if position == -1 and entry_time is not None:
                    trades.append({
                        "entry_date": entry_time.date(),
                        "entry_time": entry_time,
                        "entry_price": entry_price,
                        "exit_date": current_date.date(),
                        "exit_time": current_date,
                        "exit_price": daily["daily_open"].iloc[i],
                        "signal": -1,
                        "direction": "SHORT",
                    })
                # Open long
                entry_time = current_date
                entry_price = daily["daily_open"].iloc[i]
                position = 1

            elif cross == -1 and position >= 0:
                # Close long if open
                if position == 1 and entry_time is not None:
                    trades.append({
                        "entry_date": entry_time.date(),
                        "entry_time": entry_time,
                        "entry_price": entry_price,
                        "exit_date": current_date.date(),
                        "exit_time": current_date,
                        "exit_price": daily["daily_open"].iloc[i],
                        "signal": 1,
                        "direction": "LONG",
                    })
                # Open short
                entry_time = current_date
                entry_price = daily["daily_open"].iloc[i]
                position = -1

        # Close final open position
        if position != 0 and entry_time is not None:
            last_date = dates[-1]
            direction = "LONG" if position == 1 else "SHORT"
            trades.append({
                "entry_date": entry_time.date(),
                "entry_time": entry_time,
                "entry_price": entry_price,
                "exit_date": last_date.date(),
                "exit_time": last_date,
                "exit_price": daily["daily_close"].iloc[-1],
                "signal": position,
                "direction": direction,
            })

        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
            trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])

        logger.info("MA Crossover: %d trades generated", len(trades_df))
        return trades_df


# =============================================================================
# 2. RSI Mean Reversion
# =============================================================================

class RSIMeanReversionStrategy(BaseStrategy):
    """
    RSI Mean Reversion strategy.

    LONG:  RSI(14) < 30 (oversold) → buy at close.
    EXIT LONG: RSI(14) > 50 → sell at close.
    SHORT: RSI(14) > 70 (overbought) → short at close.
    EXIT SHORT: RSI(14) < 50 → cover at close.
    """

    name: str = "rsi_mean_reversion"
    description: str = "RSI Mean Reversion (14-period, 30/70 thresholds)"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate RSI mean reversion signals."""
        df = df.copy()

        # Daily aggregation
        daily = df.groupby(df.index.date).agg(
            daily_close=("close", "last"),
        )
        daily.index = pd.to_datetime(daily.index)

        rsi_period = 14
        oversold = 30
        overbought = 70
        neutral = 50

        daily["rsi"] = _rsi(daily["daily_close"], rsi_period)
        daily = daily.dropna()

        trades: List[Dict[str, Any]] = []
        position = 0
        entry_time = None
        entry_price = 0.0

        dates = daily.index.tolist()
        for i in range(len(dates)):
            rsi_val = daily["rsi"].iloc[i]
            price = daily["daily_close"].iloc[i]
            current_date = dates[i]

            if position == 0:
                if rsi_val < oversold:
                    position = 1
                    entry_time = current_date
                    entry_price = price
                elif rsi_val > overbought:
                    position = -1
                    entry_time = current_date
                    entry_price = price
            elif position == 1 and rsi_val > neutral:
                trades.append({
                    "entry_date": entry_time.date(),
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "exit_date": current_date.date(),
                    "exit_time": current_date,
                    "exit_price": price,
                    "signal": 1,
                    "direction": "LONG",
                })
                position = 0
                entry_time = None
            elif position == -1 and rsi_val < neutral:
                trades.append({
                    "entry_date": entry_time.date(),
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "exit_date": current_date.date(),
                    "exit_time": current_date,
                    "exit_price": price,
                    "signal": -1,
                    "direction": "SHORT",
                })
                position = 0
                entry_time = None

        # Close final open position
        if position != 0 and entry_time is not None:
            last_date = dates[-1]
            trades.append({
                "entry_date": entry_time.date(),
                "entry_time": entry_time,
                "entry_price": entry_price,
                "exit_date": last_date.date(),
                "exit_time": last_date,
                "exit_price": daily["daily_close"].iloc[-1],
                "signal": position,
                "direction": "LONG" if position == 1 else "SHORT",
            })

        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
            trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])

        logger.info("RSI Mean Reversion: %d trades generated", len(trades_df))
        return trades_df


# =============================================================================
# 3. Donchian Breakout
# =============================================================================

class DonchianBreakoutStrategy(BaseStrategy):
    """
    Donchian Channel Breakout strategy.

    LONG:  Close breaks above the 20-day high → buy.
    SHORT: Close breaks below the 20-day low  → short.
    EXIT:  Opposite breakout or 10-day channel exit.
    """

    name: str = "donchian_breakout"
    description: str = "Donchian Channel Breakout (20-period)"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate Donchian channel breakout signals."""
        df = df.copy()

        entry_period = 20
        exit_period = 10

        daily = df.groupby(df.index.date).agg(
            daily_open=("open", "first"),
            daily_high=("high", "max"),
            daily_low=("low", "min"),
            daily_close=("close", "last"),
        )
        daily.index = pd.to_datetime(daily.index)

        # Entry channel
        daily["upper_entry"] = daily["daily_high"].rolling(entry_period).max()
        daily["lower_entry"] = daily["daily_low"].rolling(entry_period).min()
        # Exit channel
        daily["upper_exit"] = daily["daily_high"].rolling(exit_period).max().shift(1)
        daily["lower_exit"] = daily["daily_low"].rolling(exit_period).min().shift(1)

        daily = daily.dropna()

        trades: List[Dict[str, Any]] = []
        position = 0
        entry_time = None
        entry_price = 0.0

        dates = daily.index.tolist()
        for i in range(1, len(dates)):
            prev = daily.iloc[i - 1]
            curr = daily.iloc[i]
            current_date = dates[i]

            if position == 0:
                # Long breakout
                if curr["daily_close"] > prev["upper_entry"]:
                    position = 1
                    entry_time = current_date
                    entry_price = curr["daily_close"]
                # Short breakout
                elif curr["daily_close"] < prev["lower_entry"]:
                    position = -1
                    entry_time = current_date
                    entry_price = curr["daily_close"]
            elif position == 1:
                # Exit long on lower exit channel breach
                if curr["daily_close"] < curr["lower_exit"]:
                    trades.append({
                        "entry_date": entry_time.date(),
                        "entry_time": entry_time,
                        "entry_price": entry_price,
                        "exit_date": current_date.date(),
                        "exit_time": current_date,
                        "exit_price": curr["daily_close"],
                        "signal": 1,
                        "direction": "LONG",
                    })
                    position = 0
                    entry_time = None
            elif position == -1:
                # Exit short on upper exit channel breach
                if curr["daily_close"] > curr["upper_exit"]:
                    trades.append({
                        "entry_date": entry_time.date(),
                        "entry_time": entry_time,
                        "entry_price": entry_price,
                        "exit_date": current_date.date(),
                        "exit_time": current_date,
                        "exit_price": curr["daily_close"],
                        "signal": -1,
                        "direction": "SHORT",
                    })
                    position = 0
                    entry_time = None

        # Close final open position
        if position != 0 and entry_time is not None:
            last_date = dates[-1]
            trades.append({
                "entry_date": entry_time.date(),
                "entry_time": entry_time,
                "entry_price": entry_price,
                "exit_date": last_date.date(),
                "exit_time": last_date,
                "exit_price": daily["daily_close"].iloc[-1],
                "signal": position,
                "direction": "LONG" if position == 1 else "SHORT",
            })

        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
            trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])

        logger.info("Donchian Breakout: %d trades generated", len(trades_df))
        return trades_df
