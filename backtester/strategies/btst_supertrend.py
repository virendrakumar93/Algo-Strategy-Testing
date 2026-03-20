"""
BTST Supertrend Breakout Strategy implementation.

Entry Logic (at 15:28 IST):
    LONG:  LTP > Previous Day High AND Supertrend(10,2) < LTP
    SHORT: LTP < Previous Day Low  AND Supertrend(10,2) > LTP

Exit Logic:
    Exit next trading day at 09:17 IST.

No Trade:
    If neither long nor short conditions are met at 15:28.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from backtester.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

# Strategy timing constants (IST)
ENTRY_HOUR: int = 15
ENTRY_MINUTE: int = 28
EXIT_HOUR: int = 9
EXIT_MINUTE: int = 17


class BTSTSupertrendStrategy(BaseStrategy):
    """
    Buy Today, Sell Tomorrow strategy using Supertrend and previous-day
    high/low breakout signals.

    Evaluates conditions at 15:28 each trading day:
        - Long if close > prev_day_high and close > supertrend
        - Short if close < prev_day_low and close < supertrend
        - Exit at 09:17 on the next trading day
    """

    name: str = "btst_supertrend_breakout"
    description: str = "BTST Supertrend Breakout - Entry at 15:28, Exit at 09:17 next day"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate BTST Supertrend Breakout signals using vectorized operations.

        The approach:
            1. Extract unique trading dates.
            2. For each trading date, find the 15:28 candle (entry evaluation).
            3. For each entry, find the 09:17 candle on the next trading day (exit).
            4. Evaluate long/short conditions vectorized across all entry candles.

        Args:
            df: DataFrame with OHLCV, supertrend, prev_day_high, prev_day_low columns.

        Returns:
            DataFrame of trade signals with columns:
                entry_date, entry_time, entry_price, exit_date, exit_time,
                exit_price, signal, direction.
        """
        required_cols = {"close", "supertrend", "prev_day_high", "prev_day_low"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Get unique trading dates
        trading_dates = pd.Series(df.index.date).unique()
        trading_dates = np.sort(trading_dates)

        logger.info(
            "Generating signals for %d trading days (%s to %s)",
            len(trading_dates),
            trading_dates[0],
            trading_dates[-1],
        )

        trades: List[Dict[str, Any]] = []

        # Build a lookup: date -> next trading date
        next_trading_day = {}
        for i in range(len(trading_dates) - 1):
            next_trading_day[trading_dates[i]] = trading_dates[i + 1]

        # Vectorized: extract all 15:28 candles
        entry_time_filter = (df.index.hour == ENTRY_HOUR) & (df.index.minute == ENTRY_MINUTE)
        entry_candles = df[entry_time_filter].copy()

        # Vectorized: extract all 09:17 candles
        exit_time_filter = (df.index.hour == EXIT_HOUR) & (df.index.minute == EXIT_MINUTE)
        exit_candles = df[exit_time_filter].copy()
        exit_candle_map = {ts.date(): row for ts, row in exit_candles.iterrows()}

        for entry_ts, entry_row in entry_candles.iterrows():
            entry_date = entry_ts.date()

            # Find next trading day for exit
            next_day = next_trading_day.get(entry_date)
            if next_day is None:
                continue  # Last trading day, no exit possible

            # Find exit candle at 09:17 next day
            exit_row_ts = None
            for ts in exit_candle_map:
                if ts == next_day:
                    exit_row_ts = ts
                    break

            if exit_row_ts is None:
                # No 09:17 candle on next day — try nearest available candle
                next_day_data = df[df.index.date == next_day]
                if next_day_data.empty:
                    logger.debug("No data for next trading day %s, skipping", next_day)
                    continue
                # Use the first available candle as exit
                exit_row = next_day_data.iloc[0]
                exit_ts = next_day_data.index[0]
            else:
                exit_row = exit_candle_map[exit_row_ts]
                exit_ts = None
                for ts_candidate, _ in exit_candles.iterrows():
                    if ts_candidate.date() == next_day:
                        exit_ts = ts_candidate
                        break
                if exit_ts is None:
                    continue

            ltp = entry_row["close"]
            supertrend_val = entry_row["supertrend"]
            prev_high = entry_row["prev_day_high"]
            prev_low = entry_row["prev_day_low"]

            # Skip if features are NaN
            if pd.isna(prev_high) or pd.isna(prev_low) or pd.isna(supertrend_val):
                continue

            # Evaluate conditions
            long_condition = (ltp > prev_high) and (supertrend_val < ltp)
            short_condition = (ltp < prev_low) and (supertrend_val > ltp)

            if long_condition:
                trades.append({
                    "entry_date": entry_date,
                    "entry_time": entry_ts,
                    "entry_price": ltp,
                    "exit_date": exit_ts.date() if hasattr(exit_ts, "date") else next_day,
                    "exit_time": exit_ts,
                    "exit_price": exit_row["close"] if isinstance(exit_row, pd.Series) else exit_row.close,
                    "signal": 1,
                    "direction": "LONG",
                })
            elif short_condition:
                trades.append({
                    "entry_date": entry_date,
                    "entry_time": entry_ts,
                    "entry_price": ltp,
                    "exit_date": exit_ts.date() if hasattr(exit_ts, "date") else next_day,
                    "exit_time": exit_ts,
                    "exit_price": exit_row["close"] if isinstance(exit_row, pd.Series) else exit_row.close,
                    "signal": -1,
                    "direction": "SHORT",
                })
            # else: no trade for this day

        trades_df = pd.DataFrame(trades)

        if trades_df.empty:
            logger.warning("No trades generated by %s", self.name)
            return trades_df

        # Ensure proper dtypes
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
        trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])

        long_count = (trades_df["signal"] == 1).sum()
        short_count = (trades_df["signal"] == -1).sum()
        logger.info(
            "Generated %d trades: %d LONG, %d SHORT",
            len(trades_df),
            long_count,
            short_count,
        )

        return trades_df
