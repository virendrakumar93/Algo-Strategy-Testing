"""
Options strategies (evaluated on underlying price data).

1. Short Straddle (ATM)
2. Iron Condor
3. Delta Neutral (basic approximation)

Note: These strategies evaluate conditions on the underlying index/stock
and simulate options P&L using approximations (since full options chain
data requires separate API calls for each strike/expiry).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from backtester.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


def _atm_strike(price: float, interval: float = 50.0) -> float:
    """Round price to nearest strike interval."""
    return round(price / interval) * interval


# =============================================================================
# 1. Short Straddle (ATM)
# =============================================================================

class ShortStraddleStrategy(BaseStrategy):
    """
    Short Straddle strategy (sell ATM Call + ATM Put).

    Entry: Sell straddle at 09:20 each day.
    Exit:  Close at 15:15 (intraday) or when loss > 1.5x premium collected.

    P&L approximation: Premium = ATR-based estimate.
    Max profit when underlying stays near entry. Loss = |move| - premium.
    """

    name: str = "short_straddle"
    description: str = "Short Straddle (ATM, intraday)"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate short straddle signals using underlying data."""
        df = df.copy()

        entry_hour, entry_min = 9, 20
        exit_hour, exit_min = 15, 15

        # Compute daily ATR for premium estimation
        daily = df.groupby(df.index.date).agg(
            daily_high=("high", "max"),
            daily_low=("low", "min"),
            daily_close=("close", "last"),
        )
        daily["tr"] = daily["daily_high"] - daily["daily_low"]
        daily["atr_20"] = daily["tr"].rolling(20).mean()
        atr_map = dict(zip(daily.index, daily["atr_20"]))

        # Entry candles (09:20)
        entry_mask = (df.index.hour == entry_hour) & (df.index.minute == entry_min)
        entry_candles = df[entry_mask]

        # Exit candles (15:15)
        exit_mask = (df.index.hour == exit_hour) & (df.index.minute == exit_min)
        exit_candles = df[exit_mask]
        exit_map = {ts.date(): row for ts, row in exit_candles.iterrows()}

        trades: List[Dict[str, Any]] = []

        for entry_ts, entry_row in entry_candles.iterrows():
            entry_date = entry_ts.date()
            entry_price = entry_row["close"]

            atr = atr_map.get(entry_date, None)
            if atr is None or np.isnan(atr):
                continue

            # Estimated straddle premium ≈ ATR * 0.6 (rough approximation)
            premium = atr * 0.6

            exit_row = exit_map.get(entry_date)
            if exit_row is None:
                continue

            exit_ts_dt = None
            for ts in exit_candles.index:
                if ts.date() == entry_date:
                    exit_ts_dt = ts
                    break
            if exit_ts_dt is None:
                continue

            exit_price = exit_row["close"]

            # Straddle P&L: premium collected - |underlying move|
            underlying_move = abs(exit_price - entry_price)
            straddle_pnl_points = premium - underlying_move

            # Encode as a synthetic trade:
            # signal = 1 if profit, -1 if loss (for compatibility)
            # entry_price = premium (credit received)
            # exit_price = underlying_move (debit paid)
            trades.append({
                "entry_date": entry_date,
                "entry_time": entry_ts,
                "entry_price": entry_price,
                "exit_date": entry_date,
                "exit_time": exit_ts_dt,
                "exit_price": exit_price,
                "signal": 1 if straddle_pnl_points >= 0 else -1,
                "direction": "SHORT_STRADDLE",
                "premium": round(premium, 2),
                "pnl_override": round(straddle_pnl_points, 2),
            })

        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
            trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])

        logger.info("Short Straddle: %d trades generated", len(trades_df))
        return trades_df


# =============================================================================
# 2. Iron Condor
# =============================================================================

class IronCondorStrategy(BaseStrategy):
    """
    Iron Condor strategy (sell OTM strangle + buy further OTM protection).

    Entry: Sell iron condor at 09:20 each day.
    Strikes: ATM ± 1*ATR (short) and ATM ± 2*ATR (long/protection).
    Exit: Close at 15:15.

    P&L: Max profit = net premium when underlying stays between short strikes.
    Max loss = spread width - premium.
    """

    name: str = "iron_condor"
    description: str = "Iron Condor (ATR-based strikes, intraday)"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate iron condor signals."""
        df = df.copy()

        entry_hour, entry_min = 9, 20
        exit_hour, exit_min = 15, 15

        daily = df.groupby(df.index.date).agg(
            daily_high=("high", "max"),
            daily_low=("low", "min"),
            daily_close=("close", "last"),
        )
        daily["tr"] = daily["daily_high"] - daily["daily_low"]
        daily["atr_20"] = daily["tr"].rolling(20).mean()
        atr_map = dict(zip(daily.index, daily["atr_20"]))

        entry_mask = (df.index.hour == entry_hour) & (df.index.minute == entry_min)
        entry_candles = df[entry_mask]

        exit_mask = (df.index.hour == exit_hour) & (df.index.minute == exit_min)
        exit_candles = df[exit_mask]
        exit_map = {ts.date(): row for ts, row in exit_candles.iterrows()}

        trades: List[Dict[str, Any]] = []

        for entry_ts, entry_row in entry_candles.iterrows():
            entry_date = entry_ts.date()
            entry_price = entry_row["close"]

            atr = atr_map.get(entry_date, None)
            if atr is None or np.isnan(atr):
                continue

            # Short strikes: ATM ± 1*ATR
            upper_short = entry_price + atr
            lower_short = entry_price - atr
            # Net premium ≈ ATR * 0.3 (rough)
            net_premium = atr * 0.3
            # Max loss per side = ATR (spread width) - premium
            max_loss = atr - net_premium

            exit_row = exit_map.get(entry_date)
            if exit_row is None:
                continue

            exit_ts_dt = None
            for ts in exit_candles.index:
                if ts.date() == entry_date:
                    exit_ts_dt = ts
                    break
            if exit_ts_dt is None:
                continue

            exit_price = exit_row["close"]

            # P&L logic
            if lower_short <= exit_price <= upper_short:
                pnl = net_premium  # Max profit
            elif exit_price > upper_short:
                breach = exit_price - upper_short
                pnl = net_premium - min(breach, atr)
            else:
                breach = lower_short - exit_price
                pnl = net_premium - min(breach, atr)

            trades.append({
                "entry_date": entry_date,
                "entry_time": entry_ts,
                "entry_price": entry_price,
                "exit_date": entry_date,
                "exit_time": exit_ts_dt,
                "exit_price": exit_price,
                "signal": 1 if pnl >= 0 else -1,
                "direction": "IRON_CONDOR",
                "premium": round(net_premium, 2),
                "pnl_override": round(pnl, 2),
            })

        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
            trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])

        logger.info("Iron Condor: %d trades generated", len(trades_df))
        return trades_df


# =============================================================================
# 3. Delta Neutral (basic approximation)
# =============================================================================

class DeltaNeutralStrategy(BaseStrategy):
    """
    Delta Neutral strategy (simplified).

    Sells a straddle and hedges with the underlying to maintain
    approximate delta neutrality. Re-hedges when delta exceeds threshold.

    Entry: Sell straddle at 09:20 → initial delta ≈ 0.
    Re-hedge: When |delta| > 0.3 (simulated via price move > 0.3*ATR).
    Exit: 15:15.

    P&L: Theta gain - hedging costs.
    """

    name: str = "delta_neutral"
    description: str = "Delta Neutral Straddle (basic approximation)"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate delta neutral signals."""
        df = df.copy()

        entry_hour, entry_min = 9, 20
        exit_hour, exit_min = 15, 15

        daily = df.groupby(df.index.date).agg(
            daily_close=("close", "last"),
            daily_high=("high", "max"),
            daily_low=("low", "min"),
        )
        daily["tr"] = daily["daily_high"] - daily["daily_low"]
        daily["atr_20"] = daily["tr"].rolling(20).mean()
        atr_map = dict(zip(daily.index, daily["atr_20"]))

        trades: List[Dict[str, Any]] = []
        trading_dates = pd.Series(df.index.date).unique()

        for date in trading_dates:
            day_data = df[df.index.date == date]
            if len(day_data) < 30:
                continue

            atr = atr_map.get(date)
            if atr is None or np.isnan(atr):
                continue

            # Find entry and exit candles
            entry_candle = day_data[
                (day_data.index.hour == entry_hour) & (day_data.index.minute == entry_min)
            ]
            exit_candle = day_data[
                (day_data.index.hour == exit_hour) & (day_data.index.minute == exit_min)
            ]

            if entry_candle.empty or exit_candle.empty:
                continue

            entry_price = entry_candle["close"].iloc[0]
            entry_ts = entry_candle.index[0]
            exit_price = exit_candle["close"].iloc[0]
            exit_ts = exit_candle.index[0]

            # Straddle premium (theta income)
            premium = atr * 0.6

            # Count re-hedges: each time intraday move exceeds 0.3*ATR from entry
            intraday = day_data.loc[entry_ts:exit_ts]
            moves = (intraday["close"] - entry_price).abs()
            rehedge_threshold = 0.3 * atr
            n_rehedges = int((moves > rehedge_threshold).sum() * 0.1)  # Approximate

            # Hedging cost per rehedge ≈ 0.05% of entry price
            hedging_cost = n_rehedges * entry_price * 0.0005

            # Final P&L
            underlying_move = abs(exit_price - entry_price)
            pnl = premium - underlying_move * 0.5 - hedging_cost  # Partial hedge

            trades.append({
                "entry_date": date,
                "entry_time": entry_ts,
                "entry_price": entry_price,
                "exit_date": date,
                "exit_time": exit_ts,
                "exit_price": exit_price,
                "signal": 1 if pnl >= 0 else -1,
                "direction": "DELTA_NEUTRAL",
                "pnl_override": round(pnl, 2),
            })

        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
            trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])

        logger.info("Delta Neutral: %d trades generated", len(trades_df))
        return trades_df
