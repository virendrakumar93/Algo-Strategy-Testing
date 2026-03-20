"""
Futures strategies.

1. Trend Following (EMA + Supertrend)
2. VWAP Reversion
3. Open Interest + Price Action
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from backtester.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


def _ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


# =============================================================================
# 1. Trend Following (EMA + Supertrend)
# =============================================================================

class EMASuperTrendStrategy(BaseStrategy):
    """
    Trend Following strategy combining EMA and Supertrend.

    LONG:  EMA(21) > EMA(55) AND close > Supertrend → buy at close.
    SHORT: EMA(21) < EMA(55) AND close < Supertrend → short at close.
    EXIT:  Supertrend direction reversal.
    """

    name: str = "ema_supertrend_trend"
    description: str = "EMA + Supertrend Trend Following"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trend following signals using EMA alignment + Supertrend."""
        if "supertrend" not in df.columns or "supertrend_direction" not in df.columns:
            raise ValueError("Supertrend columns required. Run feature engineering first.")

        df = df.copy()

        # Daily aggregation with supertrend
        daily = df.groupby(df.index.date).agg(
            daily_open=("open", "first"),
            daily_close=("close", "last"),
            st_dir=("supertrend_direction", "last"),
            st_val=("supertrend", "last"),
        )
        daily.index = pd.to_datetime(daily.index)

        daily["ema_fast"] = _ema(daily["daily_close"], 21)
        daily["ema_slow"] = _ema(daily["daily_close"], 55)
        daily = daily.dropna()

        trades: List[Dict[str, Any]] = []
        position = 0
        entry_time = None
        entry_price = 0.0

        dates = daily.index.tolist()
        for i in range(1, len(dates)):
            curr = daily.iloc[i]
            current_date = dates[i]

            ema_bullish = curr["ema_fast"] > curr["ema_slow"]
            st_bullish = curr["st_dir"] == 1

            if position == 0:
                if ema_bullish and st_bullish:
                    position = 1
                    entry_time = current_date
                    entry_price = curr["daily_close"]
                elif not ema_bullish and not st_bullish:
                    position = -1
                    entry_time = current_date
                    entry_price = curr["daily_close"]

            elif position == 1:
                if not st_bullish:
                    trades.append({
                        "entry_date": entry_time.date(), "entry_time": entry_time,
                        "entry_price": entry_price, "exit_date": current_date.date(),
                        "exit_time": current_date, "exit_price": curr["daily_close"],
                        "signal": 1, "direction": "LONG",
                    })
                    position = 0
                    entry_time = None

            elif position == -1:
                if st_bullish:
                    trades.append({
                        "entry_date": entry_time.date(), "entry_time": entry_time,
                        "entry_price": entry_price, "exit_date": current_date.date(),
                        "exit_time": current_date, "exit_price": curr["daily_close"],
                        "signal": -1, "direction": "SHORT",
                    })
                    position = 0
                    entry_time = None

        if position != 0 and entry_time is not None:
            trades.append({
                "entry_date": entry_time.date(), "entry_time": entry_time,
                "entry_price": entry_price, "exit_date": dates[-1].date(),
                "exit_time": dates[-1], "exit_price": daily["daily_close"].iloc[-1],
                "signal": position, "direction": "LONG" if position == 1 else "SHORT",
            })

        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
            trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])

        logger.info("EMA+Supertrend: %d trades generated", len(trades_df))
        return trades_df


# =============================================================================
# 2. VWAP Reversion
# =============================================================================

class VWAPReversionStrategy(BaseStrategy):
    """
    Intraday VWAP Reversion strategy.

    LONG:  Price drops > 0.5% below VWAP → buy, exit when price returns to VWAP.
    SHORT: Price rises > 0.5% above VWAP → short, exit when price returns to VWAP.

    Requires 'vwap' column (computed by FuturesInstrument).
    Falls back to computing VWAP inline if not present.
    """

    name: str = "vwap_reversion"
    description: str = "VWAP Mean Reversion (intraday)"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate VWAP reversion signals."""
        df = df.copy()

        # Compute VWAP if missing
        if "vwap" not in df.columns:
            typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
            tp_vol = typical_price * df["volume"]
            trade_date = df.index.date
            cum_tp_vol = df.groupby(trade_date).apply(
                lambda g: tp_vol.loc[g.index].cumsum()
            )
            if hasattr(cum_tp_vol, "droplevel"):
                cum_tp_vol = cum_tp_vol.droplevel(0)
            cum_vol = df.groupby(trade_date)["volume"].cumsum()
            df["vwap"] = np.where(cum_vol > 0, cum_tp_vol / cum_vol, df["close"])

        deviation_threshold = 0.005  # 0.5%

        trades: List[Dict[str, Any]] = []
        position = 0
        entry_time = None
        entry_price = 0.0
        entry_signal = 0

        trading_dates = pd.Series(df.index.date).unique()

        for date in trading_dates:
            day_data = df[df.index.date == date]
            if len(day_data) < 30:
                continue

            # Reset position at start of each day (intraday strategy)
            if position != 0 and entry_time is not None:
                # Force close at previous day end
                pass

            position = 0
            entry_time = None

            for ts, row in day_data.iloc[15:].iterrows():  # Skip first 15 min
                vwap = row["vwap"]
                price = row["close"]
                deviation = (price - vwap) / vwap

                if position == 0:
                    if deviation < -deviation_threshold:
                        position = 1
                        entry_time = ts
                        entry_price = price
                        entry_signal = 1
                    elif deviation > deviation_threshold:
                        position = -1
                        entry_time = ts
                        entry_price = price
                        entry_signal = -1
                elif position == 1 and price >= vwap:
                    trades.append({
                        "entry_date": entry_time.date(), "entry_time": entry_time,
                        "entry_price": entry_price, "exit_date": ts.date(),
                        "exit_time": ts, "exit_price": price,
                        "signal": 1, "direction": "LONG",
                    })
                    position = 0
                    entry_time = None
                elif position == -1 and price <= vwap:
                    trades.append({
                        "entry_date": entry_time.date(), "entry_time": entry_time,
                        "entry_price": entry_price, "exit_date": ts.date(),
                        "exit_time": ts, "exit_price": price,
                        "signal": -1, "direction": "SHORT",
                    })
                    position = 0
                    entry_time = None

            # Force close at day end
            if position != 0 and entry_time is not None:
                last_row = day_data.iloc[-1]
                trades.append({
                    "entry_date": entry_time.date(), "entry_time": entry_time,
                    "entry_price": entry_price, "exit_date": day_data.index[-1].date(),
                    "exit_time": day_data.index[-1], "exit_price": last_row["close"],
                    "signal": entry_signal, "direction": "LONG" if entry_signal == 1 else "SHORT",
                })
                position = 0
                entry_time = None

        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
            trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])

        logger.info("VWAP Reversion: %d trades generated", len(trades_df))
        return trades_df


# =============================================================================
# 3. Open Interest + Price Action
# =============================================================================

class OIPriceActionStrategy(BaseStrategy):
    """
    Open Interest + Price Action strategy.

    Uses volume as a proxy for OI change:
        LONG:  Price up + volume surge (> 2x avg) → bullish OI buildup → buy.
        SHORT: Price down + volume surge → bearish OI buildup → short.
        EXIT:  Volume drops below average or opposite signal.

    Note: Actual OI data requires separate API calls. This uses volume
    as an approximation suitable for backtesting.
    """

    name: str = "oi_price_action"
    description: str = "Open Interest + Price Action (volume proxy)"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate OI + price action signals."""
        df = df.copy()

        daily = df.groupby(df.index.date).agg(
            daily_open=("open", "first"),
            daily_close=("close", "last"),
            daily_volume=("volume", "sum"),
        )
        daily.index = pd.to_datetime(daily.index)

        # Volume moving average (20-day)
        daily["vol_ma_20"] = daily["daily_volume"].rolling(20).mean()
        daily["vol_ratio"] = daily["daily_volume"] / daily["vol_ma_20"]
        daily["price_change"] = daily["daily_close"].pct_change()

        daily = daily.dropna()

        volume_surge_threshold = 2.0  # 2x average volume

        trades: List[Dict[str, Any]] = []
        position = 0
        entry_time = None
        entry_price = 0.0

        dates = daily.index.tolist()
        for i in range(len(dates)):
            curr = daily.iloc[i]
            current_date = dates[i]

            is_surge = curr["vol_ratio"] > volume_surge_threshold

            if position == 0:
                if is_surge and curr["price_change"] > 0.005:  # Up + volume
                    position = 1
                    entry_time = current_date
                    entry_price = curr["daily_close"]
                elif is_surge and curr["price_change"] < -0.005:  # Down + volume
                    position = -1
                    entry_time = current_date
                    entry_price = curr["daily_close"]
            elif position == 1:
                if curr["vol_ratio"] < 0.8 or curr["price_change"] < -0.01:
                    trades.append({
                        "entry_date": entry_time.date(), "entry_time": entry_time,
                        "entry_price": entry_price, "exit_date": current_date.date(),
                        "exit_time": current_date, "exit_price": curr["daily_close"],
                        "signal": 1, "direction": "LONG",
                    })
                    position = 0
                    entry_time = None
            elif position == -1:
                if curr["vol_ratio"] < 0.8 or curr["price_change"] > 0.01:
                    trades.append({
                        "entry_date": entry_time.date(), "entry_time": entry_time,
                        "entry_price": entry_price, "exit_date": current_date.date(),
                        "exit_time": current_date, "exit_price": curr["daily_close"],
                        "signal": -1, "direction": "SHORT",
                    })
                    position = 0
                    entry_time = None

        if position != 0 and entry_time is not None:
            trades.append({
                "entry_date": entry_time.date(), "entry_time": entry_time,
                "entry_price": entry_price, "exit_date": dates[-1].date(),
                "exit_time": dates[-1], "exit_price": daily["daily_close"].iloc[-1],
                "signal": position, "direction": "LONG" if position == 1 else "SHORT",
            })

        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
            trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])

        logger.info("OI+Price Action: %d trades generated", len(trades_df))
        return trades_df
