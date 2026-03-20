"""
Backtest engine for executing strategy signals and computing trade-level P&L.

Supports configurable slippage, brokerage, and lot size.
Handles standard directional trades AND options-style trades with pnl_override.
Produces a detailed trade log and equity curve.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from backtester.utils.config import BacktestConfig

logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    """Container for a single trade's execution details."""
    trade_id: int
    direction: str
    signal: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    slippage_entry: float
    slippage_exit: float
    effective_entry: float
    effective_exit: float
    gross_pnl: float
    brokerage: float
    net_pnl: float
    lot_size: int
    net_pnl_total: float  # net_pnl * lot_size


class BacktestEngine:
    """
    Backtest engine that processes a DataFrame of trade signals
    and computes P&L accounting for slippage and brokerage.

    Supports:
        - Standard directional trades (LONG/SHORT)
        - Options strategies with 'pnl_override' column
        - Configurable position sizing via lot_size
    """

    def __init__(self, config: BacktestConfig) -> None:
        """
        Initialize the backtest engine.

        Args:
            config: Backtest configuration (capital, slippage, brokerage, lot size).
        """
        self._initial_capital = config.initial_capital
        self._slippage = config.slippage
        self._brokerage = config.brokerage
        self._lot_size = config.lot_size

    def run(self, trades_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute the backtest on a DataFrame of trade signals.

        Applies slippage and brokerage to each trade, computes P&L,
        and builds an equity curve.

        If trades_df contains a 'pnl_override' column (options strategies),
        that value is used directly as the per-unit P&L instead of computing
        from entry/exit prices.

        If trades_df contains a 'net_pnl_total' column (MF analysis mode),
        those values are used directly without modification.

        Args:
            trades_df: DataFrame with columns:
                signal (1/-1), direction (LONG/SHORT/...),
                entry_time, entry_price, exit_time, exit_price.
                Optional: pnl_override, net_pnl_total.

        Returns:
            Tuple of (trade_log_df, equity_curve_df):
                - trade_log_df: Detailed per-trade results.
                - equity_curve_df: Time-indexed equity curve.
        """
        if trades_df.empty:
            logger.warning("No trades to backtest.")
            return pd.DataFrame(), pd.DataFrame()

        has_pnl_override = "pnl_override" in trades_df.columns
        has_net_pnl_total = "net_pnl_total" in trades_df.columns

        logger.info(
            "Running backtest: %d trades, capital=%.0f, slippage=%.2f, brokerage=%.2f, lot=%d",
            len(trades_df),
            self._initial_capital,
            self._slippage,
            self._brokerage,
            self._lot_size,
        )

        results: List[TradeResult] = []
        equity = self._initial_capital
        equity_points: List[dict] = [{"time": trades_df["entry_time"].iloc[0], "equity": equity}]

        for idx, row in trades_df.iterrows():
            trade_id = len(results) + 1
            signal = int(row["signal"])
            direction = str(row["direction"])
            entry_price = float(row["entry_price"])
            exit_price = float(row["exit_price"])

            if has_pnl_override and pd.notna(row.get("pnl_override")):
                # Options-style: P&L is pre-computed per unit
                pnl_per_unit = float(row["pnl_override"])
                effective_entry = entry_price
                effective_exit = exit_price
                gross_pnl = pnl_per_unit * self._lot_size
                total_brokerage = self._brokerage * 2
                net_pnl = pnl_per_unit - (total_brokerage / self._lot_size)
                net_pnl_total = net_pnl * self._lot_size

            elif has_net_pnl_total and pd.notna(row.get("net_pnl_total")):
                # MF analysis: P&L is already fully computed
                effective_entry = entry_price
                effective_exit = exit_price
                gross_pnl = float(row["net_pnl_total"])
                total_brokerage = 0.0
                net_pnl = gross_pnl
                net_pnl_total = gross_pnl

            else:
                # Standard directional trade
                if signal == 1:  # LONG
                    effective_entry = entry_price + self._slippage
                    effective_exit = exit_price - self._slippage
                    gross_pnl_per_unit = effective_exit - effective_entry
                else:  # SHORT
                    effective_entry = entry_price - self._slippage
                    effective_exit = exit_price + self._slippage
                    gross_pnl_per_unit = effective_entry - effective_exit

                total_brokerage = self._brokerage * 2
                net_pnl = gross_pnl_per_unit - (total_brokerage / self._lot_size)
                net_pnl_total = net_pnl * self._lot_size
                gross_pnl = gross_pnl_per_unit * self._lot_size

            equity += net_pnl_total

            result = TradeResult(
                trade_id=trade_id,
                direction=direction,
                signal=signal,
                entry_time=pd.Timestamp(row["entry_time"]),
                exit_time=pd.Timestamp(row["exit_time"]),
                entry_price=entry_price,
                exit_price=exit_price,
                slippage_entry=self._slippage,
                slippage_exit=self._slippage,
                effective_entry=effective_entry,
                effective_exit=effective_exit,
                gross_pnl=gross_pnl,
                brokerage=total_brokerage,
                net_pnl=net_pnl,
                lot_size=self._lot_size,
                net_pnl_total=net_pnl_total,
            )
            results.append(result)
            equity_points.append({"time": pd.Timestamp(row["exit_time"]), "equity": equity})

        # Build trade log DataFrame
        trade_log = pd.DataFrame([vars(r) for r in results])

        # Build equity curve
        equity_df = pd.DataFrame(equity_points)
        equity_df["time"] = pd.to_datetime(equity_df["time"])
        equity_df.set_index("time", inplace=True)
        equity_df = equity_df[~equity_df.index.duplicated(keep="last")]

        logger.info(
            "Backtest complete: %d trades, final equity=%.2f, total return=%.2f%%",
            len(trade_log),
            equity,
            ((equity - self._initial_capital) / self._initial_capital) * 100,
        )

        return trade_log, equity_df
