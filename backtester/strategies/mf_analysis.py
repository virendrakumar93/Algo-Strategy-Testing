"""
Mutual Fund analysis strategies (non-backtest mode).

1. CAGR Analysis
2. Rolling Returns Analysis
3. Drawdown + Risk Ratios Analysis

These produce analysis DataFrames rather than trade signals.
They are designed for mode: "analysis" in config.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from backtester.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class MFCAGRAnalysis(BaseStrategy):
    """
    Mutual Fund CAGR Analysis.

    Computes point-to-point CAGR for various holding periods
    and generates a summary suitable for reporting.
    """

    name: str = "mf_cagr_analysis"
    description: str = "Mutual Fund CAGR Analysis"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute CAGR for multiple holding periods.

        Instead of trade signals, returns a DataFrame with CAGR metrics
        that the analytics module can process.
        """
        if "nav" not in df.columns:
            if "close" in df.columns:
                df = df.copy()
                df["nav"] = df["close"]
            else:
                raise ValueError("NAV column required for MF analysis")

        nav = df["nav"]
        total_days = (nav.index[-1] - nav.index[0]).days
        total_years = total_days / 365.25

        results: List[Dict[str, Any]] = []

        # Point-to-point CAGR for standard periods
        periods = {"1Y": 252, "2Y": 504, "3Y": 756, "5Y": 1260, "7Y": 1764, "10Y": 2520}

        for label, trading_days in periods.items():
            if len(nav) >= trading_days:
                start_nav = nav.iloc[-trading_days]
                end_nav = nav.iloc[-1]
                years = trading_days / 252
                cagr = ((end_nav / start_nav) ** (1 / years) - 1) * 100

                results.append({
                    "entry_date": nav.index[-trading_days].date(),
                    "entry_time": nav.index[-trading_days],
                    "entry_price": start_nav,
                    "exit_date": nav.index[-1].date(),
                    "exit_time": nav.index[-1],
                    "exit_price": end_nav,
                    "signal": 1,
                    "direction": f"CAGR_{label}",
                    "net_pnl_total": end_nav - start_nav,
                })

        # Overall CAGR
        if len(nav) > 1:
            overall_cagr = ((nav.iloc[-1] / nav.iloc[0]) ** (1 / max(total_years, 0.01)) - 1) * 100
            results.append({
                "entry_date": nav.index[0].date(),
                "entry_time": nav.index[0],
                "entry_price": nav.iloc[0],
                "exit_date": nav.index[-1].date(),
                "exit_time": nav.index[-1],
                "exit_price": nav.iloc[-1],
                "signal": 1,
                "direction": f"CAGR_OVERALL ({total_years:.1f}Y)",
                "net_pnl_total": nav.iloc[-1] - nav.iloc[0],
            })

        trades_df = pd.DataFrame(results)
        if not trades_df.empty:
            trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
            trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])

        logger.info("MF CAGR Analysis: %d periods computed", len(trades_df))
        return trades_df


class MFRollingReturnsAnalysis(BaseStrategy):
    """
    Mutual Fund Rolling Returns Analysis.

    Computes rolling 1Y, 3Y, 5Y returns and generates summary statistics.
    """

    name: str = "mf_rolling_returns"
    description: str = "Mutual Fund Rolling Returns Analysis"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling returns summary."""
        if "nav" not in df.columns:
            if "close" in df.columns:
                df = df.copy()
                df["nav"] = df["close"]
            else:
                raise ValueError("NAV column required")

        nav = df["nav"]
        results: List[Dict[str, Any]] = []

        for years, label in [(1, "1Y"), (3, "3Y"), (5, "5Y")]:
            window = years * 252
            if len(nav) < window:
                continue

            rolling = ((nav / nav.shift(window)) ** (1 / years) - 1) * 100
            rolling = rolling.dropna()

            results.append({
                "entry_date": rolling.index[0].date(),
                "entry_time": rolling.index[0],
                "entry_price": rolling.mean(),
                "exit_date": rolling.index[-1].date(),
                "exit_time": rolling.index[-1],
                "exit_price": rolling.iloc[-1],
                "signal": 1,
                "direction": f"ROLLING_{label}",
                "net_pnl_total": rolling.mean(),
            })

        trades_df = pd.DataFrame(results)
        if not trades_df.empty:
            trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
            trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])

        logger.info("MF Rolling Returns: %d periods analyzed", len(trades_df))
        return trades_df


class MFDrawdownRiskAnalysis(BaseStrategy):
    """
    Mutual Fund Drawdown + Risk Ratios Analysis.

    Computes: Max Drawdown, Sharpe, Sortino, Volatility.
    """

    name: str = "mf_drawdown_risk"
    description: str = "Mutual Fund Drawdown & Risk Analysis"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute risk metrics summary."""
        if "nav" not in df.columns:
            if "close" in df.columns:
                df = df.copy()
                df["nav"] = df["close"]
            else:
                raise ValueError("NAV column required")

        nav = df["nav"]
        daily_returns = nav.pct_change().dropna()

        # Annualized metrics
        risk_free_rate = 0.06  # 6% (India govt bond proxy)
        excess = daily_returns - risk_free_rate / 252
        downside = daily_returns.copy()
        downside[downside > 0] = 0

        sharpe = (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() > 0 else 0
        sortino = (excess.mean() / downside.std()) * np.sqrt(252) if downside.std() > 0 else 0
        volatility = daily_returns.std() * np.sqrt(252) * 100

        # Max drawdown
        running_max = nav.cummax()
        drawdown = (nav - running_max) / running_max * 100
        max_dd = drawdown.min()

        results = [
            {
                "entry_date": nav.index[0].date(),
                "entry_time": nav.index[0],
                "entry_price": sharpe,
                "exit_date": nav.index[-1].date(),
                "exit_time": nav.index[-1],
                "exit_price": sortino,
                "signal": 1,
                "direction": "RISK_METRICS",
                "net_pnl_total": volatility,
            }
        ]

        trades_df = pd.DataFrame(results)
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
        trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])

        logger.info(
            "MF Risk Analysis: Sharpe=%.2f, Sortino=%.2f, Vol=%.1f%%, MDD=%.1f%%",
            sharpe, sortino, volatility, max_dd,
        )
        return trades_df
