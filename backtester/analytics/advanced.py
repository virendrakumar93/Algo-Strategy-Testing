"""
Advanced analytics module.

Extends the core metrics with:
    - Rolling Sharpe ratio
    - Rolling drawdown
    - Trade duration statistics
    - Monthly returns heatmap data
    - XIRR (for mutual fund analysis)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_rolling_sharpe(
    equity_df: pd.DataFrame,
    window: int = 252,
    risk_free_rate: float = 0.06,
) -> pd.Series:
    """
    Compute rolling Sharpe ratio from equity curve.

    Args:
        equity_df: Time-indexed equity DataFrame with 'equity' column.
        window: Rolling window in trading days (default 252 = 1 year).
        risk_free_rate: Annualized risk-free rate (default 6% for India).

    Returns:
        Series of rolling Sharpe ratio values.
    """
    if equity_df.empty or len(equity_df) < window:
        return pd.Series(dtype=float)

    returns = equity_df["equity"].pct_change().dropna()
    rf_daily = risk_free_rate / 252
    excess = returns - rf_daily

    rolling_sharpe = (
        excess.rolling(window).mean() / excess.rolling(window).std()
    ) * np.sqrt(252)

    return rolling_sharpe.dropna()


def compute_rolling_drawdown(equity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling drawdown series from equity curve.

    Args:
        equity_df: Time-indexed equity DataFrame.

    Returns:
        DataFrame with 'drawdown_pct' and 'drawdown_abs' columns.
    """
    if equity_df.empty:
        return pd.DataFrame()

    equity = equity_df["equity"]
    running_max = equity.cummax()
    dd_abs = equity - running_max
    dd_pct = (dd_abs / running_max) * 100

    return pd.DataFrame({
        "drawdown_abs": dd_abs,
        "drawdown_pct": dd_pct,
    }, index=equity_df.index)


def compute_trade_duration_stats(trade_log: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute trade duration statistics.

    Args:
        trade_log: Per-trade results with entry_time and exit_time.

    Returns:
        Dictionary with duration stats (mean, median, min, max in hours).
    """
    if trade_log.empty:
        return {"mean_hours": 0, "median_hours": 0, "min_hours": 0, "max_hours": 0}

    entry = pd.to_datetime(trade_log["entry_time"])
    exit_ = pd.to_datetime(trade_log["exit_time"])
    durations = (exit_ - entry).dt.total_seconds() / 3600  # hours

    return {
        "mean_hours": round(float(durations.mean()), 2),
        "median_hours": round(float(durations.median()), 2),
        "min_hours": round(float(durations.min()), 2),
        "max_hours": round(float(durations.max()), 2),
        "std_hours": round(float(durations.std()), 2),
    }


def compute_monthly_returns(
    trade_log: pd.DataFrame,
    initial_capital: float,
) -> pd.DataFrame:
    """
    Compute monthly return matrix for heatmap visualization.

    Args:
        trade_log: Per-trade results with entry_time and net_pnl_total.
        initial_capital: Starting capital for percentage computation.

    Returns:
        DataFrame with years as index, months (1-12) as columns,
        values as monthly return percentages.
    """
    if trade_log.empty:
        return pd.DataFrame()

    tl = trade_log.copy()
    tl["entry_time"] = pd.to_datetime(tl["entry_time"])
    tl["year"] = tl["entry_time"].dt.year
    tl["month"] = tl["entry_time"].dt.month

    monthly_pnl = tl.groupby(["year", "month"])["net_pnl_total"].sum().unstack(fill_value=0)
    monthly_returns = (monthly_pnl / initial_capital) * 100

    # Rename columns to month names
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    monthly_returns.columns = [month_names.get(c, str(c)) for c in monthly_returns.columns]

    return monthly_returns


def compute_xirr(
    cashflows: List[tuple[pd.Timestamp, float]],
    guess: float = 0.1,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> float:
    """
    Compute XIRR (Extended Internal Rate of Return) for irregular cashflows.

    Useful for mutual fund SIP/SWP analysis.

    Args:
        cashflows: List of (date, amount) tuples. Negative = outflow, positive = inflow.
        guess: Initial rate guess.
        max_iter: Maximum Newton-Raphson iterations.
        tol: Convergence tolerance.

    Returns:
        Annualized XIRR as a decimal (e.g., 0.12 = 12%).
    """
    if not cashflows or len(cashflows) < 2:
        return 0.0

    dates = [cf[0] for cf in cashflows]
    amounts = [cf[1] for cf in cashflows]
    base_date = dates[0]
    day_fractions = [(d - base_date).days / 365.25 for d in dates]

    rate = guess
    for _ in range(max_iter):
        npv = sum(a / (1 + rate) ** t for a, t in zip(amounts, day_fractions))
        dnpv = sum(-t * a / (1 + rate) ** (t + 1) for a, t in zip(amounts, day_fractions))
        if abs(dnpv) < 1e-12:
            break
        new_rate = rate - npv / dnpv
        if abs(new_rate - rate) < tol:
            return new_rate
        rate = new_rate

    return rate


def compute_advanced_metrics(
    trade_log: pd.DataFrame,
    equity_df: pd.DataFrame,
    initial_capital: float,
) -> Dict[str, Any]:
    """
    Compute all advanced metrics and return as a dictionary.

    Args:
        trade_log: Per-trade results DataFrame.
        equity_df: Time-indexed equity curve.
        initial_capital: Starting capital.

    Returns:
        Dictionary with advanced metrics organized by category.
    """
    result: Dict[str, Any] = {}

    # Rolling Sharpe
    rolling_sharpe = compute_rolling_sharpe(equity_df)
    if not rolling_sharpe.empty:
        result["rolling_sharpe"] = {
            "current": round(float(rolling_sharpe.iloc[-1]), 4),
            "mean": round(float(rolling_sharpe.mean()), 4),
            "max": round(float(rolling_sharpe.max()), 4),
            "min": round(float(rolling_sharpe.min()), 4),
        }

    # Trade duration stats
    result["trade_duration"] = compute_trade_duration_stats(trade_log)

    # Monthly returns matrix
    monthly = compute_monthly_returns(trade_log, initial_capital)
    if not monthly.empty:
        result["monthly_returns"] = monthly.to_dict()

    return result
