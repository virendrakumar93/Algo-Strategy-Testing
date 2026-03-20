"""
Performance analytics module.

Computes comprehensive trading performance metrics including risk-adjusted
returns, drawdown analysis, and year-wise breakdowns.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_metrics(
    trade_log: pd.DataFrame,
    equity_df: pd.DataFrame,
    initial_capital: float,
) -> Dict[str, Any]:
    """
    Compute the full suite of performance metrics.

    Args:
        trade_log: DataFrame with per-trade results (must have 'net_pnl_total',
                   'direction', 'entry_time', 'exit_time' columns).
        equity_df: Time-indexed equity curve DataFrame.
        initial_capital: Starting capital for the backtest.

    Returns:
        Dictionary containing all computed metrics organized by category.
    """
    if trade_log.empty:
        logger.warning("Empty trade log — returning zero metrics.")
        return _empty_metrics()

    pnl = trade_log["net_pnl_total"].values
    final_equity = equity_df["equity"].iloc[-1] if not equity_df.empty else initial_capital

    # Core metrics
    total_return = final_equity - initial_capital
    total_return_pct = (total_return / initial_capital) * 100

    # CAGR
    first_trade = pd.Timestamp(trade_log["entry_time"].iloc[0])
    last_trade = pd.Timestamp(trade_log["exit_time"].iloc[-1])
    years = max((last_trade - first_trade).days / 365.25, 1 / 365.25)
    cagr = ((final_equity / initial_capital) ** (1 / years) - 1) * 100

    # Win/Loss analysis
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    total_trades = len(pnl)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0.0

    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    # Expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)
    loss_rate = (loss_count / total_trades) if total_trades > 0 else 0.0
    expectancy = (win_rate / 100 * avg_win) + (loss_rate * avg_loss)

    # Profit Factor
    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Drawdown analysis
    dd_metrics = _compute_drawdown(equity_df, initial_capital)

    # Return / MDD
    mdd = dd_metrics["max_drawdown_pct"]
    return_over_mdd = total_return_pct / abs(mdd) if mdd != 0 else float("inf")

    # Year-wise breakdown
    yearly = _compute_yearly_breakdown(trade_log, initial_capital)

    metrics: Dict[str, Any] = {
        "core": {
            "total_return": round(total_return, 2),
            "total_return_pct": round(total_return_pct, 2),
            "cagr_pct": round(cagr, 2),
            "win_rate_pct": round(win_rate, 2),
            "risk_reward_ratio": round(risk_reward, 4),
            "expectancy": round(expectancy, 2),
            "max_drawdown_pct": round(mdd, 2),
            "max_drawdown_abs": round(dd_metrics["max_drawdown_abs"], 2),
            "max_drawdown_duration_days": dd_metrics["max_drawdown_duration_days"],
            "return_over_mdd": round(return_over_mdd, 4),
        },
        "trade": {
            "total_trades": total_trades,
            "winning_trades": win_count,
            "losing_trades": loss_count,
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 4),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "largest_win": round(float(np.max(wins)), 2) if len(wins) > 0 else 0.0,
            "largest_loss": round(float(np.min(losses)), 2) if len(losses) > 0 else 0.0,
            "avg_pnl_per_trade": round(float(np.mean(pnl)), 2),
            "long_trades": int((trade_log["direction"] == "LONG").sum()),
            "short_trades": int((trade_log["direction"] == "SHORT").sum()),
        },
        "yearly": yearly,
        "summary": {
            "initial_capital": initial_capital,
            "final_equity": round(final_equity, 2),
            "backtest_start": str(first_trade),
            "backtest_end": str(last_trade),
            "backtest_duration_years": round(years, 2),
        },
    }

    logger.info(
        "Metrics computed: CAGR=%.2f%%, Win Rate=%.1f%%, MDD=%.2f%%, Trades=%d",
        cagr,
        win_rate,
        mdd,
        total_trades,
    )

    return metrics


def _compute_drawdown(
    equity_df: pd.DataFrame,
    initial_capital: float,
) -> Dict[str, Any]:
    """
    Compute drawdown metrics from equity curve.

    Args:
        equity_df: Time-indexed equity curve.
        initial_capital: Starting capital.

    Returns:
        Dictionary with max_drawdown_pct, max_drawdown_abs, max_drawdown_duration_days,
        and the full drawdown series.
    """
    if equity_df.empty:
        return {
            "max_drawdown_pct": 0.0,
            "max_drawdown_abs": 0.0,
            "max_drawdown_duration_days": 0,
            "drawdown_series": pd.Series(dtype=float),
        }

    equity = equity_df["equity"]
    running_max = equity.cummax()
    drawdown = equity - running_max
    drawdown_pct = (drawdown / running_max) * 100

    max_dd_abs = float(drawdown.min())
    max_dd_pct = float(drawdown_pct.min())

    # Drawdown duration: longest period below previous peak
    is_in_drawdown = drawdown < 0
    if is_in_drawdown.any():
        groups = (~is_in_drawdown).cumsum()
        dd_groups = is_in_drawdown.groupby(groups)
        durations = []
        for _, group in dd_groups:
            if group.any():
                start = group.index[0]
                end = group.index[-1]
                durations.append((end - start).days)
        max_dd_duration = max(durations) if durations else 0
    else:
        max_dd_duration = 0

    return {
        "max_drawdown_pct": max_dd_pct,
        "max_drawdown_abs": max_dd_abs,
        "max_drawdown_duration_days": max_dd_duration,
        "drawdown_series": drawdown_pct,
    }


def _compute_yearly_breakdown(
    trade_log: pd.DataFrame,
    initial_capital: float,
) -> List[Dict[str, Any]]:
    """
    Compute year-wise performance breakdown.

    Args:
        trade_log: Per-trade results DataFrame.
        initial_capital: Starting capital.

    Returns:
        List of dictionaries, one per year, with year-level metrics.
    """
    trade_log = trade_log.copy()
    trade_log["year"] = pd.to_datetime(trade_log["entry_time"]).dt.year

    yearly_results: List[Dict[str, Any]] = []
    for year, group in trade_log.groupby("year"):
        pnl = group["net_pnl_total"].values
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        total = float(np.sum(pnl))
        win_rate = (len(wins) / len(pnl)) * 100 if len(pnl) > 0 else 0.0
        profit_factor = (
            float(np.sum(wins)) / float(np.abs(np.sum(losses)))
            if len(losses) > 0 and np.sum(losses) != 0
            else float("inf")
        )

        yearly_results.append({
            "year": int(year),
            "num_trades": len(pnl),
            "total_pnl": round(total, 2),
            "return_pct": round((total / initial_capital) * 100, 2),
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 4),
            "avg_pnl": round(float(np.mean(pnl)), 2),
            "long_trades": int((group["direction"] == "LONG").sum()),
            "short_trades": int((group["direction"] == "SHORT").sum()),
        })

    return yearly_results


def _empty_metrics() -> Dict[str, Any]:
    """Return a zero-filled metrics dictionary for empty trade logs."""
    return {
        "core": {
            "total_return": 0.0,
            "total_return_pct": 0.0,
            "cagr_pct": 0.0,
            "win_rate_pct": 0.0,
            "risk_reward_ratio": 0.0,
            "expectancy": 0.0,
            "max_drawdown_pct": 0.0,
            "max_drawdown_abs": 0.0,
            "max_drawdown_duration_days": 0,
            "return_over_mdd": 0.0,
        },
        "trade": {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "avg_pnl_per_trade": 0.0,
            "long_trades": 0,
            "short_trades": 0,
        },
        "yearly": [],
        "summary": {
            "initial_capital": 0.0,
            "final_equity": 0.0,
            "backtest_start": "",
            "backtest_end": "",
            "backtest_duration_years": 0.0,
        },
    }


def metrics_to_flat_df(metrics: Dict[str, Any]) -> pd.DataFrame:
    """
    Flatten the nested metrics dictionary into a single-row DataFrame for CSV export.

    Args:
        metrics: Nested metrics dictionary from compute_metrics().

    Returns:
        Single-row DataFrame with all metrics as columns.
    """
    flat: Dict[str, Any] = {}
    for category in ["core", "trade", "summary"]:
        if category in metrics:
            for key, value in metrics[category].items():
                flat[f"{category}_{key}"] = value

    # Add yearly summary as total
    if "yearly" in metrics and metrics["yearly"]:
        flat["years_covered"] = len(metrics["yearly"])

    return pd.DataFrame([flat])
