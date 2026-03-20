"""
Interactive visualization module using Plotly.

Generates HTML-based interactive charts for equity curves, drawdowns,
yearly returns, trade-level P&L, monthly heatmaps, rolling returns,
and trade distribution analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


def plot_equity_curve(
    equity_df: pd.DataFrame,
    strategy_name: str,
    initial_capital: float,
    output_path: Optional[Path] = None,
) -> go.Figure:
    """
    Create an interactive equity curve chart.

    Args:
        equity_df: Time-indexed equity curve DataFrame.
        strategy_name: Strategy name for chart title.
        initial_capital: Starting capital (shown as reference line).
        output_path: If provided, save the chart as HTML.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=equity_df.index,
            y=equity_df["equity"],
            mode="lines",
            name="Equity",
            line=dict(color="#2196F3", width=2),
            hovertemplate="Date: %{x}<br>Equity: ₹%{y:,.0f}<extra></extra>",
        )
    )

    # Initial capital reference
    fig.add_hline(
        y=initial_capital,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Initial Capital: ₹{initial_capital:,.0f}",
    )

    fig.update_layout(
        title=dict(text=f"Equity Curve — {strategy_name}", font=dict(size=18)),
        xaxis_title="Date",
        yaxis_title="Equity (₹)",
        template="plotly_white",
        hovermode="x unified",
        height=500,
        margin=dict(l=80, r=40, t=60, b=60),
    )

    if output_path:
        fig.write_html(str(output_path), include_plotlyjs="cdn")
        logger.info("Equity curve saved: %s", output_path)

    return fig


def plot_drawdown_curve(
    equity_df: pd.DataFrame,
    strategy_name: str,
    output_path: Optional[Path] = None,
) -> go.Figure:
    """
    Create an interactive drawdown chart.

    Args:
        equity_df: Time-indexed equity curve DataFrame.
        strategy_name: Strategy name for chart title.
        output_path: If provided, save the chart as HTML.

    Returns:
        Plotly Figure object.
    """
    equity = equity_df["equity"]
    running_max = equity.cummax()
    drawdown_pct = ((equity - running_max) / running_max) * 100

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=drawdown_pct.index,
            y=drawdown_pct.values,
            mode="lines",
            fill="tozeroy",
            name="Drawdown",
            line=dict(color="#F44336", width=1.5),
            fillcolor="rgba(244, 67, 54, 0.3)",
            hovertemplate="Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=f"Drawdown Curve — {strategy_name}", font=dict(size=18)),
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
        hovermode="x unified",
        height=400,
        margin=dict(l=80, r=40, t=60, b=60),
    )

    if output_path:
        fig.write_html(str(output_path), include_plotlyjs="cdn")
        logger.info("Drawdown curve saved: %s", output_path)

    return fig


def plot_yearly_returns(
    yearly_metrics: List[Dict[str, Any]],
    strategy_name: str,
    output_path: Optional[Path] = None,
) -> go.Figure:
    """
    Create an interactive bar chart of year-wise returns.

    Args:
        yearly_metrics: List of year-level metrics dictionaries.
        strategy_name: Strategy name for chart title.
        output_path: If provided, save the chart as HTML.

    Returns:
        Plotly Figure object.
    """
    if not yearly_metrics:
        logger.warning("No yearly metrics to plot.")
        return go.Figure()

    years = [str(y["year"]) for y in yearly_metrics]
    returns = [y["return_pct"] for y in yearly_metrics]
    pnls = [y["total_pnl"] for y in yearly_metrics]
    trades = [y["num_trades"] for y in yearly_metrics]

    colors = ["#4CAF50" if r >= 0 else "#F44336" for r in returns]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=years,
            y=returns,
            marker_color=colors,
            name="Return %",
            text=[f"{r:.1f}%" for r in returns],
            textposition="outside",
            hovertemplate=(
                "Year: %{x}<br>"
                "Return: %{y:.2f}%<br>"
                "P&L: ₹%{customdata[0]:,.0f}<br>"
                "Trades: %{customdata[1]}<extra></extra>"
            ),
            customdata=list(zip(pnls, trades)),
        )
    )

    fig.update_layout(
        title=dict(text=f"Year-wise Returns — {strategy_name}", font=dict(size=18)),
        xaxis_title="Year",
        yaxis_title="Return (%)",
        template="plotly_white",
        height=450,
        margin=dict(l=80, r=40, t=60, b=60),
    )

    if output_path:
        fig.write_html(str(output_path), include_plotlyjs="cdn")
        logger.info("Yearly returns chart saved: %s", output_path)

    return fig


def plot_trade_pnl(
    trade_log: pd.DataFrame,
    strategy_name: str,
    output_path: Optional[Path] = None,
) -> go.Figure:
    """
    Create an interactive bar chart of trade-wise P&L.

    Args:
        trade_log: Per-trade results DataFrame with 'net_pnl_total' column.
        strategy_name: Strategy name for chart title.
        output_path: If provided, save the chart as HTML.

    Returns:
        Plotly Figure object.
    """
    if trade_log.empty:
        logger.warning("No trades to plot.")
        return go.Figure()

    pnl = trade_log["net_pnl_total"].values
    trade_ids = list(range(1, len(pnl) + 1))
    colors = ["#4CAF50" if p >= 0 else "#F44336" for p in pnl]

    directions = trade_log["direction"].values if "direction" in trade_log.columns else [""] * len(pnl)
    entry_times = trade_log["entry_time"].values if "entry_time" in trade_log.columns else [""] * len(pnl)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=trade_ids,
            y=pnl,
            marker_color=colors,
            name="Trade P&L",
            hovertemplate=(
                "Trade #%{x}<br>"
                "P&L: ₹%{y:,.0f}<br>"
                "Direction: %{customdata[0]}<br>"
                "Entry: %{customdata[1]}<extra></extra>"
            ),
            customdata=list(zip(directions, [str(t) for t in entry_times])),
        )
    )

    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=0.5)

    fig.update_layout(
        title=dict(text=f"Trade-wise P&L — {strategy_name}", font=dict(size=18)),
        xaxis_title="Trade #",
        yaxis_title="P&L (₹)",
        template="plotly_white",
        height=450,
        margin=dict(l=80, r=40, t=60, b=60),
    )

    if output_path:
        fig.write_html(str(output_path), include_plotlyjs="cdn")
        logger.info("Trade P&L chart saved: %s", output_path)

    return fig


# =============================================================================
# NEW: Monthly Returns Heatmap
# =============================================================================

def plot_monthly_heatmap(
    trade_log: pd.DataFrame,
    strategy_name: str,
    initial_capital: float,
    output_path: Optional[Path] = None,
) -> go.Figure:
    """
    Create a monthly returns heatmap (year × month grid).

    Args:
        trade_log: Per-trade results with entry_time and net_pnl_total.
        strategy_name: Strategy name for chart title.
        initial_capital: Starting capital for percentage calculation.
        output_path: If provided, save as HTML.

    Returns:
        Plotly Figure object.
    """
    if trade_log.empty:
        return go.Figure()

    from backtester.analytics.advanced import compute_monthly_returns

    monthly = compute_monthly_returns(trade_log, initial_capital)
    if monthly.empty:
        return go.Figure()

    # Ensure all 12 months present
    all_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for m in all_months:
        if m not in monthly.columns:
            monthly[m] = 0.0
    monthly = monthly[all_months]

    fig = go.Figure(
        data=go.Heatmap(
            z=monthly.values,
            x=monthly.columns.tolist(),
            y=[str(y) for y in monthly.index],
            colorscale=[
                [0.0, "#F44336"],
                [0.5, "#FFFFFF"],
                [1.0, "#4CAF50"],
            ],
            zmid=0,
            text=[[f"{v:.1f}%" for v in row] for row in monthly.values],
            texttemplate="%{text}",
            textfont={"size": 11},
            hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>",
            colorbar=dict(title="Return %"),
        )
    )

    fig.update_layout(
        title=dict(text=f"Monthly Returns Heatmap — {strategy_name}", font=dict(size=18)),
        xaxis_title="Month",
        yaxis_title="Year",
        template="plotly_white",
        height=max(300, len(monthly) * 50 + 120),
        margin=dict(l=80, r=40, t=60, b=60),
        yaxis=dict(autorange="reversed"),
    )

    if output_path:
        fig.write_html(str(output_path), include_plotlyjs="cdn")
        logger.info("Monthly heatmap saved: %s", output_path)

    return fig


# =============================================================================
# NEW: Rolling Returns Chart
# =============================================================================

def plot_rolling_returns(
    equity_df: pd.DataFrame,
    strategy_name: str,
    output_path: Optional[Path] = None,
) -> go.Figure:
    """
    Create a rolling 1-year return chart from equity curve.

    Args:
        equity_df: Time-indexed equity curve.
        strategy_name: Strategy name for chart title.
        output_path: If provided, save as HTML.

    Returns:
        Plotly Figure object.
    """
    if equity_df.empty or len(equity_df) < 10:
        return go.Figure()

    equity = equity_df["equity"]
    # Compute rolling returns with available window sizes
    fig = go.Figure()

    for window, label, color in [
        (252, "1Y Rolling", "#2196F3"),
        (63, "3M Rolling", "#FF9800"),
    ]:
        if len(equity) > window:
            rolling = ((equity / equity.shift(window)) - 1) * 100
            rolling = rolling.dropna()
            fig.add_trace(
                go.Scatter(
                    x=rolling.index,
                    y=rolling.values,
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=1.5),
                    hovertemplate=f"{label}: " + "%{y:.1f}%<extra></extra>",
                )
            )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5)

    fig.update_layout(
        title=dict(text=f"Rolling Returns — {strategy_name}", font=dict(size=18)),
        xaxis_title="Date",
        yaxis_title="Return (%)",
        template="plotly_white",
        hovermode="x unified",
        height=450,
        margin=dict(l=80, r=40, t=60, b=60),
    )

    if output_path:
        fig.write_html(str(output_path), include_plotlyjs="cdn")
        logger.info("Rolling returns chart saved: %s", output_path)

    return fig


# =============================================================================
# NEW: Trade Distribution (P&L histogram)
# =============================================================================

def plot_trade_distribution(
    trade_log: pd.DataFrame,
    strategy_name: str,
    output_path: Optional[Path] = None,
) -> go.Figure:
    """
    Create a histogram of trade P&L distribution.

    Args:
        trade_log: Per-trade results with net_pnl_total.
        strategy_name: Strategy name for chart title.
        output_path: If provided, save as HTML.

    Returns:
        Plotly Figure object.
    """
    if trade_log.empty:
        return go.Figure()

    pnl = trade_log["net_pnl_total"]

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=pnl,
            nbinsx=50,
            marker_color="#2196F3",
            opacity=0.75,
            name="P&L Distribution",
            hovertemplate="P&L Range: ₹%{x:,.0f}<br>Count: %{y}<extra></extra>",
        )
    )

    fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1)
    fig.add_vline(
        x=float(pnl.mean()),
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Mean: ₹{pnl.mean():,.0f}",
    )

    fig.update_layout(
        title=dict(text=f"Trade P&L Distribution — {strategy_name}", font=dict(size=18)),
        xaxis_title="P&L (₹)",
        yaxis_title="Frequency",
        template="plotly_white",
        height=400,
        margin=dict(l=80, r=40, t=60, b=60),
    )

    if output_path:
        fig.write_html(str(output_path), include_plotlyjs="cdn")
        logger.info("Trade distribution chart saved: %s", output_path)

    return fig


# =============================================================================
# Master plot generator (updated)
# =============================================================================

def generate_all_plots(
    equity_df: pd.DataFrame,
    trade_log: pd.DataFrame,
    metrics: Dict[str, Any],
    strategy_name: str,
    plots_dir: Path,
    initial_capital: float,
    file_prefix: Optional[str] = None,
) -> List[Path]:
    """
    Generate all interactive plots and save to disk.

    Args:
        equity_df: Time-indexed equity curve.
        trade_log: Per-trade results DataFrame.
        metrics: Full metrics dictionary.
        strategy_name: Strategy name for titles.
        plots_dir: Directory to save HTML plot files.
        initial_capital: Starting capital.
        file_prefix: Optional filename prefix (overrides strategy_name in filenames).

    Returns:
        List of paths to saved plot files.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    prefix = file_prefix or strategy_name
    saved: List[Path] = []

    # 1. Equity Curve
    eq_path = plots_dir / f"{prefix}_equity.html"
    plot_equity_curve(equity_df, strategy_name, initial_capital, eq_path)
    saved.append(eq_path)

    # 2. Drawdown Curve
    dd_path = plots_dir / f"{prefix}_drawdown.html"
    plot_drawdown_curve(equity_df, strategy_name, dd_path)
    saved.append(dd_path)

    # 3. Yearly Returns
    yearly_path = plots_dir / f"{prefix}_yearly_returns.html"
    yearly_data = metrics.get("yearly", [])
    plot_yearly_returns(yearly_data, strategy_name, yearly_path)
    saved.append(yearly_path)

    # 4. Trade P&L
    trade_path = plots_dir / f"{prefix}_trade_pnl.html"
    plot_trade_pnl(trade_log, strategy_name, trade_path)
    saved.append(trade_path)

    # 5. Monthly Heatmap (NEW)
    heatmap_path = plots_dir / f"{prefix}_monthly_heatmap.html"
    plot_monthly_heatmap(trade_log, strategy_name, initial_capital, heatmap_path)
    saved.append(heatmap_path)

    # 6. Rolling Returns (NEW)
    rolling_path = plots_dir / f"{prefix}_rolling_returns.html"
    plot_rolling_returns(equity_df, strategy_name, rolling_path)
    saved.append(rolling_path)

    # 7. Trade Distribution (NEW)
    dist_path = plots_dir / f"{prefix}_trade_distribution.html"
    plot_trade_distribution(trade_log, strategy_name, dist_path)
    saved.append(dist_path)

    logger.info("All plots saved to %s (%d files)", plots_dir, len(saved))
    return saved
