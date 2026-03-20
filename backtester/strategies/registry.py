"""
Strategy registry initialization.

Imports all concrete strategies and registers them.
To add a new strategy:
    1. Create a new module in backtester/strategies/
    2. Define a class inheriting from BaseStrategy
    3. Add a register_strategy() call below
"""

from backtester.strategies.base import register_strategy

# ---------------------------------------------------------------------------
# BTST (original)
# ---------------------------------------------------------------------------
from backtester.strategies.btst_supertrend import BTSTSupertrendStrategy

register_strategy("btst_supertrend_breakout", BTSTSupertrendStrategy)

# ---------------------------------------------------------------------------
# Equity / Index Strategies
# ---------------------------------------------------------------------------
from backtester.strategies.equity_strategies import (
    DonchianBreakoutStrategy,
    MACrossoverStrategy,
    RSIMeanReversionStrategy,
)

register_strategy("ma_crossover_50_200", MACrossoverStrategy)
register_strategy("rsi_mean_reversion", RSIMeanReversionStrategy)
register_strategy("donchian_breakout", DonchianBreakoutStrategy)

# ---------------------------------------------------------------------------
# Futures Strategies
# ---------------------------------------------------------------------------
from backtester.strategies.futures_strategies import (
    EMASuperTrendStrategy,
    OIPriceActionStrategy,
    VWAPReversionStrategy,
)

register_strategy("ema_supertrend_trend", EMASuperTrendStrategy)
register_strategy("vwap_reversion", VWAPReversionStrategy)
register_strategy("oi_price_action", OIPriceActionStrategy)

# ---------------------------------------------------------------------------
# Options Strategies
# ---------------------------------------------------------------------------
from backtester.strategies.options_strategies import (
    DeltaNeutralStrategy,
    IronCondorStrategy,
    ShortStraddleStrategy,
)

register_strategy("short_straddle", ShortStraddleStrategy)
register_strategy("iron_condor", IronCondorStrategy)
register_strategy("delta_neutral", DeltaNeutralStrategy)

# ---------------------------------------------------------------------------
# Mutual Fund Analysis
# ---------------------------------------------------------------------------
from backtester.strategies.mf_analysis import (
    MFCAGRAnalysis,
    MFDrawdownRiskAnalysis,
    MFRollingReturnsAnalysis,
)

register_strategy("mf_cagr_analysis", MFCAGRAnalysis)
register_strategy("mf_rolling_returns", MFRollingReturnsAnalysis)
register_strategy("mf_drawdown_risk", MFDrawdownRiskAnalysis)
