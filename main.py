"""
Main entry point for the backtesting and analysis framework.

Supports multi-instrument, multi-strategy execution with config-driven
pipeline orchestration.

Usage:
    python main.py --strategy btst_supertrend_breakout
    python main.py --config config.yaml
    python main.py --strategy rsi_mean_reversion --instrument equity
    python main.py --strategy mf_cagr_analysis --instrument mutual_fund
    python main.py --strategy btst_supertrend_breakout --force-refresh
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Multi-Instrument Backtesting & Analysis Framework for Indian Markets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Original BTST strategy (backward-compatible)
  python main.py --strategy btst_supertrend_breakout

  # Equity strategies
  python main.py --strategy ma_crossover_50_200 --instrument equity
  python main.py --strategy rsi_mean_reversion --instrument index
  python main.py --strategy donchian_breakout --instrument equity

  # Futures strategies
  python main.py --strategy ema_supertrend_trend --instrument futures
  python main.py --strategy vwap_reversion --instrument futures

  # Options strategies
  python main.py --strategy short_straddle --instrument options
  python main.py --strategy iron_condor --instrument options
  python main.py --strategy delta_neutral --instrument options

  # Mutual Fund analysis
  python main.py --strategy mf_cagr_analysis --instrument mutual_fund

  # Custom config
  python main.py --config my_config.yaml
  python main.py --strategy btst_supertrend_breakout --force-refresh
        """,
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Strategy name (overrides config file).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml).",
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default=None,
        help="Instrument type override (equity|index|futures|options|mutual_fund).",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        default=False,
        help="Force re-fetch data from API, bypassing cache.",
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        default=False,
        help="List all available strategies and exit.",
    )
    return parser.parse_args()


def _list_strategies() -> None:
    """Print all registered strategies and exit."""
    import backtester.strategies.registry  # noqa: F401
    from backtester.strategies.base import STRATEGY_REGISTRY

    print("\n  Available Strategies")
    print("  " + "=" * 56)
    for name, cls in sorted(STRATEGY_REGISTRY.items()):
        desc = getattr(cls, "description", "")
        print(f"  {name:<35} {desc}")
    print("  " + "=" * 56 + "\n")


def main() -> None:
    """
    Main execution pipeline.

    Steps:
        1. Parse arguments and load configuration.
        2. Setup logging.
        3. Load instrument handler and run data pipeline.
        4. Load and run strategy to generate signals.
        5. Execute backtest engine.
        6. Compute performance analytics (core + advanced).
        7. Generate interactive visualizations (7 charts).
        8. Save all outputs with config snapshot.
    """
    args = parse_args()

    if args.list_strategies:
        _list_strategies()
        sys.exit(0)

    start_time = time.time()

    # -------------------------------------------------------------------------
    # Step 1: Load Configuration
    # -------------------------------------------------------------------------
    from backtester.utils.config import InstrumentConfig, load_config

    config = load_config(args.config, strategy_override=args.strategy)

    # Apply CLI instrument override if provided
    if args.instrument:
        original_inst = config.instrument
        new_inst = InstrumentConfig(
            type=args.instrument,
            symbol=original_inst.symbol,
            instrument_token=original_inst.instrument_token,
            exchange=original_inst.exchange,
            expiry=original_inst.expiry,
            strike=original_inst.strike,
            option_type=original_inst.option_type,
            timeframe=original_inst.timeframe,
        )
        config = replace(config, instrument=new_inst)

    # -------------------------------------------------------------------------
    # Step 2: Setup Logging
    # -------------------------------------------------------------------------
    from backtester.utils.logger import setup_logging

    setup_logging(
        level=config.logging.level,
        log_to_file=config.logging.log_to_file,
        log_dir=config.output.base_dir,
        strategy_name=config.strategy.name,
        project_root=config.project_root,
    )

    logger.info("=" * 70)
    logger.info("BACKTESTING FRAMEWORK — %s", config.strategy.name.upper())
    logger.info("=" * 70)
    logger.info("Configuration loaded from: %s", args.config)
    logger.info("Instrument: %s (%s)", config.instrument.symbol, config.instrument.type)
    logger.info("Strategy: %s", config.strategy.name)
    logger.info("Analysis Mode: %s", config.analysis.mode)
    logger.info("Initial Capital: ₹%s", f"{config.backtest.initial_capital:,.0f}")
    logger.info("Lot Size: %d", config.backtest.lot_size)

    # -------------------------------------------------------------------------
    # Step 3: Data Pipeline (via Instrument Handler)
    # -------------------------------------------------------------------------
    logger.info("-" * 40)
    logger.info("STEP 1: Data Pipeline (%s → %s)", config.instrument.type, config.instrument.symbol)
    logger.info("-" * 40)

    import backtester.instruments.registry  # noqa: F401
    from backtester.instruments.base import get_instrument

    instrument = get_instrument(config.instrument.type)
    featured_df = instrument.pipeline(config, force_refresh=args.force_refresh)

    logger.info("Data pipeline complete: %d rows, %d columns", len(featured_df), len(featured_df.columns))

    if featured_df.empty:
        logger.error("Data pipeline returned empty DataFrame. Check data source.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Step 4: Load Strategy and Generate Signals
    # -------------------------------------------------------------------------
    logger.info("-" * 40)
    logger.info("STEP 2: Strategy Signal Generation")
    logger.info("-" * 40)

    import backtester.strategies.registry  # noqa: F401
    from backtester.strategies.base import get_strategy

    strategy = get_strategy(config.strategy.name)
    trades_df = strategy.generate_signals(featured_df)

    if trades_df.empty:
        logger.error("No trades generated by %s. Exiting.", config.strategy.name)
        sys.exit(1)

    logger.info("Strategy signals: %d trades generated", len(trades_df))

    # -------------------------------------------------------------------------
    # Step 5: Execute Backtest
    # -------------------------------------------------------------------------
    logger.info("-" * 40)
    logger.info("STEP 3: Backtest Execution")
    logger.info("-" * 40)

    from backtester.engines.backtest import BacktestEngine

    engine = BacktestEngine(config.backtest)
    trade_log, equity_df = engine.run(trades_df)

    if trade_log.empty:
        logger.error("Backtest produced no results. Exiting.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Step 6: Compute Analytics (Core + Advanced)
    # -------------------------------------------------------------------------
    logger.info("-" * 40)
    logger.info("STEP 4: Performance Analytics")
    logger.info("-" * 40)

    from backtester.analytics.advanced import compute_advanced_metrics
    from backtester.analytics.metrics import compute_metrics

    metrics = compute_metrics(trade_log, equity_df, config.backtest.initial_capital)
    advanced = compute_advanced_metrics(trade_log, equity_df, config.backtest.initial_capital)
    metrics["advanced"] = advanced

    _print_summary(metrics, config.strategy.name, config.instrument)

    # -------------------------------------------------------------------------
    # Step 7: Generate Visualizations (7 interactive charts)
    # -------------------------------------------------------------------------
    logger.info("-" * 40)
    logger.info("STEP 5: Generating Visualizations")
    logger.info("-" * 40)

    from backtester.outputs.manager import OutputManager

    output_mgr = OutputManager(config)

    if config.output.save_plots_html:
        from backtester.visualization.plots import generate_all_plots

        generate_all_plots(
            equity_df=equity_df,
            trade_log=trade_log,
            metrics=metrics,
            strategy_name=config.strategy.name,
            plots_dir=output_mgr.plots_dir,
            initial_capital=config.backtest.initial_capital,
            file_prefix=config.file_prefix,
        )

    # -------------------------------------------------------------------------
    # Step 8: Save Outputs
    # -------------------------------------------------------------------------
    logger.info("-" * 40)
    logger.info("STEP 6: Saving Outputs")
    logger.info("-" * 40)

    saved_paths = output_mgr.save_all(trade_log, equity_df, metrics)

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("BACKTEST COMPLETE — %.2f seconds", elapsed)
    logger.info("Run ID: %s", output_mgr.run_id)
    logger.info("Outputs saved to: %s", output_mgr.base_dir)
    logger.info("=" * 70)


def _print_summary(metrics: dict, strategy_name: str, instrument: object = None) -> None:
    """
    Print a formatted performance summary to the console.

    Args:
        metrics: Full metrics dictionary.
        strategy_name: Strategy name.
        instrument: Instrument config object (optional).
    """
    core = metrics.get("core", {})
    trade = metrics.get("trade", {})
    summary = metrics.get("summary", {})
    advanced = metrics.get("advanced", {})

    inst_info = ""
    if instrument and hasattr(instrument, "type"):
        inst_info = f" [{instrument.type.upper()} : {instrument.symbol}]"

    print("\n" + "=" * 60)
    print(f"  PERFORMANCE SUMMARY — {strategy_name.upper()}{inst_info}")
    print("=" * 60)
    print(f"  Backtest Period    : {summary.get('backtest_start', 'N/A')[:10]} to {summary.get('backtest_end', 'N/A')[:10]}")
    print(f"  Duration           : {summary.get('backtest_duration_years', 0):.1f} years")
    print(f"  Initial Capital    : ₹{summary.get('initial_capital', 0):>14,.0f}")
    print(f"  Final Equity       : ₹{summary.get('final_equity', 0):>14,.0f}")
    print("-" * 60)
    print(f"  Total Return       : {core.get('total_return_pct', 0):>10.2f}%")
    print(f"  CAGR               : {core.get('cagr_pct', 0):>10.2f}%")
    print(f"  Max Drawdown       : {core.get('max_drawdown_pct', 0):>10.2f}%")
    print(f"  MDD Duration       : {core.get('max_drawdown_duration_days', 0):>10d} days")
    print(f"  Return / MDD       : {core.get('return_over_mdd', 0):>10.4f}")
    print("-" * 60)
    print(f"  Total Trades       : {trade.get('total_trades', 0):>10d}")
    print(f"    Long             : {trade.get('long_trades', 0):>10d}")
    print(f"    Short            : {trade.get('short_trades', 0):>10d}")
    print(f"  Win Rate           : {core.get('win_rate_pct', 0):>10.1f}%")
    print(f"  Risk-Reward        : {core.get('risk_reward_ratio', 0):>10.4f}")
    print(f"  Expectancy         : ₹{core.get('expectancy', 0):>13,.2f}")
    print(f"  Profit Factor      : {trade.get('profit_factor', 0):>10.4f}")
    print(f"  Avg Win            : ₹{trade.get('avg_win', 0):>13,.2f}")
    print(f"  Avg Loss           : ₹{trade.get('avg_loss', 0):>13,.2f}")

    # Advanced metrics
    duration = advanced.get("trade_duration", {})
    if duration:
        print("-" * 60)
        print(f"  Avg Trade Duration : {duration.get('mean_hours', 0):>10.1f} hrs")
        print(f"  Med Trade Duration : {duration.get('median_hours', 0):>10.1f} hrs")

    sharpe = advanced.get("rolling_sharpe", {})
    if sharpe:
        print(f"  Rolling Sharpe     : {sharpe.get('current', 0):>10.4f}")

    print("=" * 60)

    # Year-wise breakdown
    yearly = metrics.get("yearly", [])
    if yearly:
        print("\n  YEAR-WISE BREAKDOWN")
        print("-" * 60)
        print(f"  {'Year':<8} {'Trades':>8} {'P&L':>14} {'Return':>10} {'Win%':>8}")
        print("-" * 60)
        for y in yearly:
            print(
                f"  {y['year']:<8} {y['num_trades']:>8} "
                f"₹{y['total_pnl']:>12,.0f} {y['return_pct']:>9.2f}% "
                f"{y['win_rate_pct']:>7.1f}%"
            )
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
