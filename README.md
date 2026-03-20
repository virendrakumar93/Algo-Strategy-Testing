# Multi-Instrument Backtesting & Analysis Framework

A production-grade, modular backtesting and market analysis engine for Indian financial markets, supporting equities, indices, futures, options, and mutual funds.

---

## 1. Project Overview

This framework provides a **config-driven, plug-and-play** system for:
- **Backtesting** directional strategies across equities, indices, futures, and options
- **Analyzing** mutual fund performance (CAGR, rolling returns, risk ratios)
- **Visualizing** results with 7 interactive Plotly charts
- **Tracking** every run with timestamped outputs and config snapshots

Built with institutional-grade code quality: type hints, docstrings, vectorized operations, comprehensive logging, and modular architecture.

---

## 2. Architecture

```
main.py                          ← CLI entry point
├── backtester/
│   ├── utils/
│   │   ├── config.py            ← YAML config loader (multi-instrument aware)
│   │   └── logger.py            ← Console + file logging
│   ├── instruments/             ← 🆕 Instrument abstraction layer
│   │   ├── base.py              ← BaseInstrument ABC + registry
│   │   ├── equity.py            ← Cash segment handler
│   │   ├── index.py             ← NIFTY/BANKNIFTY handler
│   │   ├── futures.py           ← Futures with rollover + VWAP
│   │   ├── options.py           ← Options with Greeks proxy + ATM
│   │   ├── mutual_fund.py       ← NAV-based analysis
│   │   └── registry.py          ← One-line registration
│   ├── data/
│   │   ├── fetcher.py           ← Kite Connect API (paginated, cached)
│   │   ├── processor.py         ← OHLCV cleaning, gap detection
│   │   └── features.py          ← Supertrend, prev day H/L (numpy)
│   ├── strategies/              ← 13 strategies across all instruments
│   │   ├── base.py              ← BaseStrategy ABC + registry
│   │   ├── registry.py          ← All strategy registrations
│   │   ├── btst_supertrend.py   ← Original BTST strategy
│   │   ├── equity_strategies.py ← MA Crossover, RSI, Donchian
│   │   ├── futures_strategies.py← EMA+ST, VWAP, OI+Price
│   │   ├── options_strategies.py← Straddle, Condor, Delta Neutral
│   │   └── mf_analysis.py       ← CAGR, Rolling Returns, Risk
│   ├── engines/
│   │   └── backtest.py          ← P&L engine (directional + options)
│   ├── analytics/
│   │   ├── metrics.py           ← Core metrics (CAGR, MDD, Profit Factor)
│   │   └── advanced.py          ← 🆕 Rolling Sharpe, XIRR, monthly heatmap
│   ├── visualization/
│   │   └── plots.py             ← 7 interactive Plotly charts
│   └── outputs/
│       └── manager.py           ← Hierarchical output + config snapshots
├── config.yaml                  ← Config with 11 templates
├── requirements.txt
└── README.md
```

### Pipeline Flow
```
Config → Instrument Handler → Data Fetch → Preprocess → Features
    → Strategy Signals → Backtest Engine → Analytics → Plots → Output
```

---

## 3. Supported Instruments

| Type | Key | Description | Data Source |
|------|-----|-------------|------------|
| Equity | `equity` | Individual stocks (RELIANCE, TCS, etc.) | Kite Connect |
| Index | `index` | NIFTY 50, BANKNIFTY, etc. | Kite Connect |
| Futures | `futures` | Index/stock futures with rollover | Kite Connect |
| Options | `options` | CE/PE with Greeks proxy | Kite Connect (underlying) |
| Mutual Fund | `mutual_fund` | NAV-based analysis | Local CSV / Kite MF |

---

## 4. Supported Strategies (13 total)

### Equity / Index (4)
| Strategy | Key | Description |
|----------|-----|-------------|
| BTST Supertrend | `btst_supertrend_breakout` | Entry 15:28 on breakout, exit 09:17 next day |
| MA Crossover | `ma_crossover_50_200` | Golden/Death Cross (SMA 50/200) |
| RSI Mean Reversion | `rsi_mean_reversion` | RSI(14) oversold/overbought reversal |
| Donchian Breakout | `donchian_breakout` | 20-day channel breakout |

### Futures (3)
| Strategy | Key | Description |
|----------|-----|-------------|
| EMA + Supertrend | `ema_supertrend_trend` | Dual trend confirmation |
| VWAP Reversion | `vwap_reversion` | Intraday mean reversion to VWAP |
| OI + Price Action | `oi_price_action` | Volume surge + directional move |

### Options (3)
| Strategy | Key | Description |
|----------|-----|-------------|
| Short Straddle | `short_straddle` | ATM straddle sell, intraday |
| Iron Condor | `iron_condor` | ATR-based wings, intraday |
| Delta Neutral | `delta_neutral` | Hedged straddle with re-balancing |

### Mutual Fund Analysis (3)
| Strategy | Key | Description |
|----------|-----|-------------|
| CAGR Analysis | `mf_cagr_analysis` | Multi-period CAGR computation |
| Rolling Returns | `mf_rolling_returns` | 1Y/3Y/5Y rolling return stats |
| Drawdown & Risk | `mf_drawdown_risk` | Sharpe, Sortino, volatility, MDD |

---

## 5. How to Add a New Strategy

**Two steps only. No other changes needed.**

### Step 1: Create your strategy file
```python
# backtester/strategies/my_strategy.py
from backtester.strategies.base import BaseStrategy
import pandas as pd

class MyStrategy(BaseStrategy):
    name = "my_strategy_name"
    description = "My custom strategy"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        trades = []
        # ... your logic ...
        return pd.DataFrame(trades)
```

### Step 2: Register it
Add one line to `backtester/strategies/registry.py`:
```python
from backtester.strategies.my_strategy import MyStrategy
register_strategy("my_strategy_name", MyStrategy)
```

Done. Run with: `python main.py --strategy my_strategy_name`

---

## 6. How to Add a New Instrument

### Step 1: Create handler
```python
# backtester/instruments/my_instrument.py
from backtester.instruments.base import BaseInstrument

class MyInstrument(BaseInstrument):
    instrument_type = "my_type"

    def fetch_data(self, config, force_refresh=False):
        # ... fetch logic ...
        pass

    def preprocess(self, df, config):
        # ... clean logic ...
        return df

    def compute_features(self, df, config):
        # ... feature logic ...
        return df
```

### Step 2: Register it
Add one line to `backtester/instruments/registry.py`:
```python
from backtester.instruments.my_instrument import MyInstrument
register_instrument("my_type", MyInstrument)
```

---

## 7. How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Configure
Edit `config.yaml` with your Kite Connect API credentials. See the 11 templates at the bottom of the file for quick setup.

### Run a Backtest
```bash
# Original BTST strategy
python main.py --strategy btst_supertrend_breakout

# With instrument override
python main.py --strategy rsi_mean_reversion --instrument equity

# Force data refresh
python main.py --strategy ma_crossover_50_200 --force-refresh

# Custom config file
python main.py --config my_config.yaml

# List all strategies
python main.py --list-strategies
```

### Run Mutual Fund Analysis
```bash
python main.py --strategy mf_cagr_analysis --instrument mutual_fund
python main.py --strategy mf_rolling_returns --instrument mutual_fund
python main.py --strategy mf_drawdown_risk --instrument mutual_fund
```
*Note: Place NAV data as CSV in `data/processed/mf_{SYMBOL}_nav.csv` with columns: `date, nav`.*

### Run Options Strategies
```bash
python main.py --strategy short_straddle --instrument options
python main.py --strategy iron_condor --instrument options
python main.py --strategy delta_neutral --instrument options
```

---

## 8. Config File Guide

The `config.yaml` file has these sections:

| Section | Purpose |
|---------|---------|
| `api` | Kite Connect credentials (supports env vars: `KITE_API_KEY`, `KITE_API_SECRET`, `KITE_ACCESS_TOKEN`) |
| `instrument` | What to trade: type, symbol, token, expiry, strike |
| `analysis` | Mode: `backtest` or `analysis` |
| `data` | Historical data: years, interval, cache settings |
| `strategy` | Which strategy + custom parameters |
| `backtest` | Capital, slippage, brokerage, lot size |
| `output` | What to save (CSV, JSON, HTML) |
| `logging` | Log level and file output |

**11 ready-to-use templates** are included at the bottom of `config.yaml`.

---

## 9. Output Folder Structure

Each run creates a timestamped, hierarchical output:

```
outputs/
└── {instrument_type}/
    └── {symbol}/
        └── {strategy_name}/
            └── {YYYYMMDD_HHMMSS}/     ← Run timestamp
                ├── plots/
                │   ├── {prefix}_equity.html
                │   ├── {prefix}_drawdown.html
                │   ├── {prefix}_yearly_returns.html
                │   ├── {prefix}_trade_pnl.html
                │   ├── {prefix}_monthly_heatmap.html
                │   ├── {prefix}_rolling_returns.html
                │   └── {prefix}_trade_distribution.html
                ├── metrics/
                │   ├── {prefix}_metrics.json
                │   ├── {prefix}_metrics.csv
                │   ├── {prefix}_yearly.csv
                │   └── {prefix}_equity_curve.csv
                ├── trades/
                │   └── {prefix}_trades.csv
                └── logs/
                    └── {prefix}_config_snapshot.yaml
```

**File naming convention**: `{instrument}_{symbol}_{strategy}_{timeframe}_{metric}.ext`
Example: `index_NIFTY 50_btst_supertrend_breakout_minute_equity.html`

---

## 10. Example Use Cases

### Use Case 1: Test BTST on NIFTY (10 years)
```yaml
instrument:
  type: "index"
  symbol: "NIFTY 50"
strategy:
  name: "btst_supertrend_breakout"
data:
  years: 10
```

### Use Case 2: RSI on RELIANCE equity
```bash
python main.py --strategy rsi_mean_reversion --instrument equity
```

### Use Case 3: Compare multiple strategies
```bash
python main.py --strategy btst_supertrend_breakout
python main.py --strategy ma_crossover_50_200
python main.py --strategy donchian_breakout
# Compare outputs in outputs/index/NIFTY 50/
```

### Use Case 4: Intraday Iron Condor
```yaml
instrument:
  type: "options"
  symbol: "NIFTY"
strategy:
  name: "iron_condor"
backtest:
  lot_size: 50
  initial_capital: 2000000
```

### Use Case 5: Mutual Fund CAGR
```bash
# Place NAV CSV at data/processed/mf_AXIS_BLUECHIP_nav.csv
python main.py --strategy mf_cagr_analysis --instrument mutual_fund
```

---

## 11. CLI Reference

```
python main.py [OPTIONS]

Options:
  --strategy NAME        Strategy name (overrides config)
  --config PATH          Config YAML file (default: config.yaml)
  --instrument TYPE      Instrument type override
  --force-refresh        Re-fetch data from API
  --list-strategies      Show all registered strategies
  -h, --help             Show help
```
