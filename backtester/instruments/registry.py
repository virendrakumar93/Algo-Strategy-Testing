"""
Instrument registry — registers all concrete instrument handlers.

To add a new instrument type:
    1. Create a new class inheriting BaseInstrument.
    2. Add a register_instrument() call here.
"""

from backtester.instruments.base import register_instrument
from backtester.instruments.equity import EquityInstrument
from backtester.instruments.futures import FuturesInstrument
from backtester.instruments.index import IndexInstrument
from backtester.instruments.mutual_fund import MutualFundInstrument
from backtester.instruments.options import OptionsInstrument

register_instrument("equity", EquityInstrument)
register_instrument("index", IndexInstrument)
register_instrument("futures", FuturesInstrument)
register_instrument("options", OptionsInstrument)
register_instrument("mutual_fund", MutualFundInstrument)
