"""
Base strategy class and strategy registry.

All strategies must inherit from BaseStrategy and implement generate_signals().
New strategies are registered in STRATEGY_REGISTRY for plug-and-play execution.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, Type

import pandas as pd

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Subclasses must implement `generate_signals()` which takes a feature-enriched
    DataFrame and returns a DataFrame with signal columns appended.

    Required signal columns:
        - 'signal': 1 for long, -1 for short, 0 for no trade.
        - 'entry_time': Timestamp of trade entry.
        - 'exit_time': Timestamp of trade exit.
    """

    name: str = "base"
    description: str = "Abstract base strategy"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trade signals from a feature-enriched DataFrame.

        Args:
            df: DataFrame with OHLCV data and computed indicators.

        Returns:
            DataFrame with trade signal columns appended:
                - 'signal': Trade direction (1=long, -1=short, 0=no trade).
                - 'entry_time': Entry timestamp.
                - 'entry_price': Entry price.
                - 'exit_time': Exit timestamp.
                - 'exit_price': Exit price.
        """
        raise NotImplementedError("Subclasses must implement generate_signals()")


# ---------------------------------------------------------------------------
# Strategy Registry
# ---------------------------------------------------------------------------
# Import concrete strategies here to populate the registry.
# New strategy addition requires ONLY:
#   1. A new class definition (inheriting BaseStrategy).
#   2. An entry in STRATEGY_REGISTRY below.
# NO changes anywhere else in the codebase.
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {}


def register_strategy(name: str, cls: Type[BaseStrategy]) -> None:
    """
    Register a strategy class in the global registry.

    Args:
        name: Unique string key for the strategy.
        cls: Strategy class (must be a subclass of BaseStrategy).

    Raises:
        TypeError: If cls is not a subclass of BaseStrategy.
        ValueError: If name is already registered.
    """
    if not issubclass(cls, BaseStrategy):
        raise TypeError(f"{cls.__name__} is not a subclass of BaseStrategy")
    if name in STRATEGY_REGISTRY:
        raise ValueError(f"Strategy '{name}' is already registered")
    STRATEGY_REGISTRY[name] = cls
    logger.debug("Registered strategy: %s -> %s", name, cls.__name__)


def get_strategy(name: str) -> BaseStrategy:
    """
    Instantiate a strategy by its registry name.

    Args:
        name: Strategy name key.

    Returns:
        An instance of the requested strategy.

    Raises:
        KeyError: If the strategy name is not found in the registry.
    """
    if name not in STRATEGY_REGISTRY:
        available = ", ".join(STRATEGY_REGISTRY.keys()) or "(none)"
        raise KeyError(
            f"Strategy '{name}' not found. Available strategies: {available}"
        )
    cls = STRATEGY_REGISTRY[name]
    logger.info("Loading strategy: %s (%s)", name, cls.__name__)
    return cls()
