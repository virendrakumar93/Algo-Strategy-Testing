"""
Base instrument abstraction and instrument registry.

Provides a unified interface for data fetching, preprocessing, and
feature computation across all instrument types (equity, index,
futures, options, mutual funds).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

import pandas as pd

from backtester.utils.config import AppConfig

logger = logging.getLogger(__name__)


class BaseInstrument(ABC):
    """
    Abstract base class for all instrument types.

    Each instrument type implements its own data fetching, preprocessing,
    and feature computation logic tailored to its market characteristics.
    """

    instrument_type: str = "base"
    description: str = "Abstract base instrument"

    @abstractmethod
    def fetch_data(self, config: AppConfig, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch raw OHLCV (or NAV) data for this instrument.

        Args:
            config: Application configuration.
            force_refresh: Bypass cache if True.

        Returns:
            Raw DataFrame with datetime index and price columns.
        """
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
        """
        Instrument-specific preprocessing (cleaning, filtering, normalization).

        Args:
            df: Raw fetched DataFrame.
            config: Application configuration.

        Returns:
            Cleaned and preprocessed DataFrame.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_features(self, df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
        """
        Compute instrument-specific technical indicators and features.

        Args:
            df: Preprocessed DataFrame.
            config: Application configuration.

        Returns:
            Feature-enriched DataFrame.
        """
        raise NotImplementedError

    def pipeline(self, config: AppConfig, force_refresh: bool = False) -> pd.DataFrame:
        """
        Run the full instrument data pipeline: fetch → preprocess → features.

        Args:
            config: Application configuration.
            force_refresh: Bypass cache.

        Returns:
            Feature-enriched DataFrame ready for strategy execution.
        """
        logger.info("Running %s pipeline for %s", self.instrument_type, config.instrument.symbol)
        raw = self.fetch_data(config, force_refresh=force_refresh)
        clean = self.preprocess(raw, config)
        featured = self.compute_features(clean, config)
        logger.info(
            "%s pipeline complete: %d rows, %d features",
            self.instrument_type,
            len(featured),
            len(featured.columns),
        )
        return featured


# ---------------------------------------------------------------------------
# Instrument Registry
# ---------------------------------------------------------------------------
INSTRUMENT_REGISTRY: Dict[str, Type[BaseInstrument]] = {}


def register_instrument(name: str, cls: Type[BaseInstrument]) -> None:
    """
    Register an instrument class in the global registry.

    Args:
        name: Instrument type key (equity, index, futures, options, mutual_fund).
        cls: Instrument class (must be a subclass of BaseInstrument).

    Raises:
        TypeError: If cls is not a subclass of BaseInstrument.
    """
    if not issubclass(cls, BaseInstrument):
        raise TypeError(f"{cls.__name__} is not a subclass of BaseInstrument")
    INSTRUMENT_REGISTRY[name] = cls
    logger.debug("Registered instrument: %s -> %s", name, cls.__name__)


def get_instrument(instrument_type: str) -> BaseInstrument:
    """
    Instantiate an instrument handler by type.

    Args:
        instrument_type: Instrument type key.

    Returns:
        An instance of the requested instrument handler.

    Raises:
        KeyError: If instrument type is not registered.
    """
    if instrument_type not in INSTRUMENT_REGISTRY:
        available = ", ".join(INSTRUMENT_REGISTRY.keys()) or "(none)"
        raise KeyError(
            f"Instrument type '{instrument_type}' not found. Available: {available}"
        )
    cls = INSTRUMENT_REGISTRY[instrument_type]
    logger.info("Loading instrument handler: %s (%s)", instrument_type, cls.__name__)
    return cls()
