"""
Data fetcher module for Kite Connect API.

Handles paginated fetching of minute-level NIFTY data with rate limiting,
caching, and error recovery.
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
from kiteconnect import KiteConnect

from backtester.utils.config import AppConfig

logger = logging.getLogger(__name__)

# Kite Connect limits: max 60 days per request for minute data, 3 req/sec
MAX_DAYS_PER_REQUEST: int = 60
API_RATE_LIMIT_DELAY: float = 0.35  # seconds between requests (~3 req/sec)


class KiteDataFetcher:
    """
    Fetches historical OHLCV data from Kite Connect API with pagination,
    rate-limit handling, and local caching.
    """

    def __init__(self, config: AppConfig) -> None:
        """
        Initialize the data fetcher.

        Args:
            config: Application configuration containing API credentials and data settings.
        """
        self._config = config
        self._kite = KiteConnect(api_key=config.api.api_key)
        self._kite.set_access_token(config.api.access_token)
        self._raw_dir = config.project_root / config.data.raw_dir
        self._processed_dir = config.project_root / config.data.processed_dir
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._processed_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, from_date: datetime, to_date: datetime) -> Path:
        """
        Generate a deterministic cache file path for a date range.

        Args:
            from_date: Start date of the range.
            to_date: End date of the range.

        Returns:
            Path to the parquet cache file.
        """
        key = f"{self._config.api.instrument_token}_{from_date.date()}_{to_date.date()}"
        hash_suffix = hashlib.md5(key.encode()).hexdigest()[:8]
        return self._raw_dir / f"nifty_minute_{hash_suffix}.parquet"

    def _is_cache_valid(self, cache_file: Path) -> bool:
        """
        Check if a cache file exists and is within the expiry window.

        Args:
            cache_file: Path to the cached parquet file.

        Returns:
            True if cache is valid and can be reused.
        """
        if not self._config.data.enable_cache:
            return False
        if not cache_file.exists():
            return False
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        age_hours = (datetime.now() - mtime).total_seconds() / 3600
        return age_hours < self._config.data.cache_expiry_hours

    def _generate_date_chunks(
        self, from_date: datetime, to_date: datetime
    ) -> List[tuple[datetime, datetime]]:
        """
        Split a large date range into chunks that respect Kite API limits.

        Args:
            from_date: Start date.
            to_date: End date.

        Returns:
            List of (chunk_start, chunk_end) tuples.
        """
        chunks: List[tuple[datetime, datetime]] = []
        current = from_date
        while current < to_date:
            chunk_end = min(current + timedelta(days=MAX_DAYS_PER_REQUEST), to_date)
            chunks.append((current, chunk_end))
            current = chunk_end + timedelta(days=1)
        return chunks

    def _fetch_chunk(self, from_date: datetime, to_date: datetime) -> pd.DataFrame:
        """
        Fetch a single chunk of historical data from Kite API with retry logic.

        Args:
            from_date: Chunk start date.
            to_date: Chunk end date.

        Returns:
            DataFrame with OHLCV data for the chunk.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                logger.debug(
                    "Fetching chunk: %s to %s (attempt %d)",
                    from_date.date(),
                    to_date.date(),
                    attempt,
                )
                records = self._kite.historical_data(
                    instrument_token=self._config.api.instrument_token,
                    from_date=from_date,
                    to_date=to_date,
                    interval=self._config.data.interval,
                )
                df = pd.DataFrame(records)
                if not df.empty:
                    df["date"] = pd.to_datetime(df["date"])
                    df.set_index("date", inplace=True)
                    df = df[["open", "high", "low", "close", "volume"]]
                time.sleep(API_RATE_LIMIT_DELAY)
                return df
            except Exception as exc:
                logger.warning(
                    "Fetch attempt %d failed for %s to %s: %s",
                    attempt,
                    from_date.date(),
                    to_date.date(),
                    exc,
                )
                if attempt < max_retries:
                    backoff = 2 ** attempt
                    logger.info("Retrying in %d seconds...", backoff)
                    time.sleep(backoff)
                else:
                    raise RuntimeError(
                        f"Failed to fetch data for {from_date.date()} to {to_date.date()} "
                        f"after {max_retries} attempts."
                    ) from exc
        return pd.DataFrame()  # unreachable, satisfies type checker

    def fetch(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch the full date range of minute-level NIFTY data.

        Uses local parquet cache when available. Fetches from API in paginated
        chunks with rate limiting when cache is stale or missing.

        Args:
            force_refresh: If True, bypass cache and re-fetch from API.

        Returns:
            Complete DataFrame with datetime index and OHLCV columns.
        """
        to_date = datetime.now()
        from_date = to_date - timedelta(days=self._config.data.years * 365)

        # Check consolidated cache (keyed by date range so config changes invalidate it)
        range_key = f"{from_date.date()}_{to_date.date()}"
        range_hash = hashlib.md5(range_key.encode()).hexdigest()[:8]
        consolidated_cache = self._processed_dir / f"nifty_minute_{range_hash}.parquet"
        if not force_refresh and self._is_cache_valid(consolidated_cache):
            logger.info("Loading data from consolidated cache: %s", consolidated_cache)
            df = pd.read_parquet(consolidated_cache)
            df.index = pd.to_datetime(df.index)
            logger.info(
                "Loaded %d rows from cache (%s to %s)",
                len(df),
                df.index.min(),
                df.index.max(),
            )
            return df

        # Paginated fetch
        chunks = self._generate_date_chunks(from_date, to_date)
        logger.info(
            "Fetching %d chunks from Kite API (%s to %s)",
            len(chunks),
            from_date.date(),
            to_date.date(),
        )

        all_frames: List[pd.DataFrame] = []
        for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
            chunk_cache = self._cache_path(chunk_start, chunk_end)
            if not force_refresh and self._is_cache_valid(chunk_cache):
                logger.debug("Using cached chunk: %s", chunk_cache.name)
                chunk_df = pd.read_parquet(chunk_cache)
                chunk_df.index = pd.to_datetime(chunk_df.index)
            else:
                chunk_df = self._fetch_chunk(chunk_start, chunk_end)
                if not chunk_df.empty:
                    chunk_df.to_parquet(chunk_cache, engine="pyarrow")
            if not chunk_df.empty:
                all_frames.append(chunk_df)
            if i % 10 == 0:
                logger.info("Progress: %d/%d chunks fetched", i, len(chunks))

        if not all_frames:
            raise RuntimeError("No data fetched from Kite API. Check credentials and date range.")

        df = pd.concat(all_frames).sort_index()
        df = df[~df.index.duplicated(keep="first")]

        # Save consolidated cache
        df.to_parquet(consolidated_cache, engine="pyarrow")
        logger.info(
            "Fetched and cached %d rows (%s to %s)",
            len(df),
            df.index.min(),
            df.index.max(),
        )
        return df


def load_or_fetch_data(config: AppConfig, force_refresh: bool = False) -> pd.DataFrame:
    """
    High-level entry point: load data from cache or fetch from API.

    If a consolidated parquet file exists and the user doesn't have Kite
    credentials configured, it loads from the local file. Otherwise, it
    fetches via the API.

    Args:
        config: Application configuration.
        force_refresh: Force re-fetch from API.

    Returns:
        DataFrame with datetime index and OHLCV columns.
    """
    # Build range-aware cache path matching the fetcher logic
    to_date = datetime.now()
    from_date = to_date - timedelta(days=config.data.years * 365)
    range_key = f"{from_date.date()}_{to_date.date()}"
    range_hash = hashlib.md5(range_key.encode()).hexdigest()[:8]
    processed_dir = config.project_root / config.data.processed_dir
    consolidated = processed_dir / f"nifty_minute_{range_hash}.parquet"

    # If credentials are placeholder values, try to load from local cache only
    if config.api.api_key.startswith("your_") and consolidated.exists():
        logger.info("API credentials not set; loading from local cache.")
        df = pd.read_parquet(consolidated)
        df.index = pd.to_datetime(df.index)
        return df

    fetcher = KiteDataFetcher(config)
    return fetcher.fetch(force_refresh=force_refresh)
