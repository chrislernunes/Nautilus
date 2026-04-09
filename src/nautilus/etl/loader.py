"""
src/nautilus/etl/loader.py
==========================
yfinance-backed price data loader with Parquet caching.

All data is REAL — no synthetic prices anywhere in this module.
Cache is keyed by ticker + start date. Staleness check uses mtime.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from nautilus.config import CACHE_DIR, CACHE_TTL_HOURS, DEFAULT_START_DATE, NIFTY_INDEX_TICKER


def _parquet_read(path) -> "pd.DataFrame":
    """Read parquet or CSV cache fallback."""
    try:
        return pd.read_parquet(path)
    except ImportError:
        csv_path = path.with_suffix(".csv")
        if csv_path.exists():
            return pd.read_csv(csv_path, index_col=0, parse_dates=True)
        raise FileNotFoundError(f"No cache found at {path}")


def _parquet_write(df: "pd.DataFrame", path) -> None:
    """Write parquet or CSV fallback."""
    try:
        df.to_parquet(path)
    except ImportError:
        df.to_csv(path.with_suffix(".csv"))
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Cache write failed: %s", exc)


logger = logging.getLogger(__name__)


def _cache_path(ticker: str) -> Path:
    safe = ticker.replace("^", "_caret_").replace(".", "_")
    return CACHE_DIR / f"{safe}.parquet"


def _is_stale(path: Path, ttl_hours: int = CACHE_TTL_HOURS) -> bool:
    if not path.exists():
        return True
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    if age > timedelta(hours=ttl_hours):
        return True
    # Also check the last date in the cached data — guards against bundled
    # parquet files whose mtime was reset by unzip to "now" even though the
    # data itself is weeks/months old.
    try:
        df = _parquet_read(path)
        if df.empty:
            return True
        last_date = pd.to_datetime(df.index).max().date()
        today = datetime.now().date()
        # Stale if last data row is more than 4 calendar days old
        # (covers weekends + 1 holiday buffer without hammering the API)
        if (today - last_date).days > 4:
            return True
    except Exception:
        return True
    return False


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def load_ohlcv(
    ticker: str,
    start: str = DEFAULT_START_DATE,
    end: str | None = None,
    force_refresh: bool = False,
    ttl_hours: int = CACHE_TTL_HOURS,
) -> pd.DataFrame:
    """
    Load OHLCV data for a single ticker with Parquet caching.

    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume].
        Index: DatetimeIndex, timezone-naive, business days.

    Raises:
        RuntimeError if fetch fails and no cache exists.
    """
    cache = _cache_path(ticker)
    should_fetch = force_refresh or _is_stale(cache, ttl_hours)

    if not should_fetch:
        logger.debug("Cache hit: %s", ticker)
        df = _parquet_read(cache)
    else:
        logger.info("Fetching %s from yfinance (start=%s)", ticker, start)
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if raw.empty:
                raise ValueError(f"yfinance returned empty DataFrame for {ticker!r}")

            df = _flatten_columns(raw)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.sort_index(inplace=True)
            _parquet_write(df, cache)
            logger.info("Cached %d rows for %s", len(df), ticker)

        except Exception as exc:
            if cache.exists():
                logger.warning("Fetch failed (%s); returning stale cache for %s", exc, ticker)
                df = _parquet_read(cache)
            else:
                raise RuntimeError(f"Cannot load {ticker}: {exc}") from exc

    df = df.loc[start:]
    if end:
        df = df.loc[:end]
    return df


def load_index(
    ticker: str = NIFTY_INDEX_TICKER,
    start: str = DEFAULT_START_DATE,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load a market index (e.g. Nifty 50 via ^NSEI)."""
    return load_ohlcv(ticker, start=start, force_refresh=force_refresh)


def load_universe(
    tickers: list[str],
    start: str = DEFAULT_START_DATE,
    force_refresh: bool = False,
    price_col: str = "Close",
) -> pd.DataFrame:
    """
    Load closing prices for a universe of tickers.
    Silently skips any ticker that fails (logs a warning).

    Returns:
        Wide DataFrame: rows = dates, columns = tickers.
    """
    series: dict[str, pd.Series] = {}

    for tk in tickers:
        try:
            df = load_ohlcv(tk, start=start, force_refresh=force_refresh)
            col = price_col if price_col in df.columns else "Close"
            s   = df[col].dropna()
            if not s.empty:
                series[tk] = s
        except Exception as exc:
            logger.warning("Skipping %s: %s", tk, exc)

    if not series:
        raise RuntimeError("No data loaded for universe — check tickers and connectivity.")

    prices = pd.DataFrame(series)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices.sort_index()