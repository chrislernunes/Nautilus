"""
python/core/historical.py
─────────────────────────────────────────────────────────────────────────────
Binance REST klines fetcher for warm-up at startup.

Fetches daily and 4h OHLCV data for BTCUSDT using the public
/api/v3/klines endpoint (no auth required).

Key design decisions:
  - Async with aiohttp; rate-limit aware (Binance: 1200 weight/min).
  - Returns pandas DataFrames with UTC DatetimeIndex.
  - 1000 bars per request (Binance max), stitched into full history.
  - Retries up to 3× with 1s backoff on 5xx.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import aiohttp
import numpy as np
import pandas as pd

log = logging.getLogger("nautilus.historical")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

BASE_URL   = "https://api.binance.com"
KLINES_EP  = "/api/v3/klines"
SYMBOL     = "BTCUSDT"
LIMIT      = 1000    # Binance max per request
MAX_RETRIES = 3

INTERVAL_MS = {
    "1s":  1_000,
    "1m":  60_000,
    "5m":  300_000,
    "15m": 900_000,
    "1h":  3_600_000,
    "4h":  14_400_000,
    "1d":  86_400_000,
}

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


# ─────────────────────────────────────────────────────────────────────────────
# Core fetch functions
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_klines(
    interval:  str,
    n_bars:    int,
    end_time:  Optional[int] = None,
    session:   Optional[aiohttp.ClientSession] = None,
) -> pd.DataFrame:
    """
    Fetch the last `n_bars` klines for BTCUSDT.

    Parameters
    ----------
    interval : str  e.g. "1d", "4h", "1h"
    n_bars   : int  number of bars to fetch (will paginate if > 1000)
    end_time : Optional[int]  end timestamp in ms (default: now)
    session  : Optional aiohttp session (created internally if None)

    Returns
    -------
    pd.DataFrame with columns: open, high, low, close, volume,
                                quote_volume, num_trades,
                                taker_buy_base, taker_buy_quote
                  indexed by UTC timestamp (DatetimeIndex)
    """
    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession()

    try:
        frames: List[pd.DataFrame] = []
        remaining = n_bars
        curr_end  = end_time or int(time.time() * 1000)

        while remaining > 0:
            limit     = min(remaining, LIMIT)
            raw_bars  = await _fetch_page(session, interval, limit, curr_end)
            if not raw_bars:
                break

            df = _parse_raw_klines(raw_bars)
            frames.append(df)
            remaining -= len(df)

            # Move window backwards
            if len(df) < limit:
                break
            curr_end = int(df.index[0].timestamp() * 1000) - 1

        if not frames:
            log.warning("No klines returned for interval=%s", interval)
            return pd.DataFrame()

        result = pd.concat(frames[::-1]).sort_index()
        result = result[~result.index.duplicated(keep="last")]
        log.info("Fetched %d %s bars for BTCUSDT", len(result), interval)
        return result

    finally:
        if owns_session:
            await session.close()


async def _fetch_page(
    session:   aiohttp.ClientSession,
    interval:  str,
    limit:     int,
    end_time:  int,
) -> list:
    """Single paginated request with retry."""
    params = {
        "symbol":    SYMBOL,
        "interval":  interval,
        "limit":     limit,
        "endTime":   end_time,
    }

    for attempt in range(MAX_RETRIES):
        try:
            async with session.get(
                BASE_URL + KLINES_EP,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 429:
                    retry_after = int(resp.headers.get("Retry-After", "5"))
                    log.warning("Rate-limited — sleeping %ds", retry_after)
                    await asyncio.sleep(retry_after)
                    continue
                resp.raise_for_status()
                return await resp.json()
        except aiohttp.ClientError as exc:
            log.warning("Request attempt %d failed: %s", attempt + 1, exc)
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(1.0 * (attempt + 1))

    return []


def _parse_raw_klines(raw: list) -> pd.DataFrame:
    """Convert raw Binance klines list to typed DataFrame."""
    df = pd.DataFrame(raw, columns=KLINE_COLS)
    df = df.drop(columns=["ignore"])

    # Types
    numeric_cols = ["open", "high", "low", "close", "volume",
                    "quote_volume", "taker_buy_base", "taker_buy_quote"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["num_trades"] = df["num_trades"].astype(np.int64)

    # Index
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time").drop(columns=["close_time"])

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Warm-up helper: fetch all data needed by the HMM at startup
# ─────────────────────────────────────────────────────────────────────────────

async def warm_up() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch warm-up data for model initialisation.

    Returns
    -------
    daily_df : ~730 days of daily bars
    h4_df    : ~365 × 6 = ~2190 four-hour bars (for feature cross-checks)
    """
    log.info("Starting historical warm-up...")
    async with aiohttp.ClientSession() as session:
        daily_task = fetch_klines("1d",  730, session=session)
        h4_task    = fetch_klines("4h", 1460, session=session)
        daily_df, h4_df = await asyncio.gather(daily_task, h4_task)

    log.info("Warm-up complete: %d daily bars, %d 4h bars",
             len(daily_df), len(h4_df))
    return daily_df, h4_df


# ─────────────────────────────────────────────────────────────────────────────
# yfinance max-history fallback for daily chart
# ─────────────────────────────────────────────────────────────────────────────

def fetch_max_daily_yfinance() -> pd.DataFrame:
    """
    Fetch maximum available daily BTC-USD history via yfinance (BTC-USD).

    Returns a DataFrame with the same column schema as fetch_klines("1d", …):
        open, high, low, close, volume, quote_volume, num_trades,
        taker_buy_base, taker_buy_quote
    indexed by UTC DatetimeIndex.

    Falls back silently to an empty DataFrame if yfinance is unavailable.
    """
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed — skipping max-history fetch")
        return pd.DataFrame()

    try:
        ticker = yf.Ticker("BTC-USD")
        raw    = ticker.history(period="max", interval="1d", auto_adjust=True)
        if raw.empty:
            log.warning("yfinance returned empty history for BTC-USD")
            return pd.DataFrame()

        # Normalise column names to lower-case
        raw.columns = [c.lower() for c in raw.columns]
        raw.index   = pd.to_datetime(raw.index, utc=True)

        # Map to expected schema (fill unavailable cols with 0)
        df = pd.DataFrame(index=raw.index)
        df["open"]            = raw.get("open",   raw.get("close", 0)).astype(float)
        df["high"]            = raw.get("high",   raw.get("close", 0)).astype(float)
        df["low"]             = raw.get("low",    raw.get("close", 0)).astype(float)
        df["close"]           = raw["close"].astype(float)
        df["volume"]          = raw.get("volume", pd.Series(0.0, index=raw.index)).astype(float)
        df["quote_volume"]    = 0.0
        df["num_trades"]      = 0
        df["taker_buy_base"]  = 0.0
        df["taker_buy_quote"] = 0.0

        df = df.dropna(subset=["close"])
        log.info("yfinance fetched %d daily bars (BTC-USD, max history)", len(df))
        return df

    except Exception as exc:
        log.warning("yfinance fetch failed: %s", exc)
        return pd.DataFrame()


async def warm_up_max() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extended warm-up: yfinance max history merged with Binance recent data.

    Strategy:
    1. Fetch yfinance max history (BTC-USD, years of data).
    2. Fetch last 730 Binance daily bars (authoritative OHLCV).
    3. Merge: use yfinance for older bars, Binance for recent bars.
       Binance bars overwrite any yfinance overlap.
    4. Also fetch 4h bars from Binance for feature cross-checks.

    Returns
    -------
    daily_df : merged max-history daily bars
    h4_df    : 1460 four-hour bars (Binance only)
    """
    log.info("Starting extended warm-up (yfinance + Binance)...")

    # Run yfinance in thread pool (it's synchronous)
    import asyncio
    loop = asyncio.get_event_loop()
    yf_df = await loop.run_in_executor(None, fetch_max_daily_yfinance)

    async with aiohttp.ClientSession() as session:
        bn_task  = fetch_klines("1d",  730, session=session)
        h4_task  = fetch_klines("4h", 1460, session=session)
        bn_daily, h4_df = await asyncio.gather(bn_task, h4_task)

    # Merge yfinance + Binance
    if yf_df.empty and bn_daily.empty:
        log.error("Both yfinance and Binance daily fetches failed")
        return pd.DataFrame(), h4_df

    if yf_df.empty:
        log.warning("yfinance unavailable — using Binance 730-bar history only")
        return bn_daily, h4_df

    if bn_daily.empty:
        log.warning("Binance daily fetch failed — using yfinance only")
        return yf_df, h4_df

    # Combine: concat then let Binance overwrite overlapping dates
    combined = pd.concat([yf_df, bn_daily])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    log.info("Merged daily bars: yfinance=%d  Binance=%d  combined=%d",
             len(yf_df), len(bn_daily), len(combined))
    return combined, h4_df
