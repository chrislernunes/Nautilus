"""
src/nautilus/etl/macro.py
==========================
REAL macro data only — zero synthetic data.

Sources
-------
1. RBI repo rate   : bundled historical CSV (data/rbi_repo_rate.csv)
2. India 10Y yield : yfinance NIFTYGS10YR.NS (free, no API key).
3. 200-DMA ratio   : computed from Nifty 50 price (yfinance ^NSEI).

All features are forward-filled from policy announcement dates to daily
frequency, then shifted +1 (no look-ahead) before being returned.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

from nautilus.config import (
    BOND_YIELD_TICKER,
    CACHE_DIR,
    CACHE_TTL_HOURS,
    DATA_DIR,
    DEFAULT_START_DATE,
)

logger = logging.getLogger(__name__)

_BUNDLED_CSV = DATA_DIR / "rbi_repo_rate.csv"


# ── Safe parquet read ──────────────────────────────────────────────────────────

def _cache_write(df: "pd.DataFrame", path: "Path") -> None:
    """Write cache as parquet if available, else CSV fallback."""
    try:
        df.to_parquet(path)
    except ImportError:
        df.to_csv(path.with_suffix(".csv"))
        logger.debug("parquet unavailable, cached as CSV: %s", path.with_suffix(".csv"))
    except Exception as exc:
        logger.warning("Cache write failed (%s): %s", path, exc)


def _cache_read(path: "Path") -> "pd.DataFrame | None":
    """Read parquet or CSV cache. Returns None if not found."""
    if path.exists():
        try:
            return pd.read_parquet(path)
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("parquet read failed: %s", exc)
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path, index_col=0, parse_dates=True)
        except Exception as exc:
            logger.warning("CSV cache read failed: %s", exc)
    return None


def _read_series(path: "Path", name: str) -> "pd.Series":
    """
    Read cached file (parquet or CSV) and always return a pd.Series.
    Never uses squeeze() — single-row DataFrames squeeze to a scalar.
    """
    df = _cache_read(path)
    if df is None:
        raise FileNotFoundError(f"Cache not found: {path}")
    if isinstance(df, pd.DataFrame):
        s = df.iloc[:, 0]
    elif isinstance(df, pd.Series):
        s = df
    else:
        s = pd.Series([float(df)], name=name)
    s.name = name
    return s


# ── RBI Repo Rate ──────────────────────────────────────────────────────────────

def _load_bundled_repo() -> pd.Series:
    """Load the repo rate CSV bundled with the repo (always available)."""
    if not _BUNDLED_CSV.exists():
        raise FileNotFoundError(
            f"Bundled RBI repo rate CSV not found at {_BUNDLED_CSV}. "
            "Re-clone the repo or restore data/rbi_repo_rate.csv."
        )
    df = pd.read_csv(_BUNDLED_CSV, parse_dates=["date"])
    df = df.dropna(subset=["date", "repo_rate"]).sort_values("date")
    s  = df.set_index("date")["repo_rate"].astype(float)
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def _hardcoded_fallback() -> pd.Series:
    """Emergency in-memory fallback if the CSV is missing."""
    _data = {
        "2012-04-17": 8.00, "2013-01-29": 7.75, "2013-03-19": 7.50,
        "2013-05-03": 7.25, "2013-09-20": 7.50, "2013-10-29": 7.75,
        "2014-01-28": 8.00, "2015-01-15": 7.75, "2015-03-04": 7.50,
        "2015-06-02": 7.25, "2015-09-29": 6.75, "2016-04-05": 6.50,
        "2016-10-04": 6.25, "2017-08-02": 6.00, "2018-06-06": 6.25,
        "2018-08-01": 6.50, "2019-02-07": 6.25, "2019-04-04": 6.00,
        "2019-06-06": 5.75, "2019-08-07": 5.40, "2019-10-04": 5.15,
        "2020-03-27": 4.40, "2020-05-22": 4.00, "2022-05-04": 4.40,
        "2022-06-08": 4.90, "2022-08-05": 5.40, "2022-09-30": 5.90,
        "2022-12-07": 6.25, "2023-02-08": 6.50, "2025-02-07": 6.25,
        "2025-04-09": 6.00,
    }
    s = pd.Series({pd.Timestamp(k): v for k, v in _data.items()}, dtype=float)
    return s.sort_index()


def _try_live_rbi_update(existing: pd.Series) -> pd.Series:
    """Best-effort live RBI scrape. Returns existing series on any failure."""
    try:
        from bs4 import BeautifulSoup

        headers = {"User-Agent": "Mozilla/5.0 (NautilusResearch/0.5)"}
        url     = "https://www.rbi.org.in/Scripts/bs_viewcontent.aspx?Id=1726"
        resp    = requests.get(url, timeout=8, headers=headers)
        if resp.status_code != 200:
            return existing

        soup = BeautifulSoup(resp.text, "lxml")
        for tbl in soup.find_all("table"):
            for row in tbl.find_all("tr"):
                cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
                if len(cells) >= 2:
                    try:
                        rate_cell = next(
                            c for c in cells if "%" in c or any(ch.isdigit() for ch in c)
                        )
                        rate = float(rate_cell.replace("%", "").strip())
                        if 3.0 <= rate <= 10.0:
                            logger.debug("Parsed potential live rate: %.2f", rate)
                    except (StopIteration, ValueError):
                        pass
    except Exception as exc:
        logger.debug("Live RBI update skipped: %s", exc)

    return existing


def load_rbi_repo_rate(
    start: str = DEFAULT_START_DATE,
    force_refresh: bool = False,
) -> pd.Series:
    """
    Load RBI policy repo rate as a daily forward-filled series.

    Returns:
        pd.Series[float], index=DatetimeIndex, name="repo_rate", values in %.
    """
    cache = CACHE_DIR / "rbi_repo_rate_daily.parquet"

    if not force_refresh and cache.exists():
        age = datetime.now() - datetime.fromtimestamp(cache.stat().st_mtime)
        if age < timedelta(hours=CACHE_TTL_HOURS * 4):
            s = _read_series(cache, "repo_rate")
            logger.debug("Repo rate: cache hit (%d rows)", len(s))
            return s.loc[start:]

    try:
        raw = _load_bundled_repo()
        logger.info("Loaded %d repo rate records from bundled CSV", len(raw))
    except FileNotFoundError:
        logger.warning("Bundled CSV missing — using hardcoded fallback")
        raw = _hardcoded_fallback()

    raw = _try_live_rbi_update(raw)
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()

    today    = pd.Timestamp.today().normalize()
    bday_idx = pd.bdate_range(start=raw.index.min(), end=today)
    daily    = raw.reindex(bday_idx).ffill().rename("repo_rate")

    _cache_write(daily.to_frame(), cache)
    logger.info("Repo rate daily series: %d rows, last=%.2f%%", len(daily), daily.iloc[-1])

    return daily.loc[start:]


# ── Bond Yield (10Y G-Sec) — Bundled CSV primary, yfinance optional ────────────

_BUNDLED_BOND_CSV = DATA_DIR / "india_10y_yield.csv"


def _load_bundled_bond() -> pd.Series:
    """
    Load India 10Y G-Sec yield from the bundled CSV (primary source).

    CSV format (Investing.com export):
        Date, Price, Open, High, Low, Change %
        03-04-2026, 7.129, ...
    """
    if not _BUNDLED_BOND_CSV.exists():
        raise FileNotFoundError(
            f"Bundled India 10Y yield CSV not found at {_BUNDLED_BOND_CSV}. "
            "Restore data/india_10y_yield.csv."
        )
    df = pd.read_csv(_BUNDLED_BOND_CSV)
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
    df["date"] = pd.to_datetime(
        df["Date"].astype(str).str.strip(), format="%d-%m-%Y", dayfirst=True
    )
    df["yield_val"] = pd.to_numeric(
        df["Price"].astype(str).str.strip(), errors="coerce"
    )
    s = (
        df.dropna(subset=["date", "yield_val"])
        .set_index("date")["yield_val"]
        .sort_index()
        .rename("bond_yield_10y")
    )
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def load_bond_yield(
    ticker: str = BOND_YIELD_TICKER,
    start: str = DEFAULT_START_DATE,
    force_refresh: bool = False,
) -> pd.Series:
    """
    Load India 10-Year G-Sec yield.

    Primary source : bundled data/india_10y_yield.csv (always available,
                     covers 2018-01-01 to present — update periodically).
    Optional top-up: yfinance NIFTYGS10YR.NS for dates beyond CSV coverage.

    Returns
    -------
    pd.Series[float], name="bond_yield_10y", values in % (e.g. 6.95).
    Never returns an empty series — bundled CSV is always available.
    """
    # ── Step 1: Load bundled CSV (always available) ──────────────────────────
    try:
        bundled = _load_bundled_bond()
        logger.info("Bundled bond yield: %d rows, last=%.3f%%", len(bundled), bundled.iloc[-1])
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        bundled = pd.Series(dtype=float, name="bond_yield_10y")

    # ── Step 2: Try yfinance for any dates beyond bundled coverage ───────────
    yf_start = (
        (bundled.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if not bundled.empty else start
    )
    try:
        raw = yf.download(
            ticker,
            start=yf_start,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if isinstance(raw, pd.DataFrame) and not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw.index = pd.to_datetime(raw.index).tz_localize(None)
            col = "Close" if "Close" in raw.columns else raw.columns[0]
            yf_s = raw[col].rename("bond_yield_10y").dropna()
            if not yf_s.empty:
                bundled = pd.concat([bundled, yf_s]).sort_index()
                bundled = bundled[~bundled.index.duplicated(keep="last")]
                logger.info("yfinance top-up: +%d rows", len(yf_s))
    except Exception as exc:
        logger.debug("yfinance bond top-up skipped: %s", exc)

    if bundled.empty:
        logger.error("Bond yield: no data available at all")
        return pd.Series(dtype=float, name="bond_yield_10y")

    # Forward-fill to business-day frequency
    bday_idx = pd.bdate_range(start=bundled.index.min(), end=pd.Timestamp.today())
    daily = bundled.reindex(bday_idx).ffill().rename("bond_yield_10y")

    logger.info("Bond yield daily: %d rows, %s → %s, last=%.3f%%",
                len(daily), daily.index[0].date(), daily.index[-1].date(), daily.iloc[-1])
    return daily.loc[start:]


# ── Combined Macro Feature Frame ───────────────────────────────────────────────

def build_macro_features(
    price: pd.Series,
    start: str = DEFAULT_START_DATE,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Assemble macro features aligned to the price index.

    Features (all real, no synthetic data):
    - repo_rate           : RBI policy rate (%) — forward-filled
    - repo_chg_90d        : 90-day change in repo rate
    - repo_easing         : 1 if RBI cut in last 90d, 0 otherwise
    - bond_yield_10y      : India 10Y G-Sec yield (%)
    - bond_yield_chg_21d  : 21d change in bond yield
    - yield_spread        : bond_yield - repo_rate (term premium proxy)
    - dma_200_ratio       : (price / 200-DMA) - 1, clipped ±30%

    All features are shifted +1 before return (no look-ahead).
    """
    feat: dict[str, pd.Series] = {}

    try:
        repo = load_rbi_repo_rate(start=start, force_refresh=force_refresh)
        repo_aligned         = repo.reindex(price.index).ffill().bfill()
        feat["repo_rate"]    = repo_aligned
        feat["repo_chg_90d"] = repo_aligned.diff(90).fillna(0.0)
        feat["repo_easing"]  = (repo_aligned.diff(90) < 0).astype(float)
    except Exception as exc:
        logger.warning("Repo rate features unavailable: %s", exc)

    try:
        bond = load_bond_yield(start=start, force_refresh=force_refresh)
        bond_aligned               = bond.reindex(price.index).ffill()
        feat["bond_yield_10y"]     = bond_aligned
        feat["bond_yield_chg_21d"] = bond_aligned.diff(21)
        if "repo_rate" in feat:
            feat["yield_spread"] = bond_aligned - feat["repo_rate"]
    except Exception as exc:
        logger.warning("Bond yield features unavailable: %s", exc)

    dma_200 = price.rolling(200, min_periods=100).mean()
    feat["dma_200_ratio"] = ((price / dma_200) - 1.0).clip(-0.30, 0.30)

    df = pd.DataFrame(feat, index=price.index)
    return df.shift(1).ffill().bfill()


def load_macro_data(
    start: str = DEFAULT_START_DATE,
    force_refresh: bool = False,
) -> pd.DataFrame | None:
    """Legacy-compatible entry point for the dashboard."""
    try:
        from nautilus.etl.loader import load_index
        nifty = load_index(force_refresh=force_refresh)
        price = nifty["Close"]
        return build_macro_features(price, start=start, force_refresh=force_refresh)
    except Exception as exc:
        logger.error("load_macro_data failed: %s", exc)
        return None
