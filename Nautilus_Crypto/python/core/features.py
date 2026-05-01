"""
python/core/features.py
─────────────────────────────────────────────────────────────────────────────
Feature engineering pipeline for the regime detection HMM.

Features computed (12 total):
  1.  log_ret_1d        — 1-day log return
  2.  log_ret_5d        — 5-day log return (momentum)
  3.  rv_5d             — realised vol, 5-day (raw std, not annualised)
  4.  rv_20d            — realised vol, 20-day
  5.  rv_ratio          — rv_5d / rv_20d (short-term vs long-term vol)
  6.  norm_range        — (high - low) / close (intra-day range)
  7.  parkinson_vol     — Parkinson high-low volatility estimator
  8.  vol_rel           — volume / vol_ma_20 (relative volume)
  9.  buy_imbalance     — taker_buy_base / volume
  10. efficiency_ratio  — Kaufman ER, 5-day
  11. ema_cross         — (EMA10 / EMA30) - 1
  12. atr_norm_ret      — 1-day log return / ATR14  (dimensionless, regime-discriminative)
  13. drawdown_60d      — price / 60-day rolling max - 1  (bear detection)

Design notes:
  - rv_5d / rv_20d use raw daily std (NOT annualised with sqrt(252)).
    Annualising inflates crypto vol by ~16x, compressing inter-regime
    separation and making BULL/BEAR states indistinguishable.
  - atr_norm_ret + drawdown_60d cleanly separate BEAR and HIGH_VOL from
    BULL and CHOP, reducing regime misclassification in crypto crashes.
  - All features are z-scored with a 252-day rolling window (no look-ahead).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("nautilus.features")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_features(
    df:            pd.DataFrame,
    z_score:       bool = True,
    z_window:      int  = 252,
    min_periods:   int  = None,
) -> pd.DataFrame:
    """
    Compute the full feature matrix from a daily OHLCV DataFrame.

    Parameters
    ----------
    df           : DataFrame with columns [open, high, low, close, volume,
                   taker_buy_base].  Index: DatetimeIndex UTC.
    z_score      : If True, rolling z-score each feature column.
    z_window     : Window for rolling mean/std used in z-scoring.
    min_periods  : Minimum observations before emitting non-NaN rows.

    Returns
    -------
    DataFrame of shape (n - lag, n_features), NaN rows dropped.
    """
    feats = pd.DataFrame(index=df.index)

    if min_periods is None:
        min_periods = max(20, min(60, z_window // 4, len(df) // 4))

    # ── 1. Returns ────────────────────────────────────────────────────────────
    log_close           = np.log(df["close"])
    feats["log_ret_1d"] = log_close.diff(1)
    feats["log_ret_5d"] = log_close.diff(5)

    # ── 2. Realised volatility (raw daily std — NOT annualised) ───────────────
    # Annualising with sqrt(252) inflates crypto vol by ~16x and collapses
    # inter-regime separation.  Raw std is more discriminative for HMM.
    log_ret1        = log_close.diff(1)
    feats["rv_5d"]  = log_ret1.rolling(5,  min_periods=3).std()
    feats["rv_20d"] = log_ret1.rolling(20, min_periods=10).std()

    # ── 3. Vol ratio ──────────────────────────────────────────────────────────
    feats["rv_ratio"] = feats["rv_5d"] / feats["rv_20d"].replace(0, np.nan)

    # ── 4. Normalised intra-day range ─────────────────────────────────────────
    feats["norm_range"] = (df["high"] - df["low"]) / df["close"]

    # ── 5. Parkinson volatility ───────────────────────────────────────────────
    feats["parkinson_vol"] = _parkinson_vol(df["high"], df["low"], window=10)

    # ── 6. Relative volume ────────────────────────────────────────────────────
    vol_ma = df["volume"].rolling(20, min_periods=10).mean()
    feats["vol_rel"] = df["volume"] / vol_ma.replace(0, np.nan)

    # ── 7. Taker buy imbalance ────────────────────────────────────────────────
    if "taker_buy_base" in df.columns:
        feats["buy_imbalance"] = (
            df["taker_buy_base"] / df["volume"].replace(0, np.nan)
        ).clip(0, 1)
    else:
        feats["buy_imbalance"] = 0.5

    # ── 8. Efficiency ratio (5-day) ───────────────────────────────────────────
    feats["efficiency_ratio"] = _efficiency_ratio(df["close"], window=5)

    # ── 9. EMA cross ──────────────────────────────────────────────────────────
    ema_fast = df["close"].ewm(span=10, adjust=False).mean()
    ema_slow = df["close"].ewm(span=30, adjust=False).mean()
    feats["ema_cross"] = (ema_fast / ema_slow.replace(0, np.nan)) - 1.0

    # ── 10. ATR-normalised return ─────────────────────────────────────────────
    # (1-day log return) / ATR14 — dimensionless, strongly separates trending
    # from volatile/choppy regimes without price-level bias.
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    feats["atr_norm_ret"] = feats["log_ret_1d"] / atr14.replace(0, np.nan)

    # ── 11. Drawdown from 60-day rolling max ──────────────────────────────────
    # Captures bear-regime depth.  Value is 0 at new highs, negative in drawdown.
    roll_max = df["close"].rolling(60, min_periods=10).max()
    feats["drawdown_60d"] = (df["close"] / roll_max.replace(0, np.nan)) - 1.0

    # ── Drop NaN rows ─────────────────────────────────────────────────────────
    feats = feats.dropna()

    # ── Z-score (rolling, no look-ahead) ─────────────────────────────────────
    if z_score:
        effective_min = min(min_periods, max(20, len(feats) // 4))
        feats = _rolling_zscore(feats, window=z_window, min_periods=effective_min)
        feats = feats.replace([np.inf, -np.inf], np.nan).dropna(how="all")

    log.debug("Feature matrix: %d rows × %d cols", *feats.shape)
    return feats


def build_features_from_bars(
    bars: list[dict],
    lookback: int = 500,
) -> pd.DataFrame:
    """
    Convenience wrapper for 1-second bars coming from the live feed.
    """
    if len(bars) < 5:
        return pd.DataFrame()

    df = pd.DataFrame(bars[-lookback:])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = df.rename(columns={"buy_volume": "taker_buy_base"})
    if "taker_buy_base" not in df.columns:
        df["taker_buy_base"] = df["volume"] * 0.5

    return build_features(df, z_score=True)


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parkinson_vol(
    high:   pd.Series,
    low:    pd.Series,
    window: int = 10,
) -> pd.Series:
    log_hl2 = (np.log(high / low.replace(0, np.nan))) ** 2
    coeff   = 1.0 / (4.0 * np.log(2))
    return (
        log_hl2.rolling(window, min_periods=max(2, window // 2)).mean() * coeff
    ).apply(np.sqrt)


def _efficiency_ratio(close: pd.Series, window: int = 5) -> pd.Series:
    direction  = (close - close.shift(window)).abs()
    volatility = close.diff().abs().rolling(window, min_periods=2).sum()
    return (direction / volatility.replace(0, np.nan)).clip(0, 1)


def _rolling_zscore(
    df:          pd.DataFrame,
    window:      int,
    min_periods: int,
) -> pd.DataFrame:
    eff_window = min(window, len(df))
    roll_mean  = df.rolling(eff_window, min_periods=min_periods).mean()
    roll_std   = df.rolling(eff_window, min_periods=min_periods).std()
    z = (df - roll_mean) / roll_std.where(roll_std > 1e-12, other=np.nan)
    z = z.fillna(0)
    return z.clip(-5, 5)
