"""
src/nautilus/strategies/momentum.py
=====================================
Price-based signals: MA filter, cross-sectional momentum, Williams VixFix.

No look-ahead hygiene: every public function returns signals shifted +1,
i.e. signal[t] = decision made at close of t-1, applied on day t.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_price_above_ma(
    price: pd.Series,
    window: int = 45,
) -> pd.Series:
    """
    Long-only binary signal: 1.0 if Close > MA(window), else 0.0.
    Shifted +1 (no look-ahead). MA NaN warm-up period → 0.
    """
    ma  = price.rolling(window, min_periods=window // 2).mean()
    raw = (price > ma).astype(float).where(ma.notna(), 0.0)
    return raw.shift(1).fillna(0.0)


def compute_price_regime(
    price: pd.Series,
    dma_window: int = 200,
) -> pd.Series:
    """Price regime for display (0 = below DMA, 1 = above). NOT shifted."""
    dma = price.rolling(dma_window, min_periods=dma_window // 2).mean()
    return (price > dma).astype(int).where(dma.notna(), 0)


def compute_combined_regime(
    price: pd.Series,
    macro_df: pd.DataFrame | None = None,
    dma_window: int = 200,
) -> pd.Series:
    """
    Combined price + macro regime for display (0/1, not shifted).

    Requires `macro_df` to contain a ``macro_regime`` column produced by
    ``build_macro_features`` in etl/macro.py. Falls back gracefully to
    price-only regime when macro data is unavailable.
    """
    pr = compute_price_regime(price, dma_window)
    if macro_df is None or "macro_regime" not in macro_df.columns:
        logger.debug("compute_combined_regime: no macro_regime column — using price-only regime")
        return pr
    mac = macro_df["macro_regime"].reindex(price.index).ffill().fillna(0).astype(int)
    return (pr & mac).astype(int)


def williams_vix_fix(
    price: pd.Series,
    pd_: int = 22,
    bbl: int = 20,
    mult: float = 2.0,
    lb: int = 50,
    ph: float = 0.85,
) -> pd.DataFrame:
    """
    Williams VixFix (CM_Williams_Vix_Fix) — synthetic fear gauge.

    Measures distance from rolling high as a proxy for implied vol / fear.
    Spikes indicate capitulation / market bottoms (historically: buy signal).

    Note v5: WVF is DISPLAY-ONLY by default. Position sizing via wvf_mult
    is opt-in (user must toggle in sidebar). The 15% boost is kept modest.

    All outputs shifted +1 (no look-ahead).

    NaN warm-up rows (before rolling windows are fully populated) are filled
    with 0.0, NOT backfilled — backfill would pull future values into the
    first bar, violating the no look-ahead guarantee.
    """
    roll_hi   = price.rolling(pd_, min_periods=pd_ // 2).max()
    wvf       = (roll_hi - price) / roll_hi * 100
    mid       = wvf.rolling(bbl, min_periods=bbl // 2).mean()
    s_dev     = wvf.rolling(bbl, min_periods=bbl // 2).std()
    upper_bb  = mid + mult * s_dev
    range_hi  = wvf.rolling(lb, min_periods=lb // 2).max() * ph
    wvf_spike = (wvf >= upper_bb) | (wvf >= range_hi)

    spike_pers = wvf_spike.rolling(5, min_periods=1).max()
    wvf_mult   = 1.0 + 0.15 * spike_pers

    df = pd.DataFrame({
        "wvf":        wvf,
        "wvf_upper":  upper_bb,
        "range_high": range_hi,
        "wvf_spike":  wvf_spike,
        "wvf_mult":   wvf_mult,
    })
    return df.shift(1).fillna(0.0)


def compute_cross_sectional_momentum(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    long_frac: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cross-sectional momentum: rank stocks by skip-last-month return,
    go long top decile, short bottom decile.

    Returns:
        (long_weights, short_weights) — both shifted +1.

    Implementation note: fully vectorized via DataFrame.rank() — avoids the
    O(n_dates × n_stocks) Python loop that was 50–100× slower.
    """
    mom = prices.shift(skip) / prices.shift(lookback) - 1

    n       = prices.shape[1]
    n_long  = max(1, int(n * long_frac))
    n_short = max(1, int(n * long_frac))

    # Count valid (non-NaN) stocks per row — rows with < 4 get zero weights
    valid_cnt = mom.notna().sum(axis=1)
    sufficient = valid_cnt >= 4

    # Rank in ascending order: rank 1 = worst, rank n = best
    ranks = mom.rank(axis=1, ascending=True, na_option="keep")

    # Long: top n_long stocks (highest ranks)
    long_threshold = (valid_cnt - n_long).clip(lower=0)
    long_mask      = (ranks.gt(long_threshold, axis=0)) & mom.notna() & sufficient.values[:, None]
    long_count     = long_mask.sum(axis=1).replace(0, np.nan)
    long_w         = long_mask.astype(float).div(long_count, axis=0).fillna(0.0)

    # Short: bottom n_short stocks (lowest ranks)
    short_mask  = (ranks.le(n_short, axis=0)) & mom.notna() & sufficient.values[:, None]
    short_count = short_mask.sum(axis=1).replace(0, np.nan)
    short_w     = short_mask.astype(float).div(short_count, axis=0).fillna(0.0)

    return long_w.shift(1), short_w.shift(1)
