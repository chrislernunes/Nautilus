"""
src/nautilus/strategies/regime.py
==================================
HMM-based macro regime detection.

v5 key fix — multiplier redesign
─────────────────────────────────
v4 used Kelly fractions as position multipliers:
    Bull=1.30, Neutral=0.40, Stress=0.15, Panic=0.00

These crushed returns in bull markets by averaging ~0.55× exposure.
v5 redesign: regime multipliers are RISK GATES, not Kelly fractions.
    Bull Quiet    = 1.00 (full exposure)
    Bull Volatile = 1.00 (still bullish)
    Neutral       = 0.75 (mild reduction)
    Stress        = 0.35 (meaningful cut)
    Panic         = 0.00 (cash)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


REGIMES: dict[int, dict] = {
    0: {"name": "Bull Quiet",    "color": "#2ECC71", "mult": 1.00, "cap": 1.50, "emoji": "🟢"},
    1: {"name": "Bull Volatile", "color": "#3498DB", "mult": 1.00, "cap": 1.20, "emoji": "🔵"},
    2: {"name": "Neutral",       "color": "#F1C40F", "mult": 0.75, "cap": 0.90, "emoji": "🟡"},
    3: {"name": "Stress",        "color": "#E67E22", "mult": 0.35, "cap": 0.50, "emoji": "🟠"},
    4: {"name": "Panic",         "color": "#E74C3C", "mult": 0.00, "cap": 0.10, "emoji": "🔴"},
}

N_REGIMES    = len(REGIMES)
REGIME_NAMES = [REGIMES[i]["name"]  for i in range(N_REGIMES)]
REGIME_COLS  = [REGIMES[i]["color"] for i in range(N_REGIMES)]
MULT_VEC     = np.array([REGIMES[i]["mult"] for i in range(N_REGIMES)], dtype=float)


def build_hmm_features(
    price: pd.Series,
    macro_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the feature matrix for HMM training.

    Core features: vol_21d, ret_21d, vol_ratio, vol_of_vol, ret_5d, drawdown
    Macro features (optional): bond_yield_chg_21d, yield_spread, repo_easing, dma_200_ratio

    All features shifted +1 (no look-ahead).
    """
    lr = np.log(price / price.shift(1))

    vol_21  = lr.rolling(21, min_periods=10).std() * np.sqrt(252)
    vol_63  = lr.rolling(63, min_periods=30).std() * np.sqrt(252)
    vol_rat = (vol_21 / vol_63.replace(0, np.nan)).clip(0.3, 3.0)
    vov     = vol_21.diff().abs().rolling(10, min_periods=5).mean()
    high_252 = price.rolling(252, min_periods=60).max()
    dd      = (price / high_252 - 1).clip(-0.60, 0.0)
    ret_5   = price.pct_change(5).clip(-0.20, 0.20)
    ret_21  = price.pct_change(21).clip(-0.35, 0.35)

    df = pd.DataFrame({
        "vol_21d":    vol_21,
        "ret_21d":    ret_21,
        "vol_ratio":  vol_rat,
        "vol_of_vol": vov,
        "ret_5d":     ret_5,
        "drawdown":   dd,
    }, index=price.index)

    if macro_df is not None:
        _macro_cols = ["bond_yield_chg_21d", "yield_spread", "repo_easing", "dma_200_ratio"]
        for col in _macro_cols:
            if col in macro_df.columns:
                df[col] = macro_df[col].reindex(price.index)

    return df.shift(1).dropna()


@dataclass
class HMMResult:
    """All outputs from a fitted HMM."""
    posteriors:    np.ndarray
    states:        np.ndarray
    trans_matrix:  np.ndarray
    soft_kelly:    np.ndarray
    dates:         pd.DatetimeIndex
    feature_names: list[str]
    model:         object = field(repr=False, default=None)


def fit_hmm(
    price: pd.Series,
    macro_df: pd.DataFrame | None = None,
    n_states: int = 5,
    n_iter: int = 200,
    random_state: int = 42,
) -> HMMResult | None:
    """
    Fit a Gaussian HMM for regime detection and return ordered results.

    State ordering: sorted by "badness" = vol - return.
    State 0 = best (low vol, high return) → Bull Quiet.
    State N-1 = worst (high vol, low return) → Panic.

    Returns None if hmmlearn is unavailable or data is insufficient.
    """
    _hmm_err: str | None = None
    try:
        from hmmlearn import hmm as hmmlib
    except ImportError:
        _hmm_err = "hmmlearn not installed — run: pip install hmmlearn"
    except Exception as exc:
        _hmm_err = f"hmmlearn import failed ({type(exc).__name__}): {exc}"
    if _hmm_err:
        logger.error(_hmm_err)
        raise RuntimeError(_hmm_err)

    try:
        from sklearn.preprocessing import RobustScaler
    except ImportError:
        _sk_err = "scikit-learn not installed — run: pip install scikit-learn"
        logger.error(_sk_err)
        raise RuntimeError(_sk_err)
    except Exception as exc:
        _sk_err = f"sklearn import failed ({type(exc).__name__}): {exc}"
        logger.error(_sk_err)
        raise RuntimeError(_sk_err)

    feat = build_hmm_features(price, macro_df)
    if len(feat) < 150:
        logger.warning("Not enough data to fit HMM (%d rows, need ≥150)", len(feat))
        return None

    X = RobustScaler().fit_transform(feat.values)

    model = hmmlib.GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=random_state,
        min_covar=1e-3,
    )
    model.fit(X)
    score_result = model.score_samples(X)
    if not (isinstance(score_result, tuple) and len(score_result) == 2):
        raise RuntimeError(f"hmmlearn score_samples returned unexpected type {type(score_result)} — check hmmlearn version (need >=0.3)")
    log_prob, posteriors = score_result
    logger.info("HMM log-likelihood: %.2f (%.4f/sample)", log_prob, log_prob / len(X))

    vol_idx  = list(feat.columns).index("vol_21d")
    ret_idx  = list(feat.columns).index("ret_21d")
    badness  = model.means_[:, vol_idx] - model.means_[:, ret_idx]
    order    = list(np.argsort(badness))

    posteriors_ord = posteriors[:, order]
    raw_states     = model.predict(X)
    remap          = {old: new for new, old in enumerate(order)}
    states_ord     = np.array([remap[s] for s in raw_states], dtype=int)

    A          = model.transmat_[np.ix_(order, order)]
    soft_kelly = posteriors_ord @ MULT_VEC

    return HMMResult(
        posteriors=posteriors_ord,
        states=states_ord,
        trans_matrix=A,
        soft_kelly=soft_kelly,
        dates=feat.index,
        feature_names=list(feat.columns),
        model=model,
    )


def markov_forecast(
    trans_matrix: np.ndarray,
    p0: np.ndarray,
    horizon: int = 20,
) -> np.ndarray:
    """
    Forward Markov chain probability propagation.

    Returns:
        (horizon+1, N) array — rows are probability vectors at t+0, t+1, ...
    """
    paths = [p0.copy()]
    p     = p0.copy()
    for _ in range(horizon):
        p = p @ trans_matrix
        paths.append(p.copy())
    return np.array(paths)
