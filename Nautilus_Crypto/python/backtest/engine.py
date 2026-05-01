"""
python/backtest/engine.py
─────────────────────────────────────────────────────────────────────────────
Vectorised backtest + walk-forward optimiser.

Transaction costs: 7.5 bps per round-turn (Binance taker 3.75 bps × 2).
The old default of 3 bps was unrealistically cheap and overstated regime-
strategy returns relative to realistic live execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from python.core.hmm_model import RegimeHMM, REGIME_SIGNAL, Regime
from python.core.features import build_features

log = logging.getLogger("nautilus.backtest")

TRADING_DAYS = 365


def _sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS))


def _sortino(returns: pd.Series, rf: float = 0.0) -> float:
    excess   = returns - rf / TRADING_DAYS
    downside = excess[excess < 0].std()
    if downside == 0:
        return 0.0
    return float(excess.mean() / downside * np.sqrt(TRADING_DAYS))


def _max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd       = (equity - roll_max) / roll_max
    return float(dd.min())


def _calmar(ann_return: float, max_dd: float) -> float:
    if max_dd == 0:
        return 0.0
    return ann_return / abs(max_dd)


def _ann_return(total_return: float, n_days: int) -> float:
    if n_days == 0:
        return 0.0
    return (1 + total_return) ** (TRADING_DAYS / n_days) - 1


@dataclass
class PerformanceStats:
    label:        str
    total_return: float
    ann_return:   float
    volatility:   float
    sharpe:       float
    sortino:      float
    max_dd:       float
    calmar:       float
    n_trades:     int
    win_rate:     float
    start_date:   str = ""
    end_date:     str = ""
    equity:       pd.Series = field(default_factory=pd.Series, repr=False)
    returns:      pd.Series = field(default_factory=pd.Series, repr=False)

    def to_dict(self) -> Dict[str, float | str | int]:
        return {
            "Label":         self.label,
            "Period":       f"{self.start_date} → {self.end_date}",
            "Total Ret %":   f"{self.total_return * 100:.1f}%",
            "Ann Ret %":     f"{self.ann_return * 100:.1f}%",
            "Volatility %":  f"{self.volatility * 100:.1f}%",
            "Sharpe":        f"{self.sharpe:.2f}",
            "Sortino":       f"{self.sortino:.2f}",
            "Max DD %":      f"{self.max_dd * 100:.1f}%",
            "Calmar":        f"{self.calmar:.2f}",
            "N Trades":      self.n_trades,
            "Win Rate %":    f"{self.win_rate * 100:.1f}%",
        }


def run_backtest(
    daily_df:   pd.DataFrame,
    model:      RegimeHMM,
    features:   pd.DataFrame,
    tc_bps:     float = 7.5,
) -> Tuple[PerformanceStats, PerformanceStats]:
    """
    Run Buy-and-Hold and Regime-Strategy backtests.

    tc_bps = 7.5 bps ≈ Binance taker 3.75 bps × 2 (entry + exit).
    """
    common_idx = features.index.intersection(daily_df.index)
    feats       = features.loc[common_idx]
    prices      = daily_df.loc[common_idx, "close"]
    log_ret     = np.log(prices / prices.shift(1)).dropna()

    if len(feats) < 10:
        raise ValueError(f"Insufficient aligned data: {len(feats)} bars")

    states, posteriors = model.predict(feats)

    signal_raw = pd.Series(
        [REGIME_SIGNAL.get(Regime(s), 0) for s in states],
        index=feats.index,
    )
    signal = signal_raw.shift(1).fillna(0)

    tc_per_trade  = tc_bps / 10_000
    position_diff = signal.diff().abs()
    tc_series     = position_diff * tc_per_trade

    aligned_ret = log_ret.reindex(signal.index).fillna(0)
    regime_ret  = (aligned_ret * signal - tc_series).fillna(0)
    bnh_ret     = aligned_ret.fillna(0)

    regime_equity = (1 + regime_ret).cumprod().ffill().fillna(1.0)
    bnh_equity    = (1 + bnh_ret).cumprod().ffill().fillna(1.0)

    n_days = len(aligned_ret)

    initial_long = 1 if (len(signal) > 0 and signal.iloc[0] == 1) else 0
    entries      = int(((signal.diff() > 0) & (signal == 1)).sum()) + initial_long
    held_returns = aligned_ret[signal == 1]
    win_rate     = float((held_returns > 0).mean()) if len(held_returns) > 0 else 0.0

    bnh_total   = float(bnh_equity.iloc[-1] - 1)
    reg_total   = float(regime_equity.iloc[-1] - 1)
    bnh_vol     = float(bnh_ret.std() * np.sqrt(TRADING_DAYS))
    reg_vol     = float(regime_ret.std() * np.sqrt(TRADING_DAYS))

    start_dt = aligned_ret.index[0].strftime("%Y-%m-%d") if len(aligned_ret) > 0 else "N/A"
    end_dt   = aligned_ret.index[-1].strftime("%Y-%m-%d") if len(aligned_ret) > 0 else "N/A"

    bnh_stats = PerformanceStats(
        label="Buy & Hold",
        total_return=bnh_total,
        ann_return=_ann_return(bnh_total, n_days),
        volatility=bnh_vol,
        sharpe=_sharpe(bnh_ret),
        sortino=_sortino(bnh_ret),
        max_dd=_max_drawdown(bnh_equity),
        calmar=_calmar(_ann_return(bnh_total, n_days), _max_drawdown(bnh_equity)),
        n_trades=1,
        win_rate=float((bnh_ret > 0).mean()),
        start_date=start_dt, end_date=end_dt,
        equity=bnh_equity, returns=bnh_ret,
    )

    regime_stats = PerformanceStats(
        label="Regime Strategy",
        total_return=reg_total,
        ann_return=_ann_return(reg_total, n_days),
        volatility=reg_vol,
        sharpe=_sharpe(regime_ret),
        sortino=_sortino(regime_ret),
        max_dd=_max_drawdown(regime_equity),
        calmar=_calmar(_ann_return(reg_total, n_days), _max_drawdown(regime_equity)),
        n_trades=entries,
        win_rate=win_rate,
        start_date=start_dt, end_date=end_dt,
        equity=regime_equity, returns=regime_ret,
    )

    return bnh_stats, regime_stats


@dataclass
class HyperParams:
    n_states:        int   = 4
    covariance_type: str   = "diag"
    n_iter:          int   = 200
    z_window:        int   = 252
    feature_set:     str   = "full"

    def label(self) -> str:
        return (f"s{self.n_states}_{self.covariance_type[:3]}"
                f"_z{self.z_window}_{self.feature_set[:3]}")


FEATURE_SETS = {
    "full":     None,
    "minimal":  ["log_ret_1d", "rv_5d", "vol_rel", "buy_imbalance"],
    "vol_only": ["rv_5d", "rv_20d", "rv_ratio", "parkinson_vol"],
}

HYPERPARAMETER_GRID = [
    HyperParams(n_states=3, covariance_type="diag",  z_window=252, feature_set="full"),
    HyperParams(n_states=4, covariance_type="diag",  z_window=252, feature_set="full"),
    HyperParams(n_states=4, covariance_type="full",  z_window=252, feature_set="full"),
    HyperParams(n_states=4, covariance_type="diag",  z_window=126, feature_set="full"),
    HyperParams(n_states=4, covariance_type="diag",  z_window=252, feature_set="minimal"),
    HyperParams(n_states=5, covariance_type="diag",  z_window=252, feature_set="full"),
]


def walk_forward_validate(
    daily_df:    pd.DataFrame,
    hp:          HyperParams,
    n_splits:    int = 5,
    test_frac:   float = 0.2,
) -> Dict[str, float]:
    n = len(daily_df)
    fold_size = int(n * test_frac)
    results   = []

    for i in range(n_splits):
        test_end   = n - i * (fold_size // n_splits)
        test_start = test_end - fold_size
        train_end  = test_start

        if train_end < 120 or test_start < 0:
            continue

        train_df = daily_df.iloc[:train_end]
        test_df  = daily_df.iloc[test_start:test_end]

        feat_cols = FEATURE_SETS.get(hp.feature_set)
        try:
            train_feats = build_features(train_df, z_score=True, z_window=hp.z_window)
            test_feats  = build_features(
                pd.concat([train_df.tail(hp.z_window), test_df]),
                z_score=True, z_window=hp.z_window
            ).tail(len(test_df))

            if feat_cols:
                available = [c for c in feat_cols if c in train_feats.columns]
                train_feats = train_feats[available]
                test_feats  = test_feats[[c for c in available if c in test_feats.columns]]

            if len(train_feats) < 60:
                continue

            model = RegimeHMM(
                n_states        = hp.n_states,
                covariance_type = hp.covariance_type,
                n_iter          = hp.n_iter,
            ).fit(train_feats)

            if not model.is_fitted:
                continue

            _, reg = run_backtest(test_df, model, test_feats)
            results.append({
                "sharpe":       reg.sharpe,
                "total_return": reg.total_return,
                "max_dd":       reg.max_dd,
            })
        except Exception as exc:
            log.debug("WF fold %d failed for %s: %s", i, hp.label(), exc)

    if not results:
        return {"sharpe": -99.0, "total_return": -1.0, "max_dd": -1.0}

    return {
        "sharpe":        float(np.mean([r["sharpe"]       for r in results])),
        "total_return":  float(np.mean([r["total_return"] for r in results])),
        "max_dd":        float(np.mean([r["max_dd"]       for r in results])),
        "n_valid_folds": len(results),
    }


def optimise_hyperparams(
    daily_df: pd.DataFrame,
    n_splits: int = 4,
) -> Tuple[HyperParams, pd.DataFrame]:
    rows = []
    best_hp     = HYPERPARAMETER_GRID[1]
    best_sharpe = -99.0

    for hp in HYPERPARAMETER_GRID:
        log.info("Evaluating %s ...", hp.label())
        metrics = walk_forward_validate(daily_df, hp, n_splits=n_splits)
        row = {"config": hp.label(), **metrics}
        rows.append(row)
        log.info("  Sharpe=%.2f  TotalRet=%.1f%%  MaxDD=%.1f%%",
                 metrics["sharpe"],
                 metrics["total_return"] * 100,
                 metrics["max_dd"] * 100)

        if metrics["sharpe"] > best_sharpe:
            best_sharpe = metrics["sharpe"]
            best_hp     = hp

    results_df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    log.info("Best config: %s (Sharpe=%.2f)", best_hp.label(), best_sharpe)
    return best_hp, results_df
