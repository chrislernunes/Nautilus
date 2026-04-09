"""
src/nautilus/backtests/engine.py
==================================
Vectorized backtest engine.

CRITICAL — shift convention (read before touching):
────────────────────────────────────────────────────
  signal[t]  = position HELD from close(t-1) to close(t)
             = decision made at close of day t-1
  returns[t] = price[t] / price[t-1] - 1
             = return earned FROM close(t-1) TO close(t)

Therefore:
  strategy_return[t] = signal[t] × returns[t]    (no additional shift)
  turnover[t]        = |signal[t] - signal[t-1]|

Callers are RESPONSIBLE for shifting their signals before passing them here.
This engine does NOT apply any shift internally (prevents double-shift bug).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Output from run_backtest."""
    name:          str
    equity_curve:  pd.Series
    daily_returns: pd.Series
    signal:        pd.Series
    metrics:       dict[str, str]


def run_backtest(
    price:    pd.Series,
    signal:   pd.Series,
    cost_bps: float = 10.0,
    name:     str   = "Strategy",
) -> BacktestResult:
    """
    Run a vectorized single-asset backtest.

    Args:
        price:    Asset price series (DatetimeIndex).
        signal:   Pre-shifted position series. Values in [-1.5, 1.5].
                  ⚠ Must already be shifted by caller. This function
                  does NOT apply any additional shift.
        cost_bps: One-way transaction cost in basis points.
        name:     Strategy label for reporting.

    Returns:
        BacktestResult with equity curve, daily returns, and metrics.
    """
    price  = price.sort_index()
    signal = signal.reindex(price.index).ffill().fillna(0.0)

    asset_ret = price.pct_change().fillna(0.0)
    gross_ret = signal * asset_ret

    cost     = cost_bps / 10_000
    turnover = signal.diff().abs().fillna(0.0)
    net_ret  = gross_ret - turnover * cost

    equity = (1.0 + net_ret).cumprod()

    return BacktestResult(
        name=name,
        equity_curve=equity,
        daily_returns=net_ret,
        signal=signal,
        metrics=compute_metrics(net_ret, name=name),
    )


def _dd_stats(equity: pd.Series) -> dict:
    """
    Compute drawdown statistics from an equity curve.

    Returns a dict with:
      max_dd          : worst peak-to-trough decline (negative float, e.g. -0.23)
      max_dd_days     : calendar days spent in that worst drawdown episode
      avg_dd          : mean daily drawdown across all underwater days
      dd_episodes     : number of distinct drawdown episodes (threshold -1%)
      longest_dd_days : calendar days in the longest drawdown episode (by duration)
      recovery_days   : calendar days to recover from the worst drawdown
                        (None if still underwater at end of series)
      current_dd      : current drawdown from all-time high (negative float)
    """
    dd = equity / equity.cummax() - 1.0

    max_dd      = float(dd.min())
    avg_dd      = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0
    current_dd  = float(dd.iloc[-1])

    # ── Episode detection ──────────────────────────────────────────────────
    # An episode starts when dd first crosses below -threshold and ends
    # when it returns to 0 (new high).
    THRESHOLD   = -0.01          # 1% — ignore micro noise
    in_dd       = (dd < THRESHOLD).astype(int)
    starts      = list(np.where(np.diff(in_dd, prepend=0) ==  1)[0])
    ends        = list(np.where(np.diff(in_dd, prepend=0) == -1)[0])

    # Clip if series ends while still in a drawdown
    if len(starts) > len(ends):
        ends.append(len(dd) - 1)

    episodes = list(zip(starts, ends))
    dd_episodes = len(episodes)

    # Calendar days per episode
    ep_days = []
    for s, e in episodes:
        ts = dd.index[s]
        te = dd.index[e]
        ep_days.append((te - ts).days)

    longest_dd_days = int(max(ep_days)) if ep_days else 0

    # ── Worst drawdown episode ─────────────────────────────────────────────
    max_dd_days   = 0
    recovery_days = None

    if episodes:
        worst_idx = int(np.argmin([dd.iloc[s:e+1].min() for s, e in episodes]))
        ws, we    = episodes[worst_idx]
        max_dd_days = ep_days[worst_idx]

        # Recovery: scan forward from 'we' for the first new all-time high
        peak_at_start = float(equity.iloc[:ws].max()) if ws > 0 else float(equity.iloc[ws])
        after_trough  = equity.iloc[we:]
        recovered     = after_trough[after_trough >= peak_at_start]
        if not recovered.empty:
            recovery_days = int((recovered.index[0] - dd.index[ws]).days)

    return {
        "max_dd":          max_dd,
        "max_dd_days":     max_dd_days,
        "avg_dd":          avg_dd,
        "dd_episodes":     dd_episodes,
        "longest_dd_days": longest_dd_days,
        "recovery_days":   recovery_days,
        "current_dd":      current_dd,
    }


def compute_metrics(
    returns: pd.Series,
    name: str = "Strategy",
    ppy: int  = 252,
) -> dict[str, str]:
    """
    Standard performance metrics from a daily return series.

    Includes drawdown depth, duration, and recovery metrics alongside
    the standard risk-adjusted return statistics.
    """
    _EMPTY_KEYS = [
        "name", "Total Return", "CAGR", "Ann. Vol", "Sharpe", "Sortino",
        "Max DD", "Max DD Days", "Longest DD Days", "Avg DD",
        "DD Episodes", "Recovery Days", "Current DD",
        "Calmar", "Win Rate", "Avg Win", "Avg Loss", "Profit Factor",
    ]
    r = returns.dropna()
    if len(r) <= 5:
        return {k: "—" for k in _EMPTY_KEYS}

    eq    = (1.0 + r).cumprod()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)

    total = float(eq.iloc[-1] - 1.0)
    cagr  = float(eq.iloc[-1] ** (1.0 / years) - 1.0)

    std    = float(r.std())
    ann_vol = std * np.sqrt(ppy)
    sharpe = float(r.mean() / std * np.sqrt(ppy)) if std > 1e-10 else np.nan

    down_r   = r[r < 0]
    down_std = float(down_r.std()) if len(down_r) > 1 else np.nan
    sortino  = float(r.mean() / down_std * np.sqrt(ppy)) if (down_std and down_std > 1e-10) else np.nan

    # ── Win/loss analytics ─────────────────────────────────────────────────
    wins   = r[r > 0]
    losses = r[r < 0]
    win_rate    = float((r > 0).mean())
    avg_win     = float(wins.mean())  if len(wins)   > 0 else np.nan
    avg_loss    = float(losses.mean()) if len(losses) > 0 else np.nan
    gross_wins  = float(wins.sum())  if len(wins)   > 0 else 0.0
    gross_losses= abs(float(losses.sum())) if len(losses) > 0 else 0.0
    profit_factor = gross_wins / gross_losses if gross_losses > 1e-10 else np.nan

    # ── Drawdown block ─────────────────────────────────────────────────────
    dds = _dd_stats(eq)
    max_dd   = dds["max_dd"]
    calmar   = float(cagr / abs(max_dd)) if max_dd < -1e-6 else np.nan
    rec_days = dds["recovery_days"]

    def _f(v, fmt: str) -> str:
        if v is None:
            return "still in DD"
        if isinstance(v, float) and np.isnan(v):
            return "—"
        return format(v, fmt)

    def _d(v) -> str:
        """Format an integer day count."""
        if v is None:
            return "still in DD"
        return f"{int(v):,}d"

    return {
        "name":             name,
        "Total Return":     _f(total,        ".1%"),
        "CAGR":             _f(cagr,         ".1%"),
        "Ann. Vol":         _f(ann_vol,      ".1%"),
        "Sharpe":           _f(sharpe,       ".2f"),
        "Sortino":          _f(sortino,      ".2f"),
        "Max DD":           _f(max_dd,       ".1%"),
        "Max DD Days":      _d(dds["max_dd_days"]),
        "Longest DD Days":  _d(dds["longest_dd_days"]),
        "Avg DD":           _f(dds["avg_dd"],".1%"),
        "DD Episodes":      str(dds["dd_episodes"]),
        "Recovery Days":    _d(rec_days),
        "Current DD":       _f(dds["current_dd"], ".1%"),
        "Calmar":           _f(calmar,        ".2f"),
        "Win Rate":         _f(win_rate,      ".1%"),
        "Avg Win":          _f(avg_win,       ".3%"),
        "Avg Loss":         _f(avg_loss,      ".3%"),
        "Profit Factor":    _f(profit_factor, ".2f"),
    }
