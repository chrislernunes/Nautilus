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


def compute_metrics(
    returns: pd.Series,
    name: str = "Strategy",
    ppy: int  = 252,
) -> dict[str, str]:
    """Standard performance metrics from a daily return series."""
    r = returns.dropna()
    if len(r) <= 5:
        return {k: "—" for k in [
            "name", "Total Return", "CAGR", "Sharpe", "Sortino",
            "Max DD", "Calmar", "Win Rate",
        ]}

    eq    = (1.0 + r).cumprod()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)

    total = eq.iloc[-1] - 1.0
    cagr  = eq.iloc[-1] ** (1.0 / years) - 1.0

    std    = r.std()
    sharpe = r.mean() / std * np.sqrt(ppy) if std > 1e-10 else np.nan

    down_std = r[r < 0].std()
    sortino  = r.mean() / down_std * np.sqrt(ppy) if down_std > 1e-10 else np.nan

    dd     = (eq / eq.cummax() - 1)
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd < -1e-6 else np.nan

    win_rate = (r > 0).mean()

    def _f(v: float, fmt: str) -> str:
        return format(v, fmt) if not np.isnan(v) else "—"

    return {
        "name":         name,
        "Total Return": _f(total,    ".1%"),
        "CAGR":         _f(cagr,     ".1%"),
        "Sharpe":       _f(sharpe,   ".2f"),
        "Sortino":      _f(sortino,  ".2f"),
        "Max DD":       _f(max_dd,   ".1%"),
        "Calmar":       _f(calmar,   ".2f"),
        "Win Rate":     _f(win_rate, ".1%"),
    }
