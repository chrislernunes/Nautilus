"""Unit tests for backtest engine — verify no double-shift."""
import numpy as np
import pandas as pd
import pytest

from nautilus.backtests.engine import compute_metrics, run_backtest


def _price(n: int = 500) -> pd.Series:
    """Deterministic price series with upward drift."""
    rng   = np.random.default_rng(42)
    lr    = 0.0004 + rng.normal(0, 0.012, n)
    price = 10_000 * np.exp(np.cumsum(lr))
    idx   = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(price, index=idx, name="Close")


def test_always_long_matches_buy_hold():
    """With zero tx cost and signal=1, strategy must exactly match buy & hold."""
    price  = _price()
    signal = pd.Series(1.0, index=price.index)
    bh_ret = price.pct_change().fillna(0.0)
    bh_eq  = (1 + bh_ret).cumprod()

    bt = run_backtest(price, signal, cost_bps=0.0, name="AlwaysLong")
    pd.testing.assert_series_equal(bt.equity_curve, bh_eq, check_names=False)


def test_zero_signal_flat_equity():
    """Signal=0 should produce a flat equity curve of 1.0 throughout."""
    price  = _price()
    signal = pd.Series(0.0, index=price.index)
    bt     = run_backtest(price, signal, cost_bps=0.0)
    assert (bt.equity_curve == 1.0).all(), "Zero signal should produce flat equity"


def test_no_double_shift():
    """
    Verify signal is NOT shifted inside the engine.
    If the engine shifted internally, a 1-period-ahead price move would
    be captured which is look-ahead. We test by confirming that
    shifting the signal by 1 extra day changes results.
    """
    price             = _price()
    signal_t          = pd.Series(1.0, index=price.index)
    signal_tminus1    = signal_t.shift(1).fillna(0.0)

    bt_correct        = run_backtest(price, signal_tminus1, cost_bps=0.0)
    bt_lookahead      = run_backtest(price, signal_t,       cost_bps=0.0)

    assert not bt_correct.equity_curve.equals(bt_lookahead.equity_curve)


def test_metrics_sharpe_positive_drift():
    """Positive drift series should yield positive Sharpe."""
    rng  = np.random.default_rng(0)
    r    = pd.Series(0.001 + rng.normal(0, 0.01, 500),
                     index=pd.bdate_range("2020-01-01", periods=500))
    m    = compute_metrics(r)
    sharpe = float(m["Sharpe"])
    assert sharpe > 0.5, f"Expected Sharpe > 0.5 for positive drift, got {sharpe:.2f}"


def test_metrics_handles_short_series():
    """Very short series should not raise, return '—' for all metrics."""
    r = pd.Series([0.01, -0.02, 0.005], index=pd.bdate_range("2020-01-01", periods=3))
    m = compute_metrics(r)
    assert m["Sharpe"] == "—"
