"""
scripts/smoke_test.py
─────────────────────────────────────────────────────────────────────────────
Smoke test: validates the full Python stack (no live data required).
Uses synthetic data to exercise every component end-to-end.

Usage:
    python scripts/smoke_test.py
"""

from __future__ import annotations
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd


# ── assert_ helper MUST be defined FIRST ─────────────────────────────────────

def assert_(cond: bool, msg: str = "assertion failed") -> None:
    if not cond:
        raise AssertionError(msg)


# ── Test runner ───────────────────────────────────────────────────────────────

results = []

def check(name: str, fn) -> bool:
    try:
        fn()
        print(f"  \u2713  {name}")
        results.append((name, True, None))
        return True
    except Exception as exc:
        print(f"  \u2717  {name}")
        print(f"       {type(exc).__name__}: {exc}")
        results.append((name, False, str(exc)))
        return False


# ── Synthetic data ────────────────────────────────────────────────────────────

def make_ohlcv(n: int = 400) -> pd.DataFrame:
    np.random.seed(42)
    idx  = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    rets = np.concatenate([
        np.random.normal( 0.003, 0.010, 100),
        np.random.normal( 0.000, 0.040, 100),
        np.random.normal(-0.003, 0.015, 100),
        np.random.normal( 0.000, 0.005, 100),
    ])
    close  = 30_000 * np.exp(np.cumsum(rets))
    high   = close * np.random.uniform(1.001, 1.020, n)
    low    = close * np.random.uniform(0.980, 0.999, n)
    volume = np.random.lognormal(10, 0.5, n)
    return pd.DataFrame({
        "open":           close * np.random.uniform(0.999, 1.001, n),
        "high":           high,
        "low":            low,
        "close":          close,
        "volume":         volume,
        "quote_volume":   volume * close,
        "taker_buy_base": volume * 0.52,
        "num_trades":     np.full(n, 10_000, dtype=int),
    }, index=idx)


df    = make_ohlcv()
feats = None
model = None

print()
print("╔══════════════════════════════════════════════════════╗")
print("║  NAUTILUS BTC  —  Stack Smoke Test                   ║")
print("╚══════════════════════════════════════════════════════╝")
print()


# ── Feature Engineering ───────────────────────────────────────────────────────

print("Feature Engineering:")

def _test_features_import():
    from python.core.features import build_features
    result = build_features(df, z_score=True)
    assert_(isinstance(result, pd.DataFrame), "not a DataFrame")

check("build_features returns DataFrame", _test_features_import)


def _test_feature_shape():
    global feats
    from python.core.features import build_features
    feats = build_features(df, z_score=True)
    assert_(feats.shape[0] > 200, f"rows={feats.shape[0]} expected >200")
    assert_(feats.shape[1] >= 9,  f"cols={feats.shape[1]} expected >=9")

check("feature shape (>200 rows, >=9 cols)", _test_feature_shape)


def _test_no_nans():
    assert_(feats is not None, "feats not built")
    assert_(not feats.isna().any().any(), "NaNs found")

check("no NaNs in features", _test_no_nans)


# ── HMM Model ─────────────────────────────────────────────────────────────────

print("\nHMM Model:")

def _test_model_fit():
    global model
    assert_(feats is not None and len(feats) > 0,
            "feats is empty — feature test must have failed")
    from python.core.hmm_model import RegimeHMM
    model = RegimeHMM(n_states=4, n_iter=40)
    model.fit(feats)
    assert_(model.is_fitted, "model.is_fitted is False after fit()")

check("model fits without error", _test_model_fit)


def _test_predict_shapes():
    assert_(model is not None and model.is_fitted, "model not fitted")
    states, posteriors = model.predict(feats)
    assert_(len(states) == len(feats),
            f"states len {len(states)} != feats len {len(feats)}")
    assert_(posteriors.shape == (len(feats), 4),
            f"posteriors shape {posteriors.shape}")

check("predict returns correct shapes", _test_predict_shapes)


def _test_posteriors_sum():
    assert_(model is not None and model.is_fitted, "model not fitted")
    _, posteriors = model.predict(feats)
    np.testing.assert_allclose(
        posteriors.sum(axis=1), np.ones(len(feats)), atol=1e-5
    )

check("posteriors sum to 1", _test_posteriors_sum)


def _test_transition_diag():
    assert_(model is not None and model.is_fitted, "model not fitted")
    diag = np.diag(model.transition_matrix)
    assert_(diag.mean() >= 0.5,
            f"diag mean {diag.mean():.2f} < 0.5")

check("transition matrix diagonal >= 0.5", _test_transition_diag)


# ── DataStore ─────────────────────────────────────────────────────────────────

print("\nDataStore:")

def _test_add_bar():
    from python.core.data_store import DataStore
    s = DataStore()
    s.add_bar_1s(
        ts_ms=1_700_000_000_000,
        open_=50000.0, high=50100.0, low=49900.0, close=50050.0,
        volume=1.5, buy_volume=0.9, sell_volume=0.6,
        num_trades=100, vwap=50025.0, is_complete=True,
    )
    bars = s.get_bars_1s()
    assert_(len(bars) == 1, f"expected 1 bar, got {len(bars)}")
    assert_(bars[0].close == 50050.0, "close price mismatch")

check("add and retrieve 1s bar", _test_add_bar)


def _test_incomplete_bar():
    from python.core.data_store import DataStore
    s = DataStore()
    s.add_bar_1s(
        ts_ms=1_700_000_000_000,
        open_=50000.0, high=50100.0, low=49900.0, close=50050.0,
        volume=1.5, buy_volume=0.9, sell_volume=0.6,
        num_trades=100, vwap=50025.0, is_complete=False,
    )
    # Incomplete bar NOT in deque, but IS in partial (visible on live chart)
    assert_(len(s._bars_1s) == 0, "incomplete bar must not be in completed deque")
    assert_(s._partial_bar is not None, "partial bar should be set for live display")

check("incomplete bar not stored", _test_incomplete_bar)


def _test_load_daily():
    from python.core.data_store import DataStore
    s = DataStore()
    s.load_daily_from_df(df)
    n = len(s.get_bars_daily())
    assert_(n == len(df), f"expected {len(df)} bars, got {n}")

check("load_daily_from_df", _test_load_daily)


# ── Backtest Engine ───────────────────────────────────────────────────────────

print("\nBacktest Engine:")

def _test_backtest_runs():
    assert_(model is not None and model.is_fitted, "model not fitted")
    from python.backtest.engine import run_backtest
    bnh, reg = run_backtest(df, model, feats)
    assert_(hasattr(bnh, "sharpe"), "bnh missing sharpe")
    assert_(hasattr(reg, "sharpe"), "regime missing sharpe")
    assert_(-1 <= bnh.max_dd <= 0, f"bnh.max_dd={bnh.max_dd:.3f}")
    assert_(-1 <= reg.max_dd <= 0, f"reg.max_dd={reg.max_dd:.3f}")

check("run_backtest returns PerformanceStats", _test_backtest_runs)


def _test_equity_starts_at_one():
    assert_(model is not None and model.is_fitted, "model not fitted")
    from python.backtest.engine import run_backtest
    bnh, reg = run_backtest(df, model, feats)
    assert_(abs(bnh.equity.iloc[0] - 1.0) < 0.01,
            f"bnh equity[0]={bnh.equity.iloc[0]:.4f}")
    assert_(abs(reg.equity.iloc[0] - 1.0) < 0.01,
            f"regime equity[0]={reg.equity.iloc[0]:.4f}")

check("equity curve starts at 1.0", _test_equity_starts_at_one)


# ── C++ Module (optional) ─────────────────────────────────────────────────────

print("\nC++ Module (optional):")
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "python" / "core"))
    import nautilus_cpp
    check("nautilus_cpp importable",
          lambda: assert_(nautilus_cpp is not None))
    check("version attribute exists",
          lambda: assert_(hasattr(nautilus_cpp, "__version__")))
    check("LWS flag attribute exists",
          lambda: assert_(hasattr(nautilus_cpp, "__built_with_lws__")))
except ImportError:
    print("  o  nautilus_cpp not compiled -- Python fallback will be used")


# ── Summary ───────────────────────────────────────────────────────────────────

n_pass = sum(1 for _, ok, _ in results if ok)
n_fail = sum(1 for _, ok, _ in results if not ok)

print(f"\n{'='*54}")
print(f"  Result: {n_pass} passed, {n_fail} failed")
if n_fail > 0:
    print("\n  Failed tests:")
    for name, ok, err in results:
        if not ok:
            print(f"    x {name}: {err}")
print(f"{'='*54}\n")

sys.exit(0 if n_fail == 0 else 1)
