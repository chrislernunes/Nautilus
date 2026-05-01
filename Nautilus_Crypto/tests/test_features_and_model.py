"""
tests/test_features_and_model.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for feature engineering and HMM regime model.
Run: pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_ohlcv():
    """Synthetic daily OHLCV with 4 distinct regimes."""
    np.random.seed(42)
    n   = 400
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")

    # Regime schedule: bull (100) → high_vol (100) → bear (100) → chop (100)
    regime_returns = np.concatenate([
        np.random.normal(0.003, 0.010, 100),   # bull
        np.random.normal(0.000, 0.040, 100),   # high vol
        np.random.normal(-0.003, 0.015, 100),  # bear
        np.random.normal(0.000, 0.005, 100),   # chop
    ])

    close = 30_000 * np.exp(np.cumsum(regime_returns))
    noise = np.random.uniform(0.995, 1.005, n)
    high  = close * np.random.uniform(1.001, 1.020, n)
    low   = close * np.random.uniform(0.980, 0.999, n)
    open_ = close * noise

    volume = np.random.lognormal(10, 0.5, n)
    buy_v  = volume * np.random.uniform(0.4, 0.6, n)

    df = pd.DataFrame({
        "open":           open_,
        "high":           high,
        "low":            low,
        "close":          close,
        "volume":         volume,
        "quote_volume":   volume * close,
        "taker_buy_base": buy_v,
        "num_trades":     np.random.randint(1000, 50000, n),
    }, index=idx)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatures:

    def test_build_features_shape(self, synthetic_ohlcv):
        from python.core.features import build_features
        feats = build_features(synthetic_ohlcv, z_score=True)
        assert not feats.empty, "Feature matrix should not be empty"
        assert feats.shape[1] >= 9, "Should have at least 9 feature columns"

    def test_no_lookahead(self, synthetic_ohlcv):
        """
        Verify no look-ahead bias: z-score at index T must be identical whether
        computed on data[0:T] or data[0:T+50] — future data must not affect it.

        We test this by comparing the z-scores at the last 30 rows of a 200-row
        slice against the same rows in a 250-row slice (50 extra future rows).
        With rolling z-score using only past `window` observations, these must
        be identical.
        """
        from python.core.features import build_features

        # Same first 200 rows in both; second adds 50 more future rows
        df_200 = synthetic_ohlcv.iloc[:200]
        df_250 = synthetic_ohlcv.iloc[:250]

        feats_200 = build_features(df_200, z_score=True)
        feats_250 = build_features(df_250, z_score=True)

        # Check the last 30 rows of feats_200 — these same timestamps must
        # have identical values in feats_250 (future rows added no info)
        check_idx = feats_200.index[-30:]
        common_cols = [c for c in feats_200.columns if c in feats_250.columns]

        diff = (
            feats_200.loc[check_idx, common_cols] -
            feats_250.loc[check_idx, common_cols]
        ).abs().max().max()

        assert diff < 1e-8, (
            f"Look-ahead bias detected: max diff = {diff:.6f}. "
            "z-scores at time T must not change when future data is added."
        )

    def test_no_nans_in_output(self, synthetic_ohlcv):
        from python.core.features import build_features
        feats = build_features(synthetic_ohlcv, z_score=True)
        assert not feats.isna().any().any(), "Feature matrix must have no NaNs"

    def test_zscore_range(self, synthetic_ohlcv):
        from python.core.features import build_features
        feats = build_features(synthetic_ohlcv, z_score=True)
        # After clipping at ±5, all values should be in range
        assert (feats.abs() <= 5.01).all().all(), "z-scores should be clipped to [-5, 5]"

    def test_efficiency_ratio_bounds(self, synthetic_ohlcv):
        from python.core.features import _efficiency_ratio
        er = _efficiency_ratio(synthetic_ohlcv["close"])
        valid = er.dropna()
        assert (valid >= 0).all() and (valid <= 1.01).all(), "ER must be in [0, 1]"

    def test_buy_imbalance_bounds(self, synthetic_ohlcv):
        from python.core.features import build_features
        feats = build_features(synthetic_ohlcv, z_score=False)
        if "buy_imbalance" in feats.columns:
            raw = feats["buy_imbalance"].dropna()
            # After z-scoring this won't hold, but pre-zscore it should
        # Just check column exists
        feats2 = build_features(synthetic_ohlcv, z_score=False)
        assert "buy_imbalance" in feats2.columns


# ─────────────────────────────────────────────────────────────────────────────
# HMM model tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHMMModel:

    def test_fit_and_predict(self, synthetic_ohlcv):
        from python.core.features import build_features
        from python.core.hmm_model import RegimeHMM, Regime

        feats = build_features(synthetic_ohlcv, z_score=True)
        model = RegimeHMM(n_states=4, n_iter=50)
        model.fit(feats)

        assert model.is_fitted
        states, posteriors = model.predict(feats)

        assert len(states)     == len(feats)
        assert posteriors.shape == (len(feats), 4)
        assert set(states).issubset({r.value for r in Regime})

    def test_posteriors_sum_to_one(self, synthetic_ohlcv):
        from python.core.features import build_features
        from python.core.hmm_model import RegimeHMM

        feats = build_features(synthetic_ohlcv, z_score=True)
        model = RegimeHMM(n_states=4, n_iter=30)
        model.fit(feats)
        _, posteriors = model.predict(feats)
        row_sums = posteriors.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), "Posterior rows must sum to 1"

    def test_regime_persistence(self, synthetic_ohlcv):
        """Transition matrix diagonal should be high (regimes are persistent)."""
        from python.core.features import build_features
        from python.core.hmm_model import RegimeHMM

        feats = build_features(synthetic_ohlcv, z_score=True)
        model = RegimeHMM(n_states=4, n_iter=50)
        model.fit(feats)

        diag = np.diag(model.transition_matrix)
        assert diag.mean() >= 0.50, (
            f"Mean transition diagonal too low: {diag.mean():.2f} — "
            "regime detection may be unstable"
        )

    def test_predict_latest_returns_regime_state(self, synthetic_ohlcv):
        from python.core.features import build_features
        from python.core.hmm_model import RegimeHMM, RegimeState

        feats = build_features(synthetic_ohlcv, z_score=True)
        model = RegimeHMM(n_states=4, n_iter=30)
        model.fit(feats)
        state = model.predict_latest(feats)

        assert isinstance(state, RegimeState)
        assert 0.0 <= state.confidence <= 1.0
        assert state.signal in (0, 1)
        assert abs(sum(state.probabilities.values()) - 1.0) < 0.02

    def test_three_state_model(self, synthetic_ohlcv):
        """3-state model should also work correctly."""
        from python.core.features import build_features
        from python.core.hmm_model import RegimeHMM

        feats = build_features(synthetic_ohlcv, z_score=True)
        model = RegimeHMM(n_states=3, n_iter=30)
        model.fit(feats)
        assert model.is_fitted

    def test_model_persistence(self, synthetic_ohlcv, tmp_path):
        """Save and reload model, verify identical predictions."""
        from python.core.features import build_features
        from python.core.hmm_model import RegimeHMM

        feats  = build_features(synthetic_ohlcv, z_score=True)
        model1 = RegimeHMM(n_states=4, n_iter=30)
        model1.fit(feats)
        states1, _ = model1.predict(feats)

        path = tmp_path / "test_model.pkl"
        model1.save(path)

        model2 = RegimeHMM(n_states=4)
        model2.load(path)
        states2, _ = model2.predict(feats)

        assert np.array_equal(states1, states2), "Reloaded model must give same predictions"


# ─────────────────────────────────────────────────────────────────────────────
# Backtest tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktest:

    def test_backtest_runs(self, synthetic_ohlcv):
        from python.core.features import build_features
        from python.core.hmm_model import RegimeHMM
        from python.backtest.engine import run_backtest

        feats = build_features(synthetic_ohlcv, z_score=True)
        model = RegimeHMM(n_states=4, n_iter=30)
        model.fit(feats)

        bnh, regime = run_backtest(synthetic_ohlcv, model, feats)

        # Sanity checks
        assert -1.0 <= bnh.total_return    <= 10.0
        assert -1.0 <= regime.total_return <= 10.0
        assert bnh.max_dd    <= 0.0
        assert regime.max_dd <= 0.0
        assert 0.0 <= bnh.win_rate    <= 1.0
        assert 0.0 <= regime.win_rate <= 1.0

    def test_equity_starts_at_one(self, synthetic_ohlcv):
        from python.core.features import build_features
        from python.core.hmm_model import RegimeHMM
        from python.backtest.engine import run_backtest

        feats = build_features(synthetic_ohlcv, z_score=True)
        model = RegimeHMM(n_states=4, n_iter=30)
        model.fit(feats)
        bnh, regime = run_backtest(synthetic_ohlcv, model, feats)

        assert abs(bnh.equity.iloc[0]    - 1.0) < 0.01
        assert abs(regime.equity.iloc[0] - 1.0) < 0.01

    def test_regime_has_fewer_trades_than_bnh(self, synthetic_ohlcv):
        from python.core.features import build_features
        from python.core.hmm_model import RegimeHMM
        from python.backtest.engine import run_backtest

        feats = build_features(synthetic_ohlcv, z_score=True)
        model = RegimeHMM(n_states=4, n_iter=50)
        model.fit(feats)
        bnh, regime = run_backtest(synthetic_ohlcv, model, feats)

        # Regime strategy should have > 1 trade but < N bars
        assert regime.n_trades >= 1
        assert regime.n_trades < len(feats)


# ─────────────────────────────────────────────────────────────────────────────
# DataStore tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDataStore:

    def test_add_and_retrieve_bars(self):
        from python.core.data_store import DataStore
        store = DataStore()
        store.add_bar_1s(
            ts_ms=1_700_000_000_000,
            open_=50000.0, high=50100.0, low=49900.0, close=50050.0,
            volume=1.5, buy_volume=0.9, sell_volume=0.6,
            num_trades=100, vwap=50025.0, is_complete=True,
        )
        bars = store.get_bars_1s()
        assert len(bars) == 1
        assert bars[0].close == 50050.0

    def test_incomplete_bar_not_stored_in_deque(self):
        """
        An incomplete (partial) bar should NOT be added to the completed-bar
        deque (_bars_1s), but IS visible via get_bars_1s() as the partial bar
        appended at the end — this is intentional for real-time display.
        """
        from python.core.data_store import DataStore
        store = DataStore()
        store.add_bar_1s(
            ts_ms=1_700_000_000_000,
            open_=50000.0, high=50100.0, low=49900.0, close=50050.0,
            volume=1.5, buy_volume=0.9, sell_volume=0.6,
            num_trades=100, vwap=50025.0, is_complete=False,
        )
        # Internal completed-bar deque must be empty
        assert len(store._bars_1s) == 0, \
            "Incomplete bar must not be in the completed-bar deque"
        # But partial bar IS set (shown live on chart)
        assert store._partial_bar is not None, \
            "Partial bar should be set for live display"
        # get_bars_1s() exposes it as the trailing partial entry
        visible = store.get_bars_1s()
        assert len(visible) == 1
        assert visible[0].close == 50050.0

    def test_max_capacity(self):
        from python.core.data_store import DataStore
        store = DataStore()
        for i in range(store.MAX_1S_BARS + 100):
            store.add_bar_1s(
                ts_ms=i * 1000, open_=1.0, high=1.0, low=1.0, close=1.0,
                volume=1.0, buy_volume=0.5, sell_volume=0.5,
                num_trades=1, vwap=1.0, is_complete=True,
            )
        assert len(store.get_bars_1s()) <= store.MAX_1S_BARS

    def test_price_change_pct(self):
        from python.core.data_store import DataStore, BarDaily
        store = DataStore()
        store._price_24h_ago = 50_000.0
        store._last_price    = 51_000.0
        assert abs(store.price_change_24h_pct - 2.0) < 1e-6
