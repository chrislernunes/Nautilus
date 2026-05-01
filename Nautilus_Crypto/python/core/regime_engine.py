"""
python/core/regime_engine.py
─────────────────────────────────────────────────────────────────────────────
Regime detection orchestrator — background daemon thread.

Steps:
  1. Fetch historical OHLCV → build features → fit HMM
  2. Tag all historical daily bars with regime labels
  3. Real-time update loop: re-predict every update_interval_s seconds
  4. Periodic rolling refit every refit_interval_bars new daily bars
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd

from python.core.data_store import DataStore
from python.core.features   import build_features
from python.core.hmm_model  import RegimeHMM, RegimeState
from python.core.historical import warm_up_max as warm_up
from python.backtest.engine import run_backtest, optimise_hyperparams, HyperParams

log = logging.getLogger("nautilus.regime_engine")


class RegimeEngine:
    def __init__(
        self,
        store:               DataStore,
        model_path:          Optional[Path] = None,
        update_interval_s:   float = 5.0,
        refit_interval_bars: int   = 30,
        run_optimiser:       bool  = False,
    ) -> None:
        self.store              = store
        self.model_path         = model_path
        self.update_interval_s  = update_interval_s
        self.refit_interval_bars = refit_interval_bars
        self.run_optimiser      = run_optimiser

        self._model: Optional[RegimeHMM] = None
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._n_daily_bars_at_last_refit = 0

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(
            target=self._run,
            name="regime-engine",
            daemon=True,
        )
        self._thread.start()
        log.info("RegimeEngine started")

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=10)

    @property
    def model(self) -> Optional[RegimeHMM]:
        return self._model

    def _run(self) -> None:
        self.store.set_status(model_fitted=False)
        self.store.set_status(error_msg="Warming up historical data...")

        try:
            daily_df, h4_df = asyncio.run(warm_up())
            self.store.load_daily_from_df(daily_df)
            log.info("Historical data loaded: %d daily bars", len(daily_df))
        except Exception as exc:
            log.error("Warm-up failed: %s", exc)
            self.store.set_status(error_msg=f"Warm-up error: {exc}")
            daily_df = pd.DataFrame()

        if daily_df.empty:
            log.error("No historical data — regime engine cannot initialise")
            self.store.set_status(error_msg="No historical data available")
            return

        self.store.set_status(error_msg="Building features...")
        try:
            features = build_features(daily_df, z_score=True)
            if features.empty:
                raise ValueError("Feature matrix is empty")
        except Exception as exc:
            log.error("Feature engineering failed: %s", exc)
            self.store.set_status(error_msg=f"Feature error: {exc}")
            return

        best_hp = HyperParams()
        if self.run_optimiser:
            self.store.set_status(error_msg="Running WF optimisation (~30s)...")
            try:
                best_hp, _ = optimise_hyperparams(daily_df, n_splits=3)
                log.info("Optimisation complete. Best: %s", best_hp.label())
            except Exception as exc:
                log.warning("Optimisation failed: %s — using defaults", exc)

        self.store.set_status(error_msg="Fitting HMM model...")

        # Build model (without model_path first so we can check feature compat)
        self._model = RegimeHMM(
            n_states        = best_hp.n_states,
            covariance_type = best_hp.covariance_type,
            n_iter          = 200,
            refit_every     = self.refit_interval_bars,
            model_path      = self.model_path,
        )

        # Stale-cache check: if loaded model was trained on different features,
        # delete it and refit from scratch.
        if (self._model.is_fitted and
                self._model._feature_names != list(features.columns)):
            log.warning(
                "Cached model feature mismatch — deleting stale pickle and refitting.\n"
                "  Cached: %s\n  Current: %s",
                self._model._feature_names, list(features.columns),
            )
            if self.model_path:
                try:
                    self.model_path.unlink(missing_ok=True)
                except Exception:
                    pass
            self._model = RegimeHMM(
                n_states        = best_hp.n_states,
                covariance_type = best_hp.covariance_type,
                n_iter          = 200,
                refit_every     = self.refit_interval_bars,
            )

        if not self._model.is_fitted:
            try:
                self._model.fit(features)
                if self.model_path:
                    self._model.model_path = self.model_path
                    self._model.save(self.model_path)
            except Exception as exc:
                log.error("Model fit failed: %s", exc)
                self.store.set_status(error_msg=f"Model fit error: {exc}")
                return

        self.store.set_status(error_msg="Running backtest...")
        try:
            bnh, regime = run_backtest(daily_df, self._model, features)
            self.store.set_perf(bnh, regime)
            log.info("Backtest: B&H Sharpe=%.2f  Regime Sharpe=%.2f",
                     bnh.sharpe, regime.sharpe)
        except Exception as exc:
            log.warning("Backtest failed: %s", exc)

        # Tag historical daily bars
        try:
            states, _ = self._model.predict(features)
            regime_ts = features.index
            ts_to_state = {ts: int(s) for ts, s in zip(regime_ts, states)}

            self.store.load_daily_from_df(
                daily_df,
                regimes=[ts_to_state.get(ts) for ts in daily_df.index],
            )

            with self.store._lock:
                self.store._regime_history = [
                    (ts, ts_to_state[ts]) for ts in regime_ts
                    if ts in ts_to_state
                ]
        except Exception as exc:
            log.warning("Historical regime tagging failed: %s", exc)

        self._n_daily_bars_at_last_refit = len(daily_df)
        self.store.set_status(model_fitted=True, error_msg="")

        log.info("RegimeEngine entering real-time update loop")
        while self._running.is_set():
            try:
                self._update_regime()
            except Exception as exc:
                log.error("Regime update error: %s", exc)
            time.sleep(self.update_interval_s)

    def _update_regime(self) -> None:
        daily_df = self.store.get_daily_df()
        if len(daily_df) < 60:
            return

        try:
            features = build_features(daily_df, z_score=True)
            if features.empty or len(features) < 5:
                return

            state = self._model.predict_latest(features)
            state.timestamp = daily_df.index[-1]
            self.store.set_regime(state, ts=state.timestamp)

            n_bars = len(daily_df)
            if (n_bars - self._n_daily_bars_at_last_refit) >= self.refit_interval_bars:
                log.info("Triggered rolling refit (%d new daily bars)",
                         n_bars - self._n_daily_bars_at_last_refit)
                self._model.fit(features)
                bnh, regime = run_backtest(daily_df, self._model, features)
                self.store.set_perf(bnh, regime)
                self._n_daily_bars_at_last_refit = n_bars

        except Exception as exc:
            log.debug("_update_regime error: %s", exc)
