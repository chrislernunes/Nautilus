"""
python/core/hmm_model.py
─────────────────────────────────────────────────────────────────────────────
Gaussian HMM Regime Detection Engine

4 hidden states:
  State 0 — BULL_TREND    : positive returns, low vol, high efficiency
  State 1 — HIGH_VOL      : elevated vol, mean-reverting, risk-off
  State 2 — BEAR_TREND    : negative returns, rising vol, drawdown
  State 3 — LOW_VOL_CHOP  : low vol, low efficiency, sideways

Transition matrix diagonal initialised at 0.88 (not 0.95):
  - 0.95 was too sticky — model could not exit a wrongly-assigned BULL state
    during multi-week BTC bear runs (e.g. 2022 crash shown as BULL TREND).
  - 0.88 is still persistent (crypto regimes last 1-3 weeks) but allows
    Viterbi to transition within a few days of a genuine regime shift.

n_iter = 200 for better EM convergence on the larger feature set.
"""

from __future__ import annotations

import logging
import pickle
import threading
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

log = logging.getLogger("nautilus.hmm")


class Regime(IntEnum):
    BULL_TREND   = 0
    HIGH_VOL     = 1
    BEAR_TREND   = 2
    LOW_VOL_CHOP = 3


REGIME_LABELS = {
    Regime.BULL_TREND:   "BULL TREND",
    Regime.HIGH_VOL:     "HIGH VOL",
    Regime.BEAR_TREND:   "BEAR TREND",
    Regime.LOW_VOL_CHOP: "CHOP / LOW VOL",
}

REGIME_COLORS = {
    Regime.BULL_TREND:   "#00d084",
    Regime.HIGH_VOL:     "#ff6b35",
    Regime.BEAR_TREND:   "#ff3358",
    Regime.LOW_VOL_CHOP: "#8b9ab5",
}

REGIME_SIGNAL = {
    Regime.BULL_TREND:   1,
    Regime.HIGH_VOL:     0,
    Regime.BEAR_TREND:   0,
    Regime.LOW_VOL_CHOP: 1,
}


@dataclass
class RegimeState:
    regime:         Regime
    label:          str
    color:          str
    signal:         int
    probabilities:  Dict[str, float]
    confidence:     float
    timestamp:      Optional[pd.Timestamp] = None


class RegimeHMM:
    """
    Production Gaussian HMM with 4 states.

    Parameters
    ----------
    n_states         : number of hidden states (default 4)
    covariance_type  : "full", "diag", "tied", "spherical" (default "diag")
    n_iter           : EM iterations (default 200)
    refit_every      : refit model every N new observations (default 20)
    min_fit_obs      : minimum observations before first fit (default 120)
    model_path       : optional path to persist/load model
    """

    def __init__(
        self,
        n_states:        int   = 4,
        covariance_type: str   = "diag",
        n_iter:          int   = 200,
        refit_every:     int   = 20,
        min_fit_obs:     int   = 120,
        model_path:      Optional[Path] = None,
    ) -> None:
        self.n_states        = n_states
        self.covariance_type = covariance_type
        self.n_iter          = n_iter
        self.refit_every     = refit_every
        self.min_fit_obs     = min_fit_obs
        self.model_path      = model_path

        self._model:     Optional[GaussianHMM] = None
        self._state_map: Dict[int, Regime]     = {}
        self._fitted     = False
        self._obs_count  = 0
        self._lock       = threading.Lock()
        self._feature_names: List[str] = []

        if model_path and model_path.exists():
            self.load(model_path)

    def fit(self, features: pd.DataFrame) -> "RegimeHMM":
        X = features.values.astype(np.float64)
        n, d = X.shape

        log.info("Fitting HMM: %d obs × %d features, n_states=%d",
                 n, d, self.n_states)

        model = GaussianHMM(
            n_components     = self.n_states,
            covariance_type  = self.covariance_type,
            n_iter           = self.n_iter,
            tol              = 1e-4,
            params           = "stmc",
            init_params      = "mc",
            random_state     = 42,
            verbose          = False,
        )

        # Transition matrix: 0.88 diagonal
        # Persistent enough for crypto weekly trends but not so sticky that
        # the model misses extended bear/high-vol regimes.
        _diag = 0.88
        _off  = (1.0 - _diag) / max(1, self.n_states - 1)
        trans_init = np.full((self.n_states, self.n_states), _off)
        np.fill_diagonal(trans_init, _diag)
        model.transmat_  = trans_init
        model.startprob_ = np.ones(self.n_states) / self.n_states

        try:
            model.fit(X)
        except Exception as exc:
            log.error("HMM fit failed: %s", exc)
            return self

        with self._lock:
            self._model         = model
            self._feature_names = list(features.columns)
            self._state_map     = self._assign_regime_labels(model, features)
            self._fitted        = True
            self._obs_count     = n

        if self.model_path:
            self.save(self.model_path)

        log.info("HMM fitted. State map: %s",
                 {k: v.name for k, v in self._state_map.items()})
        log.info("Transition matrix diagonal: %s",
                 np.round(np.diag(model.transmat_), 4))
        return self

    def predict(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("Model not fitted yet")

        X = features.values.astype(np.float64)

        with self._lock:
            raw_states = self._model.predict(X)
            posteriors = self._model.predict_proba(X)

        mapped = np.array([self._state_map[s].value for s in raw_states])
        return mapped, posteriors

    def predict_latest(self, features: pd.DataFrame) -> RegimeState:
        if not self._fitted:
            return RegimeState(
                regime        = Regime.LOW_VOL_CHOP,
                label         = "INITIALISING",
                color         = "#8b9ab5",
                signal        = 0,
                probabilities = {r.name: 0.25 for r in Regime},
                confidence    = 0.25,
            )

        states, posteriors = self.predict(features)
        last_state         = Regime(states[-1])
        last_post          = posteriors[-1]

        prob_dict: Dict[str, float] = {}
        with self._lock:
            for raw_idx in range(self.n_states):
                mapped_regime = self._state_map.get(raw_idx, Regime(raw_idx % 4))
                prob_dict[mapped_regime.name] = float(last_post[raw_idx])

        return RegimeState(
            regime        = last_state,
            label         = REGIME_LABELS[last_state],
            color         = REGIME_COLORS[last_state],
            signal        = REGIME_SIGNAL[last_state],
            probabilities = prob_dict,
            confidence    = float(last_post.max()),
        )

    def update_online(
        self,
        new_features: pd.DataFrame,
        rolling_window: int = 500,
    ) -> bool:
        self._obs_count += len(new_features)
        if self._obs_count % self.refit_every == 0:
            if len(new_features) >= self.min_fit_obs:
                log.info("Online refit triggered (obs=%d)", self._obs_count)
                self.fit(new_features.tail(rolling_window))
                return True
        return False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def transition_matrix(self) -> Optional[np.ndarray]:
        if self._model:
            return self._model.transmat_
        return None

    def save(self, path: Path) -> None:
        payload = {
            "model":          self._model,
            "state_map":      self._state_map,
            "feature_names":  self._feature_names,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        log.info("Model saved to %s", path)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self._model         = payload["model"]
        self._state_map     = payload["state_map"]
        self._feature_names = payload["feature_names"]
        self._fitted        = True
        log.info("Model loaded from %s", path)

    def _assign_regime_labels(
        self,
        model:    GaussianHMM,
        features: pd.DataFrame,
    ) -> Dict[int, Regime]:
        means = model.means_
        cols  = list(features.columns)

        ret_idx = cols.index("log_ret_1d") if "log_ret_1d" in cols else 0
        vol_idx = cols.index("rv_5d")      if "rv_5d"      in cols else min(1, len(cols)-1)

        # Also use drawdown_60d for bear detection if available
        dd_idx  = cols.index("drawdown_60d") if "drawdown_60d" in cols else None

        ret_means = means[:, ret_idx]
        vol_means = means[:, vol_idx]

        states    = list(range(self.n_states))
        state_map: Dict[int, Regime] = {}

        # Bull: highest return mean
        bull = int(np.argmax(ret_means))
        state_map[bull] = Regime.BULL_TREND
        states.remove(bull)

        if not states:
            return state_map

        # Bear: lowest return mean (among remaining)
        # If drawdown feature available, break ties by most negative drawdown
        rem_rets = [(s, ret_means[s]) for s in states]
        if dd_idx is not None:
            dd_means  = means[:, dd_idx]
            # score: combine return rank and drawdown rank
            min_ret = min(r for _, r in rem_rets)
            bear = min(
                rem_rets,
                key=lambda x: x[1] + 0.3 * float(dd_means[x[0]]),
            )[0]
        else:
            bear = min(rem_rets, key=lambda x: x[1])[0]
        state_map[bear] = Regime.BEAR_TREND
        states.remove(bear)

        if not states:
            return state_map

        # High vol: highest vol mean (among remaining)
        rem_vols = [(s, vol_means[s]) for s in states]
        high_vol = max(rem_vols, key=lambda x: x[1])[0]
        state_map[high_vol] = Regime.HIGH_VOL
        states.remove(high_vol)

        for s in states:
            state_map[s] = Regime.LOW_VOL_CHOP

        return state_map
