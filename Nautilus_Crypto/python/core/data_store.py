"""
python/core/data_store.py
─────────────────────────────────────────────────────────────────────────────
Thread-safe shared data store.

Key fixes vs previous version:
  - last_price updated on EVERY tick (not just bar close)
  - Partial (in-progress) 1s bar exposed via get_partial_bar_1s()
  - price_24h_ago falls back to last daily close when bar history is short
  - get_bars_1s() returns copy so dashboard never races the writer
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import pandas as pd

from python.core.hmm_model import RegimeState, Regime


# ─────────────────────────────────────────────────────────────────────────────
# Bar containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Bar1s:
    ts_ms:       int
    open:        float
    high:        float
    low:         float
    close:       float
    volume:      float
    buy_volume:  float
    sell_volume: float
    num_trades:  int
    vwap:        float
    regime:      Optional[int] = None


@dataclass
class BarDaily:
    ts_ms:           int
    open:            float
    high:            float
    low:             float
    close:           float
    volume:          float
    quote_volume:    float
    taker_buy_base:  float
    num_trades:      int
    regime:          Optional[int] = None


@dataclass
class SystemStatus:
    ws_connected:    bool  = False
    cpp_loaded:      bool  = False
    model_fitted:    bool  = False
    msg_per_sec:     float = 0.0
    last_update_ts:  float = 0.0
    error_msg:       str   = ""


# ─────────────────────────────────────────────────────────────────────────────
# DataStore
# ─────────────────────────────────────────────────────────────────────────────

class DataStore:
    MAX_1S_BARS    = 3_600
    MAX_DAILY_BARS = 6_000   # enough for full BTC history from 2010 via yfinance

    def __init__(self) -> None:
        self._lock         = threading.RLock()
        self._bars_1s:     Deque[Bar1s]    = deque(maxlen=self.MAX_1S_BARS)
        self._bars_daily:  Deque[BarDaily] = deque(maxlen=self.MAX_DAILY_BARS)

        self._last_price:    float = 0.0   # updated on EVERY tick
        self._price_24h_ago: float = 0.0

        # Partial (in-progress) bar — updated every trade tick
        self._partial_bar: Optional[Bar1s] = None

        self._regime_state: Optional[RegimeState] = None
        self._status:       SystemStatus = SystemStatus()

        self._perf_bnh    = None
        self._perf_regime = None

        self._regime_history: List[Tuple] = []

        # Rate tracking
        self._rate_count:   int   = 0
        self._last_rate_ts: float = time.monotonic()

    # ── Writers ───────────────────────────────────────────────────────────────

    def update_tick(self, price: float) -> None:
        """
        Called on EVERY trade (not just bar close).
        Keeps last_price fresh even between bar boundaries.
        """
        with self._lock:
            self._last_price = price
            self._rate_count += 1
            now = time.monotonic()
            dt  = now - self._last_rate_ts
            if dt >= 1.0:
                self._status.msg_per_sec = self._rate_count / dt
                self._rate_count         = 0
                self._last_rate_ts       = now
            self._status.last_update_ts = now

    def update_partial_bar(
        self,
        ts_ms: int, open_: float, high: float, low: float, close: float,
        volume: float, buy_volume: float, sell_volume: float,
        num_trades: int, vwap: float,
    ) -> None:
        """Update the in-progress (incomplete) 1s bar shown on the live chart."""
        with self._lock:
            self._partial_bar = Bar1s(
                ts_ms=ts_ms, open=open_, high=high, low=low, close=close,
                volume=volume, buy_volume=buy_volume, sell_volume=sell_volume,
                num_trades=num_trades, vwap=vwap,
                regime=self._regime_state.regime.value
                       if self._regime_state else None,
            )
            self._last_price = close  # also keep last_price current

    def add_bar_1s(
        self,
        ts_ms: int, open_: float, high: float, low: float, close: float,
        volume: float, buy_volume: float, sell_volume: float,
        num_trades: int, vwap: float, is_complete: bool,
    ) -> None:
        """Called when a 1s bar is finalised (complete=True only stored)."""
        if not is_complete:
            # Still update partial bar display and last price
            self.update_partial_bar(ts_ms, open_, high, low, close,
                                    volume, buy_volume, sell_volume,
                                    num_trades, vwap)
            return

        with self._lock:
            self._last_price = close
            bar = Bar1s(
                ts_ms=ts_ms, open=open_, high=high, low=low, close=close,
                volume=volume, buy_volume=buy_volume, sell_volume=sell_volume,
                num_trades=num_trades, vwap=vwap,
                regime=self._regime_state.regime.value
                       if self._regime_state else None,
            )
            self._bars_1s.append(bar)
            # Clear partial bar — it's now a complete bar
            self._partial_bar = None

    def add_daily_bar(self, bar: BarDaily) -> None:
        with self._lock:
            self._bars_daily.append(bar)
            bars = list(self._bars_daily)
        # Update 24h reference price
        if len(bars) >= 2:
            self._price_24h_ago = bars[-2].close

    def set_regime(self, state: RegimeState,
                   ts: Optional[pd.Timestamp] = None) -> None:
        with self._lock:
            self._regime_state = state
            if ts is not None:
                self._regime_history.append((ts, state.regime.value))
                if len(self._regime_history) > self.MAX_DAILY_BARS:
                    self._regime_history = self._regime_history[-self.MAX_DAILY_BARS:]

    def set_status(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                setattr(self._status, k, v)

    def set_perf(self, bnh, regime) -> None:
        with self._lock:
            self._perf_bnh    = bnh
            self._perf_regime = regime

    def load_daily_from_df(self, df: "pd.DataFrame",
                        regimes: "Optional[List[int]]" = None) -> None:
        with self._lock:
            self._bars_daily.clear()
            for i, (ts, row) in enumerate(df.iterrows()):
                bar = BarDaily(
                    ts_ms          = int(ts.timestamp() * 1000),
                    open           = float(row.get("open",            0)),
                    high           = float(row.get("high",            0)),
                    low            = float(row.get("low",             0)),
                    close          = float(row.get("close",           0)),
                    volume         = float(row.get("volume",          0)),
                    quote_volume   = float(row.get("quote_volume",    0)),
                    taker_buy_base = float(row.get("taker_buy_base",  0)),
                    num_trades     = int(row.get("num_trades",         0)),
                    regime         = regimes[i] if regimes and i < len(regimes) else None,
                )
                self._bars_daily.append(bar)
            bars = list(self._bars_daily)

        if len(bars) >= 2:
            self._price_24h_ago = bars[-2].close
        elif len(bars) == 1:
            self._price_24h_ago = bars[0].close
            self._last_price    = bars[0].close

    # ── Readers ───────────────────────────────────────────────────────────────

    @property
    def last_price(self) -> float:
        return self._last_price

    @property
    def price_change_24h_pct(self) -> float:
        if self._price_24h_ago == 0:
            return 0.0
        return (self._last_price - self._price_24h_ago) / self._price_24h_ago * 100

    @property
    def regime_state(self) -> Optional[RegimeState]:
        return self._regime_state

    @property
    def status(self) -> SystemStatus:
        return self._status

    @property
    def perf_bnh(self):
        return self._perf_bnh

    @property
    def perf_regime(self):
        return self._perf_regime

    def get_bars_1s(self, n: int = 300) -> List[Bar1s]:
        """
        Returns up to n complete 1s bars PLUS the current partial bar appended.
        This means the live chart always shows what's happening RIGHT NOW,
        not just bars that already closed.
        """
        with self._lock:
            bars    = list(self._bars_1s)
            partial = self._partial_bar

        if n:
            bars = bars[-n:]

        # Append partial bar (in-progress current second) if available
        if partial is not None:
            # Don't duplicate if same timestamp as last complete bar
            if not bars or bars[-1].ts_ms != partial.ts_ms:
                bars = bars + [partial]

        return bars

    def get_bars_daily(self, n: int = None) -> List[BarDaily]:
        with self._lock:
            bars = list(self._bars_daily)
        if n is None:
            return bars
        return bars[-n:] if len(bars) > n else bars

    def get_regime_history(self) -> List[Tuple]:
        with self._lock:
            return list(self._regime_history)

    def get_daily_df(self) -> "pd.DataFrame":
        import pandas as pd
        with self._lock:
            bars = list(self._bars_daily)
        if not bars:
            return pd.DataFrame()
        rows = [{
            "open":           b.open,
            "high":           b.high,
            "low":            b.low,
            "close":          b.close,
            "volume":         b.volume,
            "quote_volume":   b.quote_volume,
            "taker_buy_base": b.taker_buy_base,
            "num_trades":     b.num_trades,
        } for b in bars]
        idx = pd.to_datetime([b.ts_ms for b in bars], unit="ms", utc=True)
        return pd.DataFrame(rows, index=idx)