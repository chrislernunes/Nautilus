"""
python/dashboard/app.py

Nautilus Crypto - Plotly/Dash live dashboard.

The public contract stays intentionally small:

    NautilusApp(store).run()

main.py owns the market-data and regime threads; this module only renders the
thread-safe DataStore state and Binance options volatility data.
"""

from __future__ import annotations

import json as _json
import math
import os
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from python.core.data_store import Bar1s, BarDaily, DataStore


BG = "#000000"
PANEL = "#000000"
PANEL_2 = "#06060a"
BORDER = "#1e2028"
GRID = "#202020"
TEXT = "#f2f5f8"
MUTED = "#9ca3af"
CYAN = "#ffffff"
GREEN = "#2bd67b"
ORANGE = "#f59e0b"
RED = "#ff4d6d"
BLUE = "#b8b8b8"

LIVE_PRICE_WINDOW_BARS = 90
LIVE_PRICE_RANGE_PAD_FRACTION = 0.20
LIVE_PRICE_MIN_PAD_PCT = 0.00002

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASH_ASSETS_DIR = PROJECT_ROOT / "assets"

CRYPTO_ASSETS = ("BTC", "ETH", "SOL", "BNB", "XRP", "DOGE")
ADAPTIVE_EXPOSURE = {
    0: 1.00,
    1: 0.30,
    2: 0.00,
    3: 0.60,
}
ADAPTIVE_CANDIDATES = {
    "Defensive Regime": {0: 0.85, 1: 0.00, 2: 0.00, 3: 0.40},
    "Balanced Regime":  {0: 1.00, 1: 0.08, 2: 0.00, 3: 0.55},
    "Return Regime":    {0: 1.10, 1: 0.12, 2: 0.00, 3: 0.75},
    "Trend Guard Regime": {0: 1.05, 1: 0.04, 2: 0.00, 3: 0.65},
}

REGIME_LABEL = {
    0: "BULL TREND",
    1: "HIGH VOL",
    2: "BEAR TREND",
    3: "LOW VOL CHOP",
}
REGIME_SHORT = {
    0: "BULL",
    1: "HIGH VOL",
    2: "BEAR",
    3: "CHOP",
}
REGIME_COLOR = {
    0: GREEN,
    1: ORANGE,
    2: RED,
    3: "#94a3b8",
}
REGIME_FILL = {
    0: "rgba(34, 217, 122, 0.18)",
    1: "rgba(245, 158, 11, 0.20)",
    2: "rgba(244, 63, 94, 0.18)",
    3: "rgba(148, 163, 184, 0.14)",
}

TEXT_ANNO = "#6b7280"  # muted annotation label colour

PROB_KEYS = [
    "BULL_TREND",
    "HIGH_VOL",
    "BEAR_TREND",
    "LOW_VOL_CHOP",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _ts_to_utc(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)


def _missing_regime(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "null"}:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _bar_rows(bars: Iterable[Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for bar in bars:
        rows.append(
            {
                "time": _ts_to_utc(int(bar.ts_ms)),
                "open": _safe_float(getattr(bar, "open", 0.0)),
                "high": _safe_float(getattr(bar, "high", 0.0)),
                "low": _safe_float(getattr(bar, "low", 0.0)),
                "close": _safe_float(getattr(bar, "close", 0.0)),
                "volume": _safe_float(getattr(bar, "volume", 0.0)),
                "buy_volume": _safe_float(getattr(bar, "buy_volume", 0.0)),
                "sell_volume": _safe_float(getattr(bar, "sell_volume", 0.0)),
                "vwap": _safe_float(getattr(bar, "vwap", 0.0)),
                "num_trades": int(getattr(bar, "num_trades", 0) or 0),
                "regime": getattr(bar, "regime", None),
            }
        )
    return rows


def _rows_with_regime_context(
    rows: List[Dict[str, Any]],
    store: Any,
    fill_current: bool = False,
) -> List[Dict[str, Any]]:
    """
    Fill missing regime labels from the regime_history log.
    fill_current=True allows using the live regime state as a last-resort
    fallback for the *most recent* bar only (live charts). For daily charts
    this must be False — otherwise the entire history gets painted with the
    current regime, making the shading useless.
    """
    if not rows:
        return rows
    if all(not _missing_regime(row.get("regime")) for row in rows):
        return rows

    history = []
    try:
        history = store.get_regime_history()
    except Exception:
        history = []

    if history:
        by_day: Dict[Any, int] = {}
        for ts, regime_id in history:
            try:
                by_day[pd.Timestamp(ts).date()] = int(regime_id)
            except Exception:
                continue
        if by_day:
            for row in rows:
                if not _missing_regime(row.get("regime")):
                    continue
                day = pd.Timestamp(row["time"]).date()
                if day in by_day:
                    row["regime"] = by_day[day]

    if fill_current:
        # Only fill the last (most recent) bar with the live regime
        regime_state = getattr(store, "regime_state", None)
        if regime_state is not None and rows and _missing_regime(rows[-1].get("regime")):
            try:
                rows[-1] = dict(rows[-1])
                rows[-1]["regime"] = int(regime_state.regime.value)
            except Exception:
                pass

    return rows


def _base_figure(title: str = "") -> go.Figure:
    fig = go.Figure()
    return _style_figure(fig, title)


def _style_figure(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL_2,
        font={"family": "Inter, monospace", "color": TEXT},
        title={"text": title, "x": 0.02, "xanchor": "left", "font": {"size": 14}},
        margin={"l": 48, "r": 18, "t": 58 if title else 24, "b": 34},
        height=height,
        hovermode="x unified",
        uirevision="nautilus",
        transition={"duration": 0},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": 1.0,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 9},
            "itemsizing": "constant",
            "itemwidth": 30,
            "bgcolor": "rgba(0,0,0,0)",
        },
        xaxis={"gridcolor": GRID, "zerolinecolor": GRID, "linecolor": BORDER, "tickfont": {"size": 9}},
        yaxis={"gridcolor": GRID, "zerolinecolor": GRID, "linecolor": BORDER, "tickfont": {"size": 9}},
    )
    return fig


def _empty_figure(message: str, height: int = 420, sub: str = "") -> go.Figure:
    """Improved empty state — always renders legible text with a subtle background."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL_2,
        font={"family": "Inter, monospace", "color": TEXT},
        height=height,
        margin={"l": 12, "r": 12, "t": 12, "b": 12},
        xaxis={"visible": False, "showgrid": False},
        yaxis={"visible": False, "showgrid": False},
    )
    fig.add_annotation(
        text=message, x=0.5, y=0.54, xref="paper", yref="paper",
        showarrow=False, font={"color": "#4b5563", "size": 11}, align="center",
    )
    if sub:
        fig.add_annotation(
            text=sub, x=0.5, y=0.43, xref="paper", yref="paper",
            showarrow=False, font={"color": "#374151", "size": 9}, align="center",
        )
    fig.add_annotation(
        text="● ● ●", x=0.5, y=0.33, xref="paper", yref="paper",
        showarrow=False, font={"color": "#1f2937", "size": 9}, align="center",
    )
    return fig


def _axis_ref(axis: str, row: int) -> str:
    return axis if row == 1 else f"{axis}{row}"


def _add_regime_rect(
    fig: go.Figure,
    x0: Any,
    x1: Any,
    regime_id: int,
    row: int,
) -> None:
    fig.add_shape(
        type="rect",
        xref=_axis_ref("x", row),
        yref=f"{_axis_ref('y', row)} domain",
        x0=x0,
        x1=x1,
        y0=0,
        y1=1,
        fillcolor=REGIME_FILL[regime_id],
        line={"color": REGIME_COLOR[regime_id], "width": 0},
        opacity=0.52,
        layer="below",
    )


def _add_regime_bands(fig: go.Figure, rows: List[Dict[str, Any]], row: int = 1) -> None:
    if len(rows) < 2:
        return

    last_width = rows[-1]["time"] - rows[-2]["time"]

    start_idx = 0
    current = rows[0].get("regime")
    for idx in range(1, len(rows) + 1):
        next_regime = rows[idx].get("regime") if idx < len(rows) else None
        if idx < len(rows) and next_regime == current:
            continue
        if not _missing_regime(current):
            try:
                regime_id = int(current) % 4
            except (TypeError, ValueError):
                regime_id = -1
            if regime_id in REGIME_FILL:
                x0 = rows[start_idx]["time"]
                x1 = rows[idx]["time"] if idx < len(rows) else rows[-1]["time"] + last_width
                _add_regime_rect(fig, x0, x1, regime_id, row)
        if idx < len(rows):
            start_idx = idx
            current = next_regime


def _smooth_regime_rows(rows: List[Dict[str, Any]], min_run: int = 10) -> List[Dict[str, Any]]:
    if len(rows) < min_run * 2:
        return rows

    out = [dict(row) for row in rows]
    runs: List[Tuple[int, int, Any]] = []
    start = 0
    current = out[0].get("regime")
    for idx in range(1, len(out) + 1):
        next_regime = out[idx].get("regime") if idx < len(out) else None
        if idx < len(out) and next_regime == current:
            continue
        runs.append((start, idx, current))
        if idx < len(out):
            start = idx
            current = next_regime

    for run_idx, (start, end, regime) in enumerate(runs):
        if _missing_regime(regime) or (end - start) >= min_run:
            continue
        replacement = None
        if run_idx > 0 and not _missing_regime(runs[run_idx - 1][2]):
            replacement = runs[run_idx - 1][2]
        elif run_idx + 1 < len(runs) and not _missing_regime(runs[run_idx + 1][2]):
            replacement = runs[run_idx + 1][2]
        if replacement is None:
            continue
        for idx in range(start, end):
            out[idx]["regime"] = replacement
    return out


def _add_current_regime_band(
    fig: go.Figure,
    rows: List[Dict[str, Any]],
    store: Any,
    target_rows: Tuple[int, ...] = (1,),
) -> None:
    if len(rows) < 2:
        return
    regime_state = getattr(store, "regime_state", None)
    if regime_state is None:
        return
    try:
        regime_id = int(regime_state.regime.value) % 4
    except Exception:
        return
    last_width = rows[-1]["time"] - rows[-2]["time"]
    for row in target_rows:
        _add_regime_rect(fig, rows[0]["time"], rows[-1]["time"] + last_width, regime_id, row)


def _add_regime_legend(fig: go.Figure) -> None:
    for regime_id in range(4):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=REGIME_SHORT[regime_id],
                marker={"size": 6, "color": REGIME_COLOR[regime_id], "symbol": "square"},
                hoverinfo="skip",
                showlegend=True,
            )
        )


class IVSurface:
    """
    Poll Binance option mark IVs in a background thread.

    Binance endpoints:
        GET https://eapi.binance.com/eapi/v1/mark
        GET https://vapi.binance.com/vapi/v1/mark

    Symbols are expected as UNDERLYING-YYMMDD-STRIKE-C/P.
    """

    ENDPOINTS = (
        "https://eapi.binance.com/eapi/v1/mark",
        "https://vapi.binance.com/vapi/v1/mark",
    )
    SPOT_ENDPOINT = "https://api.binance.com/api/v3/ticker/price"
    POLL_MS = 5000
    ASSETS = ("BTC", "ETH", "SOL", "BNB", "XRP")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, Any]] = {}
        self._spots: Dict[str, float] = {}
        self._ts_ms = 0
        self._error = ""
        self._blocked_until: Dict[str, float] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="iv-surface",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float], int, str]:
        with self._lock:
            return dict(self._data), dict(self._spots), self._ts_ms, self._error

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            last_error = ""
            try:
                parsed: Dict[str, Dict[str, Any]] = {}
                for endpoint in self.ENDPOINTS:
                    if self._blocked_until.get(endpoint, 0.0) > time.monotonic():
                        continue
                    try:
                        req = urllib.request.Request(
                            endpoint,
                            headers={"User-Agent": "nautilus-btc/plotly-dashboard"},
                        )
                        with urllib.request.urlopen(req, timeout=4) as resp:
                            raw = _json.loads(resp.read().decode("utf-8"))
                    except urllib.error.HTTPError as exc:
                        if exc.code in {418, 429}:
                            retry_after = _safe_float(exc.headers.get("Retry-After"), 60.0)
                            self._blocked_until[endpoint] = time.monotonic() + max(30.0, min(retry_after, 900.0))
                        last_error = f"{urllib.parse.urlparse(endpoint).netloc}: HTTP {exc.code}"
                        continue
                    except Exception as exc:
                        last_error = f"{urllib.parse.urlparse(endpoint).netloc}: {str(exc)[:90]}"
                        continue

                    parsed = self._parse_mark_payload(raw)
                    if parsed:
                        last_error = ""
                        break
                    last_error = f"{urllib.parse.urlparse(endpoint).netloc}: no option IVs"

                if not parsed:
                    with self._lock:
                        self._error = last_error or "No Binance option IV data returned"
                    elapsed = time.monotonic() - started
                    self._stop.wait(max(0.25, self.POLL_MS / 1000.0 - elapsed))
                    continue

                spots = self._fetch_spots({vals["underlying"] for vals in parsed.values()})

                with self._lock:
                    self._data = parsed
                    if spots:
                        self._spots = spots
                    self._ts_ms = int(time.time() * 1000)
                    self._error = ""
            except Exception as exc:
                with self._lock:
                    self._error = str(exc)[:120]

            elapsed = time.monotonic() - started
            sleep_s = max(0.05, self.POLL_MS / 1000.0 - elapsed)
            self._stop.wait(sleep_s)

    def _parse_mark_payload(self, raw: Any) -> Dict[str, Dict[str, Any]]:
        items = raw.get("data", []) if isinstance(raw, dict) else raw
        if not isinstance(items, list):
            return {}

        parsed: Dict[str, Dict[str, Any]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            sym = str(item.get("symbol", ""))
            parts = sym.split("-")
            if len(parts) != 4:
                continue
            asset, _, _, _ = parts
            if asset not in self.ASSETS:
                continue
            iv_val = _safe_float(item.get("markIV"))
            if iv_val <= 0:
                bid_iv = _safe_float(item.get("bidIV"))
                ask_iv = _safe_float(item.get("askIV"))
                ivs = [iv for iv in (bid_iv, ask_iv) if iv > 0]
                iv_val = sum(ivs) / len(ivs) if ivs else 0.0
            if iv_val <= 0:
                continue
            parsed[sym] = {
                "underlying": asset,
                "iv": iv_val,
                "mark_price": _safe_float(item.get("markPrice")),
            }
        return parsed

    def _fetch_spots(self, assets: Iterable[str]) -> Dict[str, float]:
        pairs = [f"{asset}USDT" for asset in assets if asset in self.ASSETS]
        if not pairs:
            return {}
        try:
            payload = _json.dumps(pairs)
            url = f"{self.SPOT_ENDPOINT}?symbols={urllib.parse.quote(payload)}"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "nautilus-btc/plotly-dashboard"},
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                raw = _json.loads(resp.read().decode("utf-8"))
            out: Dict[str, float] = {}
            for item in raw:
                sym = str(item.get("symbol", ""))
                if sym.endswith("USDT"):
                    out[sym[:-4]] = _safe_float(item.get("price"))
            return {k: v for k, v in out.items() if v > 0}
        except Exception:
            return {}


_iv_surface = IVSurface()


class TopTraderPositions:
    """Poll Binance USD-M futures top trader long/short position ratio."""

    ENDPOINT = "https://fapi.binance.com/futures/data/topLongShortPositionRatio"
    POLL_MS = 30000

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: List[Dict[str, Any]] = []
        self._ts_ms = 0
        self._error = ""
        self._blocked_until = 0.0
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="top-trader-positions",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self) -> Tuple[List[Dict[str, Any]], int, str]:
        with self._lock:
            return [dict(row) for row in self._data], self._ts_ms, self._error

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                if self._blocked_until > time.monotonic():
                    self._stop.wait(max(1.0, self._blocked_until - time.monotonic()))
                    continue
                query = urllib.parse.urlencode(
                    {
                        "symbol": "BTCUSDT",
                        "period": "5m",
                        "limit": "96",
                    }
                )
                req = urllib.request.Request(
                    f"{self.ENDPOINT}?{query}",
                    headers={"User-Agent": "nautilus-btc/plotly-dashboard"},
                )
                with urllib.request.urlopen(req, timeout=4) as resp:
                    raw = _json.loads(resp.read().decode("utf-8"))

                parsed = self._parse_payload(raw)
                if parsed:
                    with self._lock:
                        self._data = parsed
                        self._ts_ms = int(time.time() * 1000)
                        self._error = ""
                else:
                    with self._lock:
                        self._error = "No Binance top trader position rows"
            except urllib.error.HTTPError as exc:
                if exc.code in {418, 429}:
                    retry_after = _safe_float(exc.headers.get("Retry-After"), 60.0)
                    self._blocked_until = time.monotonic() + max(30.0, min(retry_after, 900.0))
                with self._lock:
                    self._error = f"HTTP {exc.code}"
            except Exception as exc:
                with self._lock:
                    self._error = str(exc)[:120]

            elapsed = time.monotonic() - started
            self._stop.wait(max(0.25, self.POLL_MS / 1000.0 - elapsed))

    def _parse_payload(self, raw: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        rows: List[Dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            ts_ms = int(_safe_float(item.get("timestamp")))
            if ts_ms <= 0:
                continue
            long_val = _safe_float(item.get("longAccount", item.get("longPosition")))
            short_val = _safe_float(item.get("shortAccount", item.get("shortPosition")))
            ratio = _safe_float(item.get("longShortRatio"))
            if long_val <= 0 and short_val <= 0:
                continue
            if ratio <= 0 and short_val > 0:
                ratio = long_val / short_val
            rows.append(
                {
                    "time": _ts_to_utc(ts_ms),
                    "long_pct": long_val * 100.0,
                    "short_pct": short_val * 100.0,
                    "ratio": ratio,
                }
            )
        rows.sort(key=lambda row: row["time"])
        return rows


_top_trader_positions = TopTraderPositions()


class PerpPremiumIndex:
    """
    Poll Binance USD-M futures premium index for BTCUSDT perpetual.
    Fetches funding rate, mark price, index price, and premium.
    Endpoint: GET https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT
    Also fetches funding rate history: GET https://fapi.binance.com/fapi/v1/fundingRate
    """

    PREMIUM_ENDPOINT = "https://fapi.binance.com/fapi/v1/premiumIndex"
    FUNDING_ENDPOINT = "https://fapi.binance.com/fapi/v1/fundingRate"
    POLL_MS = 10000

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: Dict[str, Any] = {}
        self._history: List[Dict[str, Any]] = []
        self._ts_ms = 0
        self._error = ""
        self._blocked_until = 0.0
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="perp-premium",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self) -> Tuple[Dict[str, Any], List[Dict[str, Any]], int, str]:
        with self._lock:
            return dict(self._current), list(self._history), self._ts_ms, self._error

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                if self._blocked_until > time.monotonic():
                    self._stop.wait(max(1.0, self._blocked_until - time.monotonic()))
                    continue
                current = self._fetch_premium_index()
                history = self._fetch_funding_history()
                if current or history:
                    with self._lock:
                        if current:
                            self._current = current
                        if history:
                            self._history = history
                        self._ts_ms = int(time.time() * 1000)
                        self._error = ""
                else:
                    with self._lock:
                        self._error = "No premium index data"
            except urllib.error.HTTPError as exc:
                if exc.code in {418, 429}:
                    retry_after = _safe_float(exc.headers.get("Retry-After"), 60.0)
                    self._blocked_until = time.monotonic() + max(30.0, min(retry_after, 900.0))
                with self._lock:
                    self._error = f"HTTP {exc.code}"
            except Exception as exc:
                with self._lock:
                    self._error = str(exc)[:120]
            elapsed = time.monotonic() - started
            self._stop.wait(max(0.25, self.POLL_MS / 1000.0 - elapsed))

    def _fetch_premium_index(self) -> Dict[str, Any]:
        url = f"{self.PREMIUM_ENDPOINT}?symbol=BTCUSDT"
        req = urllib.request.Request(url, headers={"User-Agent": "nautilus-btc/plotly-dashboard"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            raw = _json.loads(resp.read().decode("utf-8"))
        if not isinstance(raw, dict):
            return {}
        return {
            "mark_price": _safe_float(raw.get("markPrice")),
            "index_price": _safe_float(raw.get("indexPrice")),
            "last_funding_rate": _safe_float(raw.get("lastFundingRate")) * 100.0,
            "next_funding_time": int(_safe_float(raw.get("nextFundingTime"))),
            "interest_rate": _safe_float(raw.get("interestRate")) * 100.0,
            "premium": (_safe_float(raw.get("markPrice")) - _safe_float(raw.get("indexPrice"))) /
                       max(_safe_float(raw.get("indexPrice")), 1.0) * 100.0,
        }

    def _fetch_funding_history(self) -> List[Dict[str, Any]]:
        query = urllib.parse.urlencode({"symbol": "BTCUSDT", "limit": "100"})
        url = f"{self.FUNDING_ENDPOINT}?{query}"
        req = urllib.request.Request(url, headers={"User-Agent": "nautilus-btc/plotly-dashboard"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            raw = _json.loads(resp.read().decode("utf-8"))
        if not isinstance(raw, list):
            return []
        rows: List[Dict[str, Any]] = []
        for item in raw:
            ts_ms = int(_safe_float(item.get("fundingTime")))
            rate = _safe_float(item.get("fundingRate")) * 100.0
            if ts_ms > 0:
                rows.append({"time": _ts_to_utc(ts_ms), "rate": rate})
        rows.sort(key=lambda r: r["time"])
        return rows


_perp_premium = PerpPremiumIndex()


class PerpMsIndex:
    """
    Real-time BTCUSDT perpetual mark/index price sampled at ~500ms
    for the high-frequency premium chart.
    Polls fapi.binance.com/fapi/v1/premiumIndex with no rate throttle.
    Stores a rolling 10-minute window of (timestamp_ms, premium_bps) samples.
    """

    ENDPOINT = "https://fapi.binance.com/fapi/v1/premiumIndex"
    POLL_MS = 250          # target ~4 samples/sec
    WINDOW_S = 600         # 10-minute rolling window

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Each entry: (ts_ms: int, mark: float, index: float, premium_bps: float)
        self._samples: List[Tuple[int, float, float, float]] = []
        self._error = ""
        self._blocked_until = 0.0
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, name="perp-ms-index", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self) -> Tuple[List[Tuple[int, float, float, float]], str]:
        with self._lock:
            return list(self._samples), self._error

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                if self._blocked_until > time.monotonic():
                    self._stop.wait(max(0.1, self._blocked_until - time.monotonic()))
                    continue
                req = urllib.request.Request(
                    f"{self.ENDPOINT}?symbol=BTCUSDT",
                    headers={"User-Agent": "nautilus-btc/plotly-dashboard"},
                )
                with urllib.request.urlopen(req, timeout=2) as resp:
                    raw = _json.loads(resp.read().decode("utf-8"))
                if isinstance(raw, dict):
                    mark = _safe_float(raw.get("markPrice"))
                    index_ = _safe_float(raw.get("indexPrice"))
                    if mark > 0 and index_ > 0:
                        premium_bps = (mark - index_) / index_ * 10000.0
                        ts_ms = int(time.time() * 1000)
                        cutoff = ts_ms - self.WINDOW_S * 1000
                        with self._lock:
                            self._samples.append((ts_ms, mark, index_, premium_bps))
                            # Trim to window
                            self._samples = [s for s in self._samples if s[0] >= cutoff]
                            self._error = ""
            except urllib.error.HTTPError as exc:
                if exc.code in {418, 429}:
                    self._blocked_until = time.monotonic() + 60.0
                with self._lock:
                    self._error = f"HTTP {exc.code}"
            except Exception as exc:
                with self._lock:
                    self._error = str(exc)[:80]
            elapsed = time.monotonic() - started
            self._stop.wait(max(0.05, self.POLL_MS / 1000.0 - elapsed))


_perp_ms_index = PerpMsIndex()


class SpotTicker:
    """Poll compact 24h spot tickers for the top-bar crypto strip."""

    SPOT_ENDPOINT = "https://api.binance.com/api/v3/ticker/24hr"
    FUTURES_ENDPOINT = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    POLL_MS = 3000

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, float]] = {}
        self._ts_ms = 0
        self._error = ""
        self._blocked_until = 0.0
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="spot-ticker",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self) -> Tuple[Dict[str, Dict[str, float]], int, str]:
        with self._lock:
            return {asset: dict(vals) for asset, vals in self._data.items()}, self._ts_ms, self._error

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                if self._blocked_until > time.monotonic():
                    self._stop.wait(max(1.0, self._blocked_until - time.monotonic()))
                    continue
                parsed: Dict[str, Dict[str, float]] = {}
                try:
                    parsed = self._fetch_spot()
                except Exception:
                    parsed = {}
                if not parsed:
                    parsed = self._fetch_futures()
                if parsed:
                    with self._lock:
                        self._data = parsed
                        self._ts_ms = int(time.time() * 1000)
                        self._error = ""
            except urllib.error.HTTPError as exc:
                if exc.code in {418, 429}:
                    retry_after = _safe_float(exc.headers.get("Retry-After"), 60.0)
                    self._blocked_until = time.monotonic() + max(30.0, min(retry_after, 900.0))
                with self._lock:
                    self._error = f"HTTP {exc.code}"
            except Exception as exc:
                with self._lock:
                    self._error = str(exc)[:120]

            elapsed = time.monotonic() - started
            self._stop.wait(max(0.25, self.POLL_MS / 1000.0 - elapsed))

    def _fetch_spot(self) -> Dict[str, Dict[str, float]]:
        """Try batch spot fetch first; fall back to individual requests."""
        symbols = [f"{asset}USDT" for asset in CRYPTO_ASSETS]
        try:
            # Batch request (may fail in some regions)
            query = urllib.parse.urlencode({"symbols": _json.dumps(symbols)})
            req = urllib.request.Request(
                f"{self.SPOT_ENDPOINT}?{query}",
                headers={"User-Agent": "nautilus-btc/plotly-dashboard"},
            )
            with urllib.request.urlopen(req, timeout=4) as resp:
                raw = _json.loads(resp.read().decode("utf-8"))
            result = self._parse_tickers(raw)
            if result:
                return result
        except Exception:
            pass
        # Individual fallback
        result: Dict[str, Dict[str, float]] = {}
        for sym in symbols:
            try:
                req = urllib.request.Request(
                    f"https://api.binance.com/api/v3/ticker/24hr?symbol={sym}",
                    headers={"User-Agent": "nautilus-btc/plotly-dashboard"},
                )
                with urllib.request.urlopen(req, timeout=3) as resp:
                    item = _json.loads(resp.read().decode("utf-8"))
                if isinstance(item, dict):
                    price = _safe_float(item.get("lastPrice"))
                    change = _safe_float(item.get("priceChangePercent"))
                    asset = sym[:-4]
                    if price > 0:
                        result[asset] = {"price": price, "change": change}
            except Exception:
                continue
        return result

    def _fetch_futures(self) -> Dict[str, Dict[str, float]]:
        req = urllib.request.Request(
            self.FUTURES_ENDPOINT,
            headers={"User-Agent": "nautilus-btc/plotly-dashboard"},
        )
        with urllib.request.urlopen(req, timeout=4) as resp:
            raw = _json.loads(resp.read().decode("utf-8"))
        return self._parse_tickers(raw)

    def _parse_tickers(self, raw: Any) -> Dict[str, Dict[str, float]]:
        parsed: Dict[str, Dict[str, float]] = {}
        if not isinstance(raw, list):
            return parsed
        for item in raw:
            if not isinstance(item, dict):
                continue
            sym = str(item.get("symbol", ""))
            if not sym.endswith("USDT"):
                continue
            asset = sym[:-4]
            if asset not in CRYPTO_ASSETS:
                continue
            price = _safe_float(item.get("lastPrice"))
            change = _safe_float(item.get("priceChangePercent"))
            if price > 0:
                parsed[asset] = {"price": price, "change": change}
        return parsed


_spot_ticker = SpotTicker()


def _current_regime_id(store: Any) -> Optional[int]:
    regime_state = getattr(store, "regime_state", None)
    if regime_state is None:
        return None
    try:
        return int(regime_state.regime.value) % 4
    except Exception:
        return None


def _append_live_tick(rows: List[Dict[str, Any]], store: "DataStore") -> List[Dict[str, Any]]:
    price = _safe_float(store.last_price)
    if price <= 0:
        return rows

    now = datetime.now(timezone.utc)
    regime_id = _current_regime_id(store)
    if rows and pd.Timestamp(rows[-1]["time"]).to_pydatetime() >= now:
        out = list(rows)
        last = dict(out[-1])
        last["close"] = price
        last["high"] = max(_safe_float(last.get("high")), price)
        low = _safe_float(last.get("low"), price)
        last["low"] = min(low if low > 0 else price, price)
        last["vwap"] = price
        if regime_id is not None:
            last["regime"] = regime_id
        out[-1] = last
        return out

    base = dict(rows[-1]) if rows else {}
    prev_close = _safe_float(base.get("close"), price)
    tick = {
        "time": now,
        "open": prev_close if prev_close > 0 else price,
        "high": max(prev_close, price) if prev_close > 0 else price,
        "low": min(prev_close, price) if prev_close > 0 else price,
        "close": price,
        "volume": 0.0,
        "buy_volume": 0.0,
        "sell_volume": 0.0,
        "vwap": price,
        "num_trades": 0,
        "regime": regime_id,
    }
    return rows + [tick]


def _live_rows(store: "DataStore", n: int = 240) -> List[Dict[str, Any]]:
    # fill_current=True: the most recent incomplete bar gets the live regime label
    rows = _rows_with_regime_context(_bar_rows(store.get_bars_1s(n=n)), store, fill_current=True)
    return _append_live_tick(rows, store)


def _log_price_axis_range(prices: "pd.Series") -> Optional[List[float]]:
    px = pd.to_numeric(prices, errors="coerce")
    px = px[px > 0]
    if px.empty:
        return None

    px_min = float(px.min())
    px_max = float(px.max())
    px_span = px_max - px_min
    min_pad = px_max * LIVE_PRICE_MIN_PAD_PCT

    if px_span > 0:
        pad = max(px_span * LIVE_PRICE_RANGE_PAD_FRACTION, min_pad)
        y_lo = px_min - pad
        y_hi = px_max + pad
        if y_lo <= 0:
            y_lo = px_min / (1.0 + LIVE_PRICE_MIN_PAD_PCT)
    else:
        y_lo = px_min / (1.0 + LIVE_PRICE_MIN_PAD_PCT)
        y_hi = px_max * (1.0 + LIVE_PRICE_MIN_PAD_PCT)

    return [math.log10(y_lo), math.log10(y_hi)]


def _daily_figure(store: "DataStore") -> go.Figure:
    rows = _bar_rows(store.get_bars_daily())
    # fill_current=False: do NOT paint all history with today's live regime
    rows = _rows_with_regime_context(rows, store, fill_current=False)
    if len(rows) < 2:
        return _empty_figure("Daily BTCUSDT — Loading history...", height=470, sub="Fetching OHLCV from data store")

    df = pd.DataFrame(rows)
    df = df.tail(900)
    colors = [GREEN if c >= o else RED for c, o in zip(df["close"], df["open"])]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.72, 0.28],
    )
    # min_run=90: suppress flickers shorter than 90 days on daily chart
    daily_records = _smooth_regime_rows(df.to_dict("records"), min_run=90)
    _add_regime_bands(fig, daily_records, row=1)
    _add_regime_bands(fig, daily_records, row=2)
    # Only draw current-regime band if no history could be resolved at all
    if not fig.layout.shapes:
        _add_current_regime_band(fig, daily_records, store, target_rows=(1, 2))
    fig.add_trace(
        go.Candlestick(
            x=df["time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
            increasing_line_color=GREEN,
            decreasing_line_color=RED,
            increasing_fillcolor="rgba(43, 214, 123, 0.45)",
            decreasing_fillcolor="rgba(255, 77, 109, 0.45)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["close"].ewm(span=20, adjust=False).mean(),
            mode="lines",
            name="EMA 20",
            line={"color": CYAN, "width": 1.4},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["close"].ewm(span=50, adjust=False).mean(),
            mode="lines",
            name="EMA 50",
            line={"color": ORANGE, "width": 1.25},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=df["time"],
            y=df["volume"],
            name="Volume",
            marker={"color": colors, "opacity": 0.72},
        ),
        row=2,
        col=1,
    )
    _add_regime_legend(fig)

    last = df.iloc[-1]
    first = df.iloc[0]
    chg = ((last["close"] / first["close"]) - 1.0) * 100 if first["close"] else 0.0
    title = f"Daily BTCUSDT  ${last['close']:,.0f}  ({chg:+.1f}%)"
    _style_figure(fig, title=title, height=470)
    fig.update_xaxes(rangeslider_visible=False, gridcolor=GRID)
    fig.update_yaxes(title_text="Price", tickprefix="$", row=1, col=1, gridcolor=GRID)
    fig.update_yaxes(title_text="Vol", row=2, col=1, gridcolor=GRID)
    return fig


def _live_figure(store: "DataStore", rows: Optional[List[Dict[str, Any]]] = None) -> go.Figure:
    if rows is None:
        rows = _live_rows(store, n=LIVE_PRICE_WINDOW_BARS)
    else:
        rows = rows[-LIVE_PRICE_WINDOW_BARS:]
    if len(rows) < 2:
        return _empty_figure("Live BTCUSDT 1s bars — Connecting...", height=470, sub="Waiting for WebSocket feed")

    df = pd.DataFrame(rows)
    volume_colors = [GREEN if b >= s else RED for b, s in zip(df["buy_volume"], df["sell_volume"])]
    imbalance = df["buy_volume"] - df["sell_volume"]
    chg_1m = 0.0
    if len(df) >= 60 and df["close"].iloc[-60]:
        chg_1m = ((df["close"].iloc[-1] / df["close"].iloc[-60]) - 1.0) * 100

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.70, 0.30],
    )
    live_records = df.to_dict("records")
    _add_regime_bands(fig, live_records, row=1)
    _add_regime_bands(fig, live_records, row=2)
    if not fig.layout.shapes:
        _add_current_regime_band(fig, live_records, store, target_rows=(1, 2))
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["close"],
            mode="lines",
            name="Close",
            line={"color": CYAN, "width": 2.2},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[df["time"].iloc[-1]],
            y=[df["close"].iloc[-1]],
            mode="markers",
            name="Now",
            marker={"color": CYAN, "size": 7, "line": {"color": "#000000", "width": 1}},
            hovertemplate="Now<br>%{x}<br>$%{y:,.2f}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    if (df["vwap"] > 0).any():
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["vwap"],
                mode="lines",
                name="VWAP",
                line={"color": ORANGE, "width": 1.2, "dash": "dot"},
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Bar(
            x=df["time"],
            y=df["volume"],
            name="Volume",
            marker={"color": volume_colors, "opacity": 0.65},
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=imbalance,
            mode="lines",
            name="Buy-Sell",
            line={"color": BLUE, "width": 1.1},
            opacity=0.85,
        ),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=float(df["close"].iloc[-1]),
        line={"color": MUTED, "width": 1, "dash": "dot"},
        row=1,
        col=1,
    )

    last_ts = pd.Timestamp(df["time"].iloc[-1])
    title = f"Live BTCUSDT  1s bars  {last_ts.strftime('%H:%M:%S')} UTC  1m {chg_1m:+.3f}%"
    _style_figure(fig, title=title, height=470)
    last_ts_s = int(pd.Timestamp(df["time"].iloc[-1]).timestamp())
    last_px = int(_safe_float(df["close"].iloc[-1]) * 100)
    fig.update_layout(uirevision=f"live-btc", datarevision=f"{last_ts_s}-{last_px}")

    # ── Y-axis: explicit range prevents line escaping the chart on each update.
    # A tight relative pad keeps one-second BTC moves visible instead of
    # burying them in a wide absolute-price band.
    _price_range = _log_price_axis_range(df["close"])

    fig.update_yaxes(
        title_text="Price ($, log)",
        tickprefix="$",
        tickformat=",.2f",
        exponentformat="none",
        type="log",
        range=_price_range,
        autorange=_price_range is None,
        row=1,
        col=1,
        gridcolor=GRID,
    )
    fig.update_yaxes(title_text="Vol / Delta", autorange=True, row=2, col=1, gridcolor=GRID)

    # ── X-axis: rolling 90-second window anchored to the latest tick.
    # Without an explicit range the x-axis rescales from the first bar on
    # every callback, compressing the chart horizontally as new data arrives.
    _x_latest = df["time"].iloc[-1]
    _x_oldest = df["time"].iloc[0]
    _x_end    = pd.Timestamp(_x_latest) + pd.Timedelta(seconds=1)
    _x_start  = pd.Timestamp(_x_oldest) - pd.Timedelta(seconds=1)
    fig.update_xaxes(range=[_x_start, _x_end], autorange=False, gridcolor=GRID)
    return fig


def _live_distribution_figure(
    store: "DataStore",
    rows: Optional[List[Dict[str, Any]]] = None,
) -> go.Figure:
    if rows is None:
        rows = _live_rows(store, n=240)
    if len(rows) < 8:
        return _empty_figure("1s Return Distribution — Loading...", height=470, sub="Need 8+ 1-second bars")

    df = pd.DataFrame(rows)
    df = df[df["close"] > 0].copy()
    if len(df) < 8:
        return _empty_figure("1s Return Distribution — Loading...", height=470, sub="Need 8+ 1-second bars")

    df["ret"] = (df["close"].apply(math.log).diff() * 100.0)
    df = df.dropna(subset=["ret"])
    if df.empty:
        return _empty_figure("No live return observations yet", height=470)

    fig = go.Figure()
    has_regime = False
    for regime_id in range(4):
        values = [
            float(value)
            for value in df.loc[df["regime"].apply(lambda val: _same_regime(val, regime_id)), "ret"]
        ]
        if not values:
            continue
        has_regime = True
        fig.add_trace(
            go.Histogram(
                x=values,
                nbinsx=34,
                name=REGIME_SHORT[regime_id],
                marker={"color": REGIME_COLOR[regime_id]},
                opacity=0.72,
                hovertemplate="%{x:.4f}% 1s return<br>%{y} bars<extra></extra>",
            )
        )

    if not has_regime:
        fig.add_trace(
            go.Histogram(
                x=df["ret"],
                nbinsx=34,
                name="1s returns",
                marker={"color": CYAN},
                opacity=0.72,
                hovertemplate="%{x:.4f}% 1s return<br>%{y} bars<extra></extra>",
            )
        )

    values = [float(x) for x in df["ret"]]
    latest = values[-1]
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / max(1, len(values) - 1)
    stdev = math.sqrt(variance)
    fig.add_vline(x=mean, line={"color": CYAN, "width": 2, "dash": "dot"})
    fig.add_vline(x=latest, line={"color": RED, "width": 2})

    _style_figure(
        fig,
        title=f"1s Return Distribution  latest {latest:+.4f}%",
        height=470,
    )
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text="1s log return", ticksuffix="%", gridcolor=GRID)
    fig.update_yaxes(title_text=f"Bars / sigma {stdev:.4f}%", gridcolor=GRID)
    return fig


def _probability_figure(store: "DataStore") -> go.Figure:
    regime_state = store.regime_state
    if regime_state is None:
        return _empty_figure("Regime Probabilities — HMM initializing...", height=320)

    values = [float(regime_state.probabilities.get(key, 0.0)) for key in PROB_KEYS]
    labels = [REGIME_SHORT[i] for i in range(4)]
    fig = go.Figure(
        go.Bar(
            x=[v * 100 for v in values],
            y=labels,
            orientation="h",
            marker={"color": [REGIME_COLOR[i] for i in range(4)]},
            text=[f"{v:.0%}" for v in values],
            textposition="auto",
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )
    _style_figure(fig, title="Regime Probabilities", height=320)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(range=[0, 100], ticksuffix="%", gridcolor=GRID)
    fig.update_yaxes(autorange="reversed", gridcolor=GRID)
    return fig


def _returns_figure(store: "DataStore") -> go.Figure:
    rows = _bar_rows(store.get_bars_daily())
    if len(rows) < 25:
        return _empty_figure("Daily Return Distribution — Loading...", height=330, sub="Need 25+ daily bars")

    grouped: Dict[int, List[float]] = defaultdict(list)
    all_rets: List[float] = []
    for prev, cur in zip(rows[:-1], rows[1:]):
        if prev["close"] <= 0 or cur["close"] <= 0:
            continue
        ret = math.log(cur["close"] / prev["close"]) * 100
        all_rets.append(ret)
        regime = cur.get("regime")
        try:
            key = int(regime) % 4 if regime is not None else 4
        except (TypeError, ValueError):
            key = 4
        grouped[key].append(ret)

    if not all_rets:
        return _empty_figure("No return observations — check daily bar data", height=330)

    fig = go.Figure()
    for regime_id in range(4):
        values = grouped.get(regime_id, [])
        if not values:
            continue
        fig.add_trace(
            go.Histogram(
                x=values,
                nbinsx=44,
                name=REGIME_SHORT[regime_id],
                marker={"color": REGIME_COLOR[regime_id]},
                opacity=0.72,
                hovertemplate="%{x:.2f}% return<br>%{y} days<extra></extra>",
            )
        )
    if grouped.get(4):
        fig.add_trace(
            go.Histogram(
                x=grouped[4],
                nbinsx=44,
                name="Unlabeled",
                marker={"color": MUTED},
                opacity=0.55,
            )
        )

    mean = sum(all_rets) / len(all_rets)
    variance = sum((x - mean) ** 2 for x in all_rets) / max(1, len(all_rets) - 1)
    stdev = math.sqrt(variance)
    fig.add_vline(x=mean, line={"color": CYAN, "width": 2, "dash": "dot"})
    _style_figure(
        fig,
        title=f"Daily Return Distribution  μ {mean:+.2f}%  σ {stdev:.2f}%",
        height=330,
    )
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text="Daily log return", ticksuffix="%", gridcolor=GRID)
    fig.update_yaxes(title_text="Days", gridcolor=GRID)
    return fig


def _parse_iv_points(
    data: Dict[str, Dict[str, Any]],
    assets: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    asset_filter = set(assets) if assets else None
    points: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc).date()
    for sym, vals in data.items():
        parts = sym.split("-")
        if len(parts) != 4:
            continue
        asset, expiry_s, strike_s, cp = parts
        if asset_filter and asset not in asset_filter:
            continue
        try:
            expiry = datetime.strptime(expiry_s, "%y%m%d").replace(tzinfo=timezone.utc)
            strike = int(strike_s)
        except ValueError:
            continue
        iv = _safe_float(vals.get("iv")) * 100.0
        if iv <= 0:
            continue
        dte = max(0, (expiry.date() - now).days)
        points.append(
            {
                "underlying": asset,
                "expiry_key": expiry_s,
                "expiry_label": expiry.strftime("%d %b %y"),
                "dte": dte,
                "strike": strike,
                "cp": cp,
                "iv": iv,
            }
        )
    return points


def _nearest_iv(points: List[Dict[str, Any]], target_strike: float) -> Optional[float]:
    if not points:
        return None
    by_strike: Dict[int, List[float]] = defaultdict(list)
    for point in points:
        by_strike[int(point["strike"])].append(float(point["iv"]))
    nearest = min(by_strike.keys(), key=lambda strike: abs(strike - target_strike))
    vals = by_strike[nearest]
    return sum(vals) / len(vals) if vals else None


def _same_regime(value: Any, regime_id: int) -> bool:
    if _missing_regime(value):
        return False
    try:
        return int(value) % 4 == regime_id
    except (TypeError, ValueError):
        return False


def _iv_surface_figure(store: "DataStore") -> go.Figure:
    data, spots, ts_ms, error = _iv_surface.snapshot()
    points = _parse_iv_points(data, assets=("BTC",))
    if not points:
        message = "Waiting for Binance BTC option IV data"
        if error:
            message = f"{message}: {error}"
        return _empty_figure(message, height=500)

    expiries = sorted({(p["dte"], p["expiry_key"], p["expiry_label"]) for p in points})[:8]
    expiry_keys = {exp_key for _, exp_key, _ in expiries}
    points = [p for p in points if p["expiry_key"] in expiry_keys]

    spot = _safe_float(spots.get("BTC")) or _safe_float(store.last_price)
    if spot <= 0:
        strikes_all = sorted({p["strike"] for p in points})
        spot = float(strikes_all[len(strikes_all) // 2]) if strikes_all else 0.0
    if spot <= 0:
        return _empty_figure("No BTC option strikes available", height=490)

    moneyness = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
    by_expiry: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for point in points:
        by_expiry[point["expiry_key"]].append(point)
    y = [dte for dte, _, _ in expiries]
    y_labels = [label for _, _, label in expiries]
    z: List[List[float]] = []
    for _, exp_key, _ in expiries:
        exp_points = by_expiry.get(exp_key, [])
        row: List[float] = []
        fallback = sum(p["iv"] for p in exp_points) / len(exp_points) if exp_points else 0.0
        for mny in moneyness:
            iv = _nearest_iv(exp_points, spot * (1.0 + mny / 100.0))
            row.append(float(iv if iv is not None else fallback))
        if any(v > 0 for v in row):
            z.append(row)
        else:
            z.append([0.0 for _ in moneyness])

    fig = go.Figure(
        data=[
            go.Surface(
                x=moneyness,
                y=y,
                z=z,
                colorscale="Plasma",
                colorbar={"title": "IV %", "thickness": 12},
                contours={
                    "z": {
                        "show": True,
                        "usecolormap": True,
                        "highlightcolor": TEXT,
                        "project_z": True,
                    }
                },
                hovertemplate=(
                    "Moneyness %{x:+.0f}%<br>"
                    "DTE %{y} days<br>"
                    "IV %{z:.2f}%<extra></extra>"
                ),
            )
        ]
    )

    age = int(time.time() * 1000 - ts_ms) if ts_ms else 0
    title = f"BTC IV Surface — {len(points)} contracts"
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL_2,
        font={"family": "Inter, monospace", "color": TEXT},
        title={"text": title, "x": 0.02, "xanchor": "left", "font": {"size": 14}},
        margin={"l": 10, "r": 10, "t": 52, "b": 10},
        height=490,
        uirevision=f"iv-surface-{ts_ms}",
        datarevision=f"iv-surface-{ts_ms}-{age // 1000}",
        scene={
            "bgcolor": PANEL,
            "xaxis": {
                "title": "Moneyness",
                "gridcolor": GRID,
                "showbackground": True,
                "backgroundcolor": PANEL_2,
                "ticksuffix": "%",
            },
            "yaxis": {
                "title": "Expiry",
                "gridcolor": GRID,
                "showbackground": True,
                "backgroundcolor": PANEL_2,
                "tickmode": "array",
                "tickvals": y,
                "ticktext": y_labels,
            },
            "zaxis": {
                "title": "IV %",
                "gridcolor": GRID,
                "showbackground": True,
                "backgroundcolor": PANEL_2,
            },
            "camera": {"eye": {"x": 1.35, "y": 1.55, "z": 0.95}},
            "aspectmode": "cube",
        },
    )
    return fig


def _vol_term_structure_figure() -> go.Figure:
    data, spots, ts_ms, error = _iv_surface.snapshot()
    points = _parse_iv_points(data)
    if not points:
        return _empty_figure("Vol Term Structure — Awaiting Binance Options API", height=330, sub=error or "Connecting to eapi.binance.com...")

    by_asset_expiry: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    expiry_meta: Dict[Tuple[str, str], Tuple[int, str]] = {}
    for point in points:
        if point["dte"] <= 0:
            continue
        key = (point["underlying"], point["expiry_key"])
        by_asset_expiry[key].append(point)
        expiry_meta[key] = (point["dte"], point["expiry_label"])

    asset_counts = defaultdict(int)
    for point in points:
        asset_counts[point["underlying"]] += 1
    assets = [asset for asset, _ in sorted(asset_counts.items(), key=lambda item: item[1], reverse=True)[:5]]

    fig = go.Figure()
    palette = {
        "BTC": "#ffffff",
        "ETH": GREEN,
        "SOL": ORANGE,
        "BNB": RED,
        "XRP": "#b8b8b8",
    }
    for asset in assets:
        rows: List[Tuple[int, str, float]] = []
        for (asset_key, expiry_key), exp_points in by_asset_expiry.items():
            if asset_key != asset:
                continue
            dte, label = expiry_meta[(asset_key, expiry_key)]
            spot = _safe_float(spots.get(asset))
            if spot <= 0 and exp_points:
                strikes = sorted({p["strike"] for p in exp_points})
                spot = float(strikes[len(strikes) // 2])
            iv = _nearest_iv(exp_points, spot) if spot > 0 else None
            if iv is not None:
                rows.append((dte, label, iv))
        rows.sort(key=lambda row: row[0])
        if not rows:
            continue
        fig.add_trace(
            go.Scatter(
                x=[row[0] for row in rows],
                y=[row[2] for row in rows],
                customdata=[row[1] for row in rows],
                mode="lines+markers",
                name=asset,
                line={"color": palette.get(asset, MUTED), "width": 2},
                marker={"size": 6},
                hovertemplate="%{fullData.name}<br>%{customdata}<br>DTE %{x}<br>ATM IV %{y:.2f}%<extra></extra>",
            )
        )

    age = int(time.time() * 1000 - ts_ms) if ts_ms else 0
    _style_figure(fig, title="Crypto Vol Term Structure", height=330)
    fig.update_layout(uirevision=f"vol-term-{ts_ms}", datarevision=f"vol-term-{ts_ms}-{age // 1000}")
    fig.update_layout(showlegend=True)
    fig.update_xaxes(title_text="Days to expiry", gridcolor=GRID)
    fig.update_yaxes(title_text="ATM IV", ticksuffix="%", gridcolor=GRID)
    return fig


def _top_trader_positions_figure() -> go.Figure:
    rows, ts_ms, error = _top_trader_positions.snapshot()
    if len(rows) < 2:
        return _empty_figure("Top Trader Positions — Connecting...", height=430, sub=error or "fapi.binance.com/futures/data/topLongShortPositionRatio")

    df = pd.DataFrame(rows)
    latest = df.iloc[-1]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["long_pct"],
            mode="lines",
            name="Top long %",
            line={"color": GREEN, "width": 2.2},
            fill="tozeroy",
            fillcolor="rgba(43, 214, 123, 0.16)",
            hovertemplate="Long %{y:.2f}%<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["short_pct"],
            mode="lines",
            name="Top short %",
            line={"color": RED, "width": 2.2},
            fill="tozeroy",
            fillcolor="rgba(255, 77, 109, 0.14)",
            hovertemplate="Short %{y:.2f}%<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["ratio"],
            mode="lines+markers",
            name="L/S ratio",
            line={"color": CYAN, "width": 1.7},
            marker={"size": 4},
            hovertemplate="Ratio %{y:.3f}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=[df["time"].iloc[0], df["time"].iloc[-1]],
            y=[1.0, 1.0],
            mode="lines",
            name="Neutral",
            line={"color": MUTED, "width": 1, "dash": "dot"},
            hoverinfo="skip",
        ),
        secondary_y=True,
    )

    age = int(time.time() * 1000 - ts_ms) if ts_ms else 0
    title = (
        "Top Trader Positions  "
        f"Long {latest['long_pct']:.1f}%  Short {latest['short_pct']:.1f}%  Ratio {latest['ratio']:.2f}"
    )
    _style_figure(fig, title=title, height=430)
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(title_text="Position share", ticksuffix="%", range=[0, 100], gridcolor=GRID, secondary_y=False)
    fig.update_yaxes(title_text="Long / Short", gridcolor=GRID, secondary_y=True)
    return fig


def _perp_premium_figure() -> go.Figure:
    """
    FIXED: BTCUSDT Perpetual Funding Rate chart.
    Replaced broken Table-in-subplot with a pure bar chart + annotation overlay.
    The Table subplot caused blank rendering in Plotly Dash with shared_xaxes=False.
    """
    current, history, ts_ms, error = _perp_premium.snapshot()
    if not history and not current:
        return _empty_figure("Perp Premium — Connecting to Binance Futures API", height=430,
                             sub=error or "fapi.binance.com/fapi/v1/premiumIndex")

    fig = go.Figure()

    if history:
        df_h = pd.DataFrame(history)
        bar_colors = [GREEN if r <= 0 else (RED if r > 0.02 else ORANGE) for r in df_h["rate"]]
        fig.add_trace(go.Bar(
            x=df_h["time"], y=df_h["rate"],
            name="Funding Rate (8h)",
            marker={"color": bar_colors, "opacity": 0.82},
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:+.4f}%<extra></extra>",
        ))
        if len(df_h) > 1:
            fig.add_hline(y=0, line={"color": BORDER, "width": 1, "dash": "dot"})

    # Live metrics as annotation box (avoids broken Table-in-subplot)
    mark  = _safe_float(current.get("mark_price"))
    index_ = _safe_float(current.get("index_price"))
    fr    = _safe_float(current.get("last_funding_rate"))
    prem  = _safe_float(current.get("premium"))
    next_ts  = int(current.get("next_funding_time", 0))
    next_str = _ts_to_utc(next_ts).strftime("%H:%M UTC") if next_ts > 0 else "--"
    fr_col   = "#22d97a" if fr <= 0 else ("#f43f5e" if fr > 0.02 else "#f59e0b")
    prem_col = "#22d97a" if prem <= 0 else "#f43f5e"

    if mark > 0:
        lines = [
            f"<span style='color:#4b5563'>Mark  </span> <b>${mark:,.2f}</b>",
            f"<span style='color:#4b5563'>Index </span> <b>${index_:,.2f}</b>",
            f"<span style='color:#4b5563'>Spread</span> <b style='color:{prem_col}'>{prem:+.4f}%</b>",
            f"<span style='color:#4b5563'>FR    </span> <b style='color:{fr_col}'>{fr:+.4f}%</b>",
            f"<span style='color:#4b5563'>Next  </span> <b style='color:#f59e0b'>{next_str}</b>",
        ]
        fig.add_annotation(
            text="<br>".join(lines),
            x=1.0, y=1.0, xref="paper", yref="paper",
            xanchor="right", yanchor="top", showarrow=False,
            font={"size": 10, "family": "Inter, monospace", "color": TEXT},
            bgcolor="rgba(6,6,10,0.88)", bordercolor=BORDER, borderwidth=1, borderpad=8,
            align="left",
        )

    ts_utc = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    title  = (f"Perp Premium Index  {fr:+.4f}%  Mark ${mark:,.2f}  {ts_utc}"
              if mark > 0 else "BTCUSDT Perp — Funding Rate History")
    _style_figure(fig, title=title, height=430)
    fig.update_layout(showlegend=False, uirevision=f"perp-premium-{ts_ms}")
    fig.update_xaxes(gridcolor=GRID, linecolor=BORDER, tickfont={"size": 9})
    fig.update_yaxes(title_text="Funding Rate %", ticksuffix="%",
                     gridcolor=GRID, linecolor=BORDER, tickfont={"size": 9})
    return fig


def _perp_ms_figure() -> go.Figure:
    """Real-time millisecond BTCUSDT perpetual premium index (10-min rolling window)."""
    samples, error = _perp_ms_index.snapshot()

    if not samples:
        return _empty_figure("Perp Premium (RT) — Connecting...", height=430, sub=error or "fapi.binance.com @ 250ms polling")

    ts_ms_arr = [s[0] for s in samples]
    marks = [s[1] for s in samples]
    indices = [s[2] for s in samples]
    prems_bps = [s[3] for s in samples]
    times = [datetime.fromtimestamp(t / 1000, tz=timezone.utc) for t in ts_ms_arr]

    latest_prem = prems_bps[-1] if prems_bps else 0.0
    latest_mark = marks[-1] if marks else 0.0

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.52, 0.48],
    )

    bar_colors = [GREEN if p <= 0 else (RED if p > 10 else ORANGE) for p in prems_bps]
    fig.add_trace(
        go.Bar(
            x=times, y=prems_bps,
            name="Premium bps",
            marker={"color": bar_colors, "opacity": 0.80},
            hovertemplate="%{x|%H:%M:%S.%L}<br>%{y:+.2f} bps<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_hline(y=0, line={"color": MUTED, "width": 1, "dash": "dot"}, row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=times, y=marks,
            mode="lines", name="Mark",
            line={"color": CYAN, "width": 1.8},
            hovertemplate="%{x|%H:%M:%S.%L}<br>Mark $%{y:,.2f}<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=times, y=indices,
            mode="lines", name="Index",
            line={"color": MUTED, "width": 1.4, "dash": "dash"},
            hovertemplate="%{x|%H:%M:%S.%L}<br>Index $%{y:,.2f}<extra></extra>",
        ),
        row=2, col=1,
    )

    latest_time_str = datetime.fromtimestamp(ts_ms_arr[-1] / 1000, tz=timezone.utc).strftime("%H:%M:%S") if ts_ms_arr else "--"
    title = f"Perp Premium Index  {latest_prem:+.2f} bps  Mark ${latest_mark:,.2f}  {latest_time_str} UTC"
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL_2,
        font={"family": "Inter, monospace", "color": TEXT},
        title={"text": title, "x": 0.02, "xanchor": "left", "font": {"size": 13}},
        margin={"l": 48, "r": 18, "t": 50, "b": 24},
        height=430,
        hovermode="x unified",
        uirevision="perp-ms",
        datarevision=f"perp-ms-{ts_ms_arr[-1] if ts_ms_arr else 0}",
        legend={"orientation": "h", "yanchor": "top", "y": 1.0, "xanchor": "right", "x": 1,
                "font": {"size": 9}, "bgcolor": "rgba(0,0,0,0)"},
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(title_text="Premium (bps)", ticksuffix=" bps", gridcolor=GRID, row=1, col=1)
    fig.update_yaxes(title_text="Price ($)", tickprefix="$", gridcolor=GRID, row=2, col=1)
    return fig


def _max_drawdown_from_equity(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    drawdown = (equity - equity.cummax()) / equity.cummax()
    return float(drawdown.min())


def _summary_from_returns(
    label: str,
    returns: pd.Series,
    equity: pd.Series,
    n_trades: int,
) -> Dict[str, str | int]:
    if returns.empty or equity.empty:
        return {}
    n_days = max(1, len(returns))
    total = float(equity.iloc[-1] - 1.0)
    ann = (1.0 + total) ** (365.0 / n_days) - 1.0 if total > -0.999 else -1.0
    vol = float(returns.std() * math.sqrt(365.0)) if len(returns) > 1 else 0.0
    sharpe = float(returns.mean() / returns.std() * math.sqrt(365.0)) if len(returns) > 1 and returns.std() else 0.0
    downside = returns[returns < 0].std()
    sortino = float(returns.mean() / downside * math.sqrt(365.0)) if downside and not math.isnan(downside) else 0.0
    max_dd = _max_drawdown_from_equity(equity)
    calmar = ann / abs(max_dd) if max_dd else 0.0
    win_rate = float((returns > 0).mean()) if len(returns) else 0.0
    return {
        "Label": label,
        "Period": f"{equity.index[0].strftime('%Y-%m-%d')} → {equity.index[-1].strftime('%Y-%m-%d')}",
        "Total Ret %": f"{total * 100:.1f}%",
        "Ann Ret %": f"{ann * 100:.1f}%",
        "Volatility %": f"{vol * 100:.1f}%",
        "Sharpe": f"{sharpe:.2f}",
        "Sortino": f"{sortino:.2f}",
        "Max DD %": f"{max_dd * 100:.1f}%",
        "Calmar": f"{calmar:.2f}",
        "N Trades": n_trades,
        "Win Rate %": f"{win_rate * 100:.1f}%",
    }


def _strategy_path(
    returns: pd.Series,
    prices: pd.Series,
    regimes: pd.Series,
    exposure_map: Dict[int, float],
    target_vol: float = 0.55,
    dd_guard: float = 0.12,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    raw_exposure = regimes.map(exposure_map).fillna(0.0).shift(1).fillna(0.0)
    ema_50 = prices.ewm(span=50, adjust=False).mean()
    ema_200 = prices.ewm(span=200, adjust=False).mean()
    # Trend filter: require price above 200 EMA OR strong bull regime
    trend_ok = (prices.shift(1) >= ema_200.shift(1)) | (ema_50.shift(1) >= ema_200.shift(1))
    raw_exposure = raw_exposure.where(trend_ok | (regimes.shift(1) == 0), raw_exposure.clip(upper=0.08))

    # Tighter vol-targeted position sizing (55% target vs 65% before)
    realised_vol = returns.rolling(14).std() * math.sqrt(365.0)
    vol_scale = (target_vol / realised_vol).clip(lower=0.20, upper=1.10).shift(1).fillna(1.0)
    target_exposure = (raw_exposure * vol_scale).clip(lower=0.0, upper=1.10)

    # Bear regime hard gate: if regime=BEAR, force to 0
    bear_mask = (regimes.shift(1) == 2)
    target_exposure = target_exposure.where(~bear_mask, 0.0)

    equity = 1.0
    peak = 1.0
    prev_exposure = 0.0
    out_returns: List[float] = []
    out_exposure: List[float] = []
    for ret, exp in zip(returns, target_exposure):
        drawdown = equity / peak - 1.0
        if drawdown <= -dd_guard:
            exp = min(float(exp), 0.08)   # hard circuit breaker — tighter than before
        elif drawdown <= -(dd_guard * 0.55):
            exp = min(float(exp), 0.28)   # early warning reduce
        exp = float(exp)
        cost = abs(exp - prev_exposure) * 0.00015  # 1.5 bps turnover cost
        strat_ret = float(ret) * exp - cost
        equity *= max(0.0, 1.0 + strat_ret)
        peak = max(peak, equity)
        out_returns.append(strat_ret)
        out_exposure.append(exp)
        prev_exposure = exp

    strategy_returns = pd.Series(out_returns, index=returns.index)
    strategy_equity = (1.0 + strategy_returns).cumprod()
    exposure = pd.Series(out_exposure, index=returns.index)
    return strategy_returns, strategy_equity, exposure


def _adaptive_performance(store: "DataStore") -> Optional[Tuple[pd.Series, pd.Series, Dict[str, Any], Dict[str, Any]]]:
    rows = _bar_rows(store.get_bars_daily())
    rows = _rows_with_regime_context(rows, store)
    if len(rows) < 25:
        return None

    price = _safe_float(store.last_price)
    if price > 0:
        now = datetime.now(timezone.utc)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        last_day = pd.Timestamp(rows[-1]["time"]).date()
        if last_day == today.date():
            rows[-1] = dict(rows[-1])
            rows[-1]["close"] = price
            rows[-1]["high"] = max(_safe_float(rows[-1].get("high")), price)
            low = _safe_float(rows[-1].get("low"), price)
            rows[-1]["low"] = min(low if low > 0 else price, price)
        elif today.date() > last_day:
            current = dict(rows[-1])
            previous = _safe_float(current.get("close"), price)
            regime_id = _current_regime_id(store)
            current.update(
                {
                    "time": today,
                    "open": previous if previous > 0 else price,
                    "high": max(previous, price) if previous > 0 else price,
                    "low": min(previous, price) if previous > 0 else price,
                    "close": price,
                    "volume": 0.0,
                    "regime": regime_id if regime_id is not None else current.get("regime"),
                }
            )
            rows.append(current)

    df = pd.DataFrame(rows)
    df = df[df["close"] > 0].copy()
    if len(df) < 25:
        return None

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.drop_duplicates(subset=["time"], keep="last").set_index("time").sort_index()
    returns = df["close"].pct_change().fillna(0.0)
    regimes = df["regime"].apply(lambda val: int(val) % 4 if not _missing_regime(val) else 3)
    bnh_equity = (1.0 + returns).cumprod()
    bnh_dd = abs(_max_drawdown_from_equity(bnh_equity))

    best: Optional[Tuple[str, pd.Series, pd.Series, pd.Series, float]] = None
    for label, exposure_map in ADAPTIVE_CANDIDATES.items():
        candidate_returns, candidate_equity, candidate_exposure = _strategy_path(
            returns,
            df["close"],
            regimes,
            exposure_map,
        )
        total = float(candidate_equity.iloc[-1] - 1.0)
        n_days = max(1, len(candidate_returns))
        ann = (1.0 + total) ** (365.0 / n_days) - 1.0 if total > -0.999 else -1.0
        dd = abs(_max_drawdown_from_equity(candidate_equity))
        vol = float(candidate_returns.std() * math.sqrt(365.0)) if len(candidate_returns) > 1 else 0.0
        sharpe = float(candidate_returns.mean() / candidate_returns.std() * math.sqrt(365.0)) if candidate_returns.std() else 0.0
        dd_bonus = max(0.0, bnh_dd - dd)
        calmar = ann / abs(dd) if dd > 0.001 else 0.0
        score = total + 0.50 * sharpe + 1.50 * dd_bonus + 0.40 * calmar - 0.25 * vol - 0.70 * dd
        if best is None or score > best[4]:
            best = (label, candidate_returns, candidate_equity, candidate_exposure, score)

    if best is None:
        return None
    label, adaptive_returns, adaptive_equity, exposure, _ = best
    n_trades = int((exposure.diff().abs().fillna(0.0) > 0.05).sum())
    bnh_d = _summary_from_returns("Buy & Hold", returns, bnh_equity, 1)
    adaptive_d = _summary_from_returns(label, adaptive_returns, adaptive_equity, n_trades)
    if not bnh_d or not adaptive_d:
        return None
    return bnh_equity, adaptive_equity, bnh_d, adaptive_d


def _performance_figure(store: "DataStore") -> go.Figure:
    bnh = store.perf_bnh
    regime = store.perf_regime
    adaptive = _adaptive_performance(store)
    if adaptive is not None:
        bnh_equity, regime_equity, bnh_d, reg_d = adaptive
    elif bnh is not None and regime is not None:
        bnh_equity = bnh.equity
        regime_equity = regime.equity
        bnh_d = bnh.to_dict()
        reg_d = regime.to_dict()
    else:
        return _empty_figure("Performance — Loading backtest metrics...", height=330, sub="Requires 25+ daily bars of history")

    # Fixed metric order — matches both PerformanceStats.to_dict() and _summary_from_returns()
    METRIC_ORDER = [
        "Total Ret %", "Ann Ret %", "Volatility %",
        "Sharpe", "Sortino", "Max DD %", "Calmar",
        "N Trades", "Win Rate %",
    ]
    metrics = [m for m in METRIC_ORDER if m in bnh_d or m in reg_d]
    bnh_values = [str(bnh_d.get(key, "—")) for key in metrics]
    reg_values = [str(reg_d.get(key, "—")) for key in metrics]
    strategy_label = str(reg_d.get("Label", "Regime"))
    period = bnh_d.get("Period", reg_d.get("Period", ""))

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.56, 0.44],
        vertical_spacing=0.06,
        specs=[[{"type": "xy"}], [{"type": "table"}]],
    )

    if bnh_equity is not None and not bnh_equity.empty:
        fig.add_trace(
            go.Scatter(
                x=bnh_equity.index,
                y=bnh_equity,
                mode="lines",
                name="Buy & Hold",
                line={"color": MUTED, "width": 1.7},
            ),
            row=1,
            col=1,
        )
    if regime_equity is not None and not regime_equity.empty:
        fig.add_trace(
            go.Scatter(
                x=regime_equity.index,
                y=regime_equity,
                mode="lines",
                name="Adaptive Regime",
                line={"color": GREEN, "width": 2.2},
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Table(
            header={
                "values": ["Metric", "Buy & Hold", strategy_label],
                "fill_color": "#050505",
                "font": {"color": TEXT, "size": 12},
                "align": "left",
            },
            cells={
                "values": [metrics, bnh_values, reg_values],
                "fill_color": [["#000000"] * len(metrics)],
                "font": {"color": [MUTED, TEXT, GREEN], "size": 12},
                "align": ["left", "right", "right"],
                "height": 22,
            },
        ),
        row=2,
        col=1,
    )

    _style_figure(fig, title=f"Performance  {period}", height=330)
    fig.update_yaxes(title_text="Equity", gridcolor=GRID, row=1, col=1)
    fig.update_xaxes(gridcolor=GRID, row=1, col=1)
    return fig


def _metric_card(label: str, value: str, sub: str = "", accent: str = CYAN) -> html.Div:
    return html.Div(
        [
            html.Div(label, className="metric-label"),
            html.Div(value, className="metric-value", style={"color": accent}),
            html.Div(sub, className="metric-sub"),
        ],
        className="metric-card",
    )


def _price_card(asset: str, price: float, change: float) -> html.Div:
    if price <= 0:
        return _metric_card(f"{asset}USDT", "--", "--", MUTED)
    if price >= 1000:
        price_text = f"${price:,.2f}"
    elif price >= 1:
        price_text = f"${price:,.3f}"
    else:
        price_text = f"${price:,.5f}"
    accent = GREEN if change >= 0 else RED
    return _metric_card(f"{asset}USDT", price_text, f"{change:+.2f}% 24h", accent)


def _status_cards(store: "DataStore") -> List[html.Div]:
    price = _safe_float(store.last_price)
    tickers, _, _ = _spot_ticker.snapshot()
    premium_current, _, _, _ = _perp_premium.snapshot()
    status = store.status
    regime = store.regime_state

    # Use SpotTicker for BTC 24h change if the store hasn't computed it yet
    chg = _safe_float(store.price_change_24h_pct)
    if chg == 0.0 and "BTC" in tickers:
        chg = _safe_float(tickers["BTC"].get("change"))

    cards = []
    for asset in CRYPTO_ASSETS:
        if asset == "BTC":
            cards.append(_price_card("BTC", price, chg))
            continue
        vals = tickers.get(asset, {})
        cards.append(_price_card(asset, _safe_float(vals.get("price")), _safe_float(vals.get("change"))))

    if regime is None:
        regime_value = "Initializing"
        regime_sub = "Waiting for HMM"
        regime_color = MUTED
    else:
        regime_id = int(regime.regime.value)
        regime_value = REGIME_LABEL.get(regime_id, regime.label)
        signal = "LONG" if int(regime.signal) == 1 else "FLAT"
        regime_sub = f"{regime.confidence:.0%} confidence / {signal}"
        regime_color = REGIME_COLOR.get(regime_id, CYAN)

    ws = "Connected" if status.ws_connected else "Offline"
    ws_color = GREEN if status.ws_connected else RED
    cpp = "C++ ingestion" if status.cpp_loaded else "Python ingestion"
    hmm = "HMM fitted" if status.model_fitted else "HMM fitting"

    # Funding rate card - show from perp premium or fall back gracefully
    fr = _safe_float(premium_current.get("last_funding_rate"))
    prem = _safe_float(premium_current.get("premium"))
    mark = _safe_float(premium_current.get("mark_price"))
    if premium_current and mark > 0:
        fr_text = f"{fr:+.4f}%"
        fr_color = GREEN if fr < 0 else (RED if fr > 0.03 else ORANGE)
        fr_sub = f"Premium {prem:+.4f}%"
    elif premium_current:
        fr_text = f"{fr:+.4f}%"
        fr_color = GREEN if fr < 0 else (RED if fr > 0.03 else ORANGE)
        fr_sub = f"Premium {prem:+.4f}%"
    else:
        fr_text = "Fetching"
        fr_color = MUTED
        fr_sub = "fapi.binance.com"

    cards.extend([
        _metric_card("Regime", regime_value, regime_sub, regime_color),
        _metric_card("Funding Rate", fr_text, fr_sub, fr_color),
        _metric_card("Feed", ws, f"{cpp} / {status.msg_per_sec:.0f} msg/s", ws_color),
        _metric_card("Model", hmm, datetime.now(timezone.utc).strftime("%H:%M UTC"), CYAN),
    ])
    return cards


def _find_port(host: str, preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, preferred))
            return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


class NautilusApp:
    REFRESH_MS = 500
    SLOW_REFRESH_MS = 8000
    SURFACE_REFRESH_MS = 5000
    POSITIONS_REFRESH_MS = 15000
    MS_REFRESH_MS = 250

    def __init__(
        self,
        store: "DataStore",
        host: str = "127.0.0.1",
        port: int = 8050,
        refresh_ms: Optional[int] = None,
    ) -> None:
        self.store = store
        self.host = os.getenv("NAUTILUS_DASH_HOST", host)
        self.port = int(os.getenv("NAUTILUS_DASH_PORT", str(port)))
        self.refresh_ms = int(os.getenv("NAUTILUS_DASH_REFRESH_MS", str(refresh_ms or self.REFRESH_MS)))
        self.slow_refresh_ms = int(os.getenv("NAUTILUS_DASH_SLOW_REFRESH_MS", str(self.SLOW_REFRESH_MS)))
        self.surface_refresh_ms = int(os.getenv("NAUTILUS_DASH_SURFACE_REFRESH_MS", str(self.SURFACE_REFRESH_MS)))
        self.positions_refresh_ms = int(os.getenv("NAUTILUS_DASH_POSITIONS_REFRESH_MS", str(self.POSITIONS_REFRESH_MS)))
        self.ms_refresh_ms = int(os.getenv("NAUTILUS_DASH_MS_REFRESH_MS", str(self.MS_REFRESH_MS)))
        self._app = Dash(
            __name__,
            title="Nautilus Crypto",
            update_title=None,
            assets_folder=str(DASH_ASSETS_DIR),
        )
        self._app.index_string = INDEX_STRING
        self._build_layout()
        self._bind_callbacks()
        _iv_surface.start()
        _top_trader_positions.start()
        _perp_premium.start()
        _perp_ms_index.start()
        _spot_ticker.start()

    def run(self) -> None:
        port = _find_port(self.host, self.port)
        self.port = port
        print(f"\nNautilus Crypto Plotly dashboard: http://{self.host}:{port}\n", flush=True)
        try:
            self._app.run(
                host=self.host,
                port=port,
                debug=False,
                use_reloader=False,
                dev_tools_hot_reload=False,
            )
        finally:
            _iv_surface.stop()
            _top_trader_positions.stop()
            _perp_premium.stop()
            _perp_ms_index.stop()
            _spot_ticker.stop()

    def _build_layout(self) -> None:
        self._app.layout = html.Div(
            [
                dcc.Interval(id="fast-refresh", interval=self.refresh_ms, n_intervals=0),
                dcc.Interval(id="slow-refresh", interval=self.slow_refresh_ms, n_intervals=0),
                dcc.Interval(id="surface-refresh", interval=self.surface_refresh_ms, n_intervals=0),
                dcc.Interval(id="positions-refresh", interval=self.positions_refresh_ms, n_intervals=0),
                dcc.Interval(id="ms-refresh", interval=self.ms_refresh_ms, n_intervals=0),
                html.Header(
                    [
                        html.Div(
                            [
                                html.Div(
                                    html.Img(
                                        src=self._app.get_asset_url("nautilus-logo.webp"),
                                        className="brand-logo-img",
                                        alt="Nautilus Crypto",
                                    ),
                                    className="brand-logo",
                                ),
                                html.Div("NAUTILUS CRYPTO", className="brand"),
                            ],
                            className="brand-block",
                        ),
                        html.Div(id="status-cards", className="top-metrics"),
                        html.Div(id="clock", className="clock"),
                    ],
                    className="topbar",
                ),
                html.Section(
                    [
                        dcc.Graph(id="daily-chart", className="chart", config={"displayModeBar": False}),
                        dcc.Graph(id="live-chart", className="chart", config={"displayModeBar": False}),
                        dcc.Graph(id="live-dist-chart", className="chart", config={"displayModeBar": False}),
                    ],
                    className="chart-grid three live-row",
                ),
                html.Section(
                    [
                        dcc.Graph(id="vol-term-chart", className="chart", config={"displayModeBar": False}),
                        dcc.Graph(id="returns-chart", className="chart", config={"displayModeBar": False}),
                        dcc.Graph(id="performance-chart", className="chart", config={"displayModeBar": False}),
                    ],
                    className="chart-grid three compact",
                ),
                html.Section(
                    [
                        dcc.Graph(id="iv-surface", className="chart", config={"displayModeBar": True}),
                        dcc.Graph(id="perp-premium-chart", className="chart", config={"displayModeBar": False}),
                        dcc.Graph(id="top-trader-positions", className="chart", config={"displayModeBar": False}),
                        dcc.Graph(id="perp-ms-chart", className="chart", config={"displayModeBar": False}),
                    ],
                    className="chart-grid lower-four",
                ),
            ],
            className="page",
        )

    def _bind_callbacks(self) -> None:
        @self._app.callback(
            Output("clock", "children"),
            Output("status-cards", "children"),
            Output("live-chart", "figure"),
            Output("live-dist-chart", "figure"),
            Input("fast-refresh", "n_intervals"),
        )
        def _refresh_fast(_: int):
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            live_rows = _live_rows(self.store, n=240)
            return (
                now,
                _status_cards(self.store),
                _live_figure(self.store, live_rows),
                _live_distribution_figure(self.store, live_rows),
            )

        @self._app.callback(
            Output("daily-chart", "figure"),
            Output("returns-chart", "figure"),
            Output("performance-chart", "figure"),
            Input("slow-refresh", "n_intervals"),
        )
        def _refresh_slow(_: int):
            return (
                _daily_figure(self.store),
                _returns_figure(self.store),
                _performance_figure(self.store),
            )

        @self._app.callback(
            Output("iv-surface", "figure"),
            Output("vol-term-chart", "figure"),
            Output("perp-premium-chart", "figure"),
            Input("surface-refresh", "n_intervals"),
        )
        def _refresh_surface(_: int):
            return _iv_surface_figure(self.store), _vol_term_structure_figure(), _perp_premium_figure()

        @self._app.callback(
            Output("top-trader-positions", "figure"),
            Input("positions-refresh", "n_intervals"),
        )
        def _refresh_positions(_: int):
            return _top_trader_positions_figure()

        @self._app.callback(
            Output("perp-ms-chart", "figure"),
            Input("ms-refresh", "n_intervals"),
        )
        def _refresh_ms(_: int):
            return _perp_ms_figure()


INDEX_STRING = f"""
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
            :root {{
                color-scheme: dark;
                --bg: {BG};
                --panel: {PANEL};
                --panel-2: {PANEL_2};
                --border: {BORDER};
                --text: {TEXT};
                --muted: {MUTED};
                --cyan: {CYAN};
            }}
            * {{
                box-sizing: border-box;
            }}
            body {{
                margin: 0;
                background: var(--bg);
                color: var(--text);
                font-family: 'IBM Plex Mono', 'JetBrains Mono', monospace;
                letter-spacing: 0;
            }}
            #react-entry-point,
            #_dash-app-content {{
                min-height: 100vh;
                background: #000000;
            }}
            .page {{
                width: 100%;
                min-height: 100vh;
                padding: 18px;
                background: #000000;
            }}
            .topbar {{
                display: grid;
                grid-template-columns: auto minmax(0, 1fr) auto;
                align-items: center;
                gap: 10px;
                min-height: 58px;
                padding: 0 4px 10px;
            }}
            .brand-block {{
                display: flex;
                align-items: center;
                gap: 8px;
                min-width: 150px;
            }}
            .brand-logo {{
                width: 34px;
                height: 34px;
                flex: 0 0 34px;
                overflow: hidden;
                background: #000000;
            }}
            .brand-logo-img {{
                display: block;
                width: 100%;
                height: 100%;
                object-fit: cover;
                object-position: center;
                transform: scale(2.35);
            }}
            .brand {{
                color: var(--cyan);
                font-size: 18px;
                font-weight: 760;
                line-height: 1;
            }}
            .subtitle {{
                color: var(--muted);
                font-size: 11px;
                margin-top: 5px;
            }}
            .clock {{
                color: var(--muted);
                font-size: 11px;
                text-align: right;
                white-space: nowrap;
            }}
            .top-metrics {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(84px, 1fr));
                gap: 6px;
                min-width: 0;
            }}
            .metric-card {{
                min-height: 43px;
                border: 1px solid var(--border);
                border-radius: 6px;
                background: var(--panel);
                padding: 6px 8px;
                overflow: hidden;
            }}
            .metric-label {{
                color: var(--muted);
                font-size: 9px;
                text-transform: uppercase;
            }}
            .metric-value {{
                margin-top: 4px;
                font-size: clamp(12px, 1.05vw, 15px);
                line-height: 1.1;
                font-weight: 730;
                overflow-wrap: anywhere;
            }}
            .metric-sub {{
                color: var(--muted);
                margin-top: 4px;
                font-size: 9px;
                overflow-wrap: anywhere;
            }}
            .chart-grid {{
                display: grid;
                gap: 12px;
                margin-bottom: 12px;
            }}
            .chart-grid.two {{
                grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
            }}
            .chart-grid.three {{
                grid-template-columns: repeat(3, minmax(0, 1fr));
            }}
            .chart-grid.live-row {{
                grid-template-columns: minmax(0, 1.08fr) minmax(0, 1.08fr) minmax(280px, 0.84fr);
            }}
            .chart-grid.surface-row {{
                grid-template-columns: minmax(0, 1.80fr) minmax(300px, 0.70fr);
                align-items: stretch;
            }}
            .chart-grid.lower {{
                grid-template-columns: repeat(3, minmax(0, 1fr));
            }}
            .chart-grid.lower-four {{
                grid-template-columns: minmax(0, 1.4fr) minmax(0, 0.9fr) minmax(0, 0.85fr) minmax(0, 0.85fr);
            }}
            .chart-grid.one {{
                grid-template-columns: minmax(0, 1fr);
            }}
            .chart {{
                border: 1px solid var(--border);
                border-radius: 8px;
                background: var(--panel);
                overflow: hidden;
            }}
            .chart,
            .chart > div,
            .chart .js-plotly-plot,
            .chart .plot-container,
            .chart .svg-container {{
                background: #000000 !important;
            }}
            .dash-graph {{
                background: #000000 !important;
            }}
            ::-webkit-scrollbar {{ width:4px; height:4px; }}
            ::-webkit-scrollbar-track {{ background:var(--bg); }}
            ::-webkit-scrollbar-thumb {{ background:#1e2028; border-radius:2px; }}
            @media (max-width: 1100px) {{
                .top-metrics,
                .chart-grid.two,
                .chart-grid.three,
                .chart-grid.live-row,
                .chart-grid.surface-row,
                .chart-grid.lower,
                .chart-grid.lower-four,
                .chart-grid.one {{
                    grid-template-columns: minmax(0, 1fr);
                }}
                .topbar {{
                    align-items: flex-start;
                    grid-template-columns: minmax(0, 1fr);
                }}
                .clock {{
                    text-align: left;
                }}
            }}
        </style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
"""
