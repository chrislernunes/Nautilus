"""
main.py
─────────────────────────────────────────────────────────────────────────────
Nautilus BTC — Main Entry Point

Usage:
  python main.py                    # full live dashboard
  python main.py --backtest         # backtest only  (no UI)
  python main.py --optimise         # WF optimisation + dashboard
  python main.py --no-cpp           # force Python-only bar builder
  python main.py --no-optim         # skip WF search, use default config
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

# ── Path bootstrap ────────────────────────────────────────────────────────────
# Ensure the project root is always on sys.path regardless of CWD.
# Without this, `from python.core.xxx import ...` fails when main.py is run
# from a parent directory or via an IDE that sets a different working dir.
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

# ── Windows asyncio fix ───────────────────────────────────────────────────────
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ─────────────────────────────────────────────────────────────────────────────
# Dependency check
# ─────────────────────────────────────────────────────────────────────────────

_REQUIRED = {
    "numpy":      "numpy>=1.26",
    "pandas":     "pandas>=2.1",
    "hmmlearn":   "hmmlearn>=0.3",
    "aiohttp":    "aiohttp>=3.9",
    "websockets": "websockets>=12",
    "plotly":     "plotly>=5.18",
    "dash":       "dash>=2.14",
    "rich":       "rich>=13",
}
_missing = []
for _mod, _pkg in _REQUIRED.items():
    try:
        __import__(_mod)
    except ImportError:
        _missing.append(_pkg)

if _missing:
    print("\nERROR: Missing Python packages. Run:\n")
    print(f"  pip install {' '.join(_missing)}\n")
    print("Or:  pip install -r requirements.txt\n")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# File-only logging (Rich owns stdout/stderr during dashboard)
# ─────────────────────────────────────────────────────────────────────────────

Path("logs").mkdir(exist_ok=True)
_fh = logging.FileHandler("logs/nautilus.log", encoding="utf-8")
_fh.setFormatter(logging.Formatter(
    "%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
))
logging.root.handlers.clear()
logging.root.addHandler(_fh)
logging.root.setLevel(logging.INFO)
for _lg in ("websockets", "asyncio", "aiohttp", "hmmlearn"):
    logging.getLogger(_lg).setLevel(logging.WARNING)

log = logging.getLogger("nautilus.main")

# ─────────────────────────────────────────────────────────────────────────────
# Optional C++ module
# nautilus_cpp is a compiled extension — linters will warn about a missing
# module because the .pyd/.so is not in the source tree.  The TYPE_CHECKING
# block below is a static-analysis stub that suppresses that warning without
# affecting runtime behaviour.
# ─────────────────────────────────────────────────────────────────────────────

if TYPE_CHECKING:
    try:
        import nautilus_cpp  # noqa: F401
    except ImportError:
        pass  # not built yet — suppress yellow underline in VS Code / Pylance

CPP_AVAILABLE = False
nautilus_cpp  = None  # type: ignore[assignment]

def _try_import_cpp() -> bool:
    global nautilus_cpp, CPP_AVAILABLE
    try:
        sys.path.insert(0, str(Path(__file__).parent / "python" / "core"))
        import nautilus_cpp as _cpp  # type: ignore[import]
        nautilus_cpp  = _cpp
        CPP_AVAILABLE = True
        log.info("nautilus_cpp loaded (LWS=%s)",
                 getattr(_cpp, "__built_with_lws__", False))
        return True
    except ImportError as exc:
        log.warning("nautilus_cpp not available: %s", exc)
        return False

# ─────────────────────────────────────────────────────────────────────────────
# Pure-Python 1-second bar builder
# ─────────────────────────────────────────────────────────────────────────────

class PythonBarBuilder:
    """1s OHLCV bar builder with per-tick store updates."""

    def __init__(self, store):
        self._store  = store
        self._bar    = None
        self._bar_ts = 0

    def on_trade(self, price: float, qty: float,
                 trade_time_ms: int, is_buyer_maker: bool) -> None:
        ts           = (trade_time_ms // 1000) * 1000
        is_taker_buy = not is_buyer_maker

        # Always update last_price immediately for live chart
        self._store.update_tick(price)

        if self._bar is None or ts > self._bar_ts:
            if self._bar is not None:
                self._emit_bar(self._bar, complete=True)
            self._bar_ts = ts
            self._bar = {
                "ts": ts, "open": price, "high": price, "low": price,
                "close": price, "vol": qty,
                "buy":  qty  if is_taker_buy else 0.0,
                "sell": 0.0  if is_taker_buy else qty,
                "vn":   price * qty, "vd": qty, "n": 1,
            }
        else:
            b = self._bar
            b["high"]  = max(b["high"], price)
            b["low"]   = min(b["low"],  price)
            b["close"] = price
            b["vol"]  += qty
            if is_taker_buy: b["buy"]  += qty
            else:             b["sell"] += qty
            b["vn"] += price * qty
            b["vd"] += qty
            b["n"]  += 1

        b = self._bar
        vwap = b["vn"] / b["vd"] if b["vd"] > 0 else b["close"]
        self._store.update_partial_bar(
            ts_ms=b["ts"], open_=b["open"], high=b["high"],
            low=b["low"], close=b["close"],
            volume=b["vol"], buy_volume=b["buy"], sell_volume=b["sell"],
            num_trades=b["n"], vwap=vwap,
        )

    def _emit_bar(self, b: dict, complete: bool) -> None:
        vwap = b["vn"] / b["vd"] if b["vd"] > 0 else b["close"]
        self._store.add_bar_1s(
            ts_ms=b["ts"], open_=b["open"], high=b["high"],
            low=b["low"], close=b["close"],
            volume=b["vol"], buy_volume=b["buy"], sell_volume=b["sell"],
            num_trades=b["n"], vwap=vwap, is_complete=complete,
        )

# ─────────────────────────────────────────────────────────────────────────────
# WebSocket runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_ws_thread(store, cpp_client, py_bar_builder, stop_event):
    import json
    from python.core.ws_bridge import WSBridge

    def on_raw_message(payload: str) -> None:
        if cpp_client is not None:
            cpp_client.feed_message(payload)
            return
        try:
            msg  = json.loads(payload)
            data = msg.get("data", msg)

            if data.get("e") == "aggTrade" or ("T" in data and "p" in data):
                price = float(data["p"])
                qty   = float(data["q"])
                t_ms  = int(data["T"])
                maker = bool(data.get("m", False))
                py_bar_builder.on_trade(price, qty, t_ms, maker)

            elif data.get("e") == "kline":
                k = data.get("k", data)
                if k.get("x") and k.get("i") == "1d":
                    from python.core.data_store import BarDaily
                    store.add_daily_bar(BarDaily(
                        ts_ms=int(k["t"]), open=float(k["o"]),
                        high=float(k["h"]), low=float(k["l"]),
                        close=float(k["c"]), volume=float(k["v"]),
                        quote_volume=float(k["q"]),
                        taker_buy_base=float(k["V"]),
                        num_trades=int(k["n"]),
                    ))

        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            log.debug("WS parse error: %s", exc)

    def on_connect():
        store.set_status(ws_connected=True, error_msg="")
        log.info("WebSocket connected")

    def on_disconnect(reason: str):
        store.set_status(ws_connected=False, error_msg=f"WS: {reason[:40]}")
        log.warning("WebSocket disconnected: %s", reason)

    bridge = WSBridge(
        on_message    = on_raw_message,
        on_connect    = on_connect,
        on_disconnect = on_disconnect,
    )

    async def _runner():
        ws_task = asyncio.create_task(bridge.run())
        while not stop_event.is_set():
            await asyncio.sleep(0.1)
        bridge.stop()
        ws_task.cancel()
        try:
            await ws_task
        except asyncio.CancelledError:
            pass

    asyncio.run(_runner())

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Nautilus BTC Regime Dashboard")
    parser.add_argument("--backtest",  action="store_true")
    parser.add_argument("--optimise",  action="store_true")
    parser.add_argument("--no-cpp",    action="store_true")
    parser.add_argument("--no-optim",  action="store_true")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("  NAUTILUS  —  Crypto Regime System  v1.0")
    log.info("=" * 60)

    use_cpp = (not args.no_cpp) and _try_import_cpp()

    from python.core.data_store import DataStore
    store = DataStore()
    store.set_status(cpp_loaded=use_cpp)

    cpp_client     = None
    py_bar_builder = None

    if use_cpp and nautilus_cpp is not None:
        def on_cpp_status(msg: str):
            log.info(msg)
            if "Connected" in msg:
                store.set_status(ws_connected=True)
            elif "lost" in msg.lower() or "error" in msg.lower():
                store.set_status(ws_connected=False)

        def on_bar_cpp(ts_ms, open_, high, low, close,
                       volume, buy_vol, sell_vol, num_trades, vwap, is_complete):
            store.update_tick(close)
            store.add_bar_1s(
                ts_ms=ts_ms, open_=open_, high=high, low=low, close=close,
                volume=volume, buy_volume=buy_vol, sell_volume=sell_vol,
                num_trades=num_trades, vwap=vwap, is_complete=is_complete,
            )

        cpp_client = nautilus_cpp.WebSocketClient(
            on_bar=on_bar_cpp, on_status=on_cpp_status,
        )
    else:
        py_bar_builder = PythonBarBuilder(store=store)

    from python.core.regime_engine import RegimeEngine
    Path("models").mkdir(exist_ok=True)
    engine = RegimeEngine(
        store               = store,
        model_path          = Path("models/hmm_model.pkl"),
        update_interval_s   = 5.0,
        refit_interval_bars = 30,
        run_optimiser       = args.optimise and not args.no_optim,
    )
    engine.start()

    if args.backtest:
        from rich.console import Console
        from rich.progress import Progress, SpinnerColumn, TextColumn
        console = Console()
        console.print("\n[bold bright_cyan]NAUTILUS[/] — Backtest mode\n")
        with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                      console=console) as prog:
            prog.add_task("Fitting HMM model...", total=None)
            for _ in range(120):
                if store.status.model_fitted:
                    break
                time.sleep(1)
        if store.perf_bnh:
            _print_backtest_results(console, store)
        else:
            console.print("[red]Model not fitted — check logs/nautilus.log[/]")
        engine.stop()
        return

    stop_ws   = threading.Event()
    ws_thread = threading.Thread(
        target=_run_ws_thread,
        args=(store, cpp_client, py_bar_builder, stop_ws),
        name="ws-bridge", daemon=True,
    )
    ws_thread.start()
    log.info("WebSocket thread started")

    for _ in range(50):
        if store.status.ws_connected:
            break
        time.sleep(0.1)

    if cpp_client is not None and getattr(nautilus_cpp, "__built_with_lws__", False):
        cpp_client.start()

    from python.dashboard.app import NautilusApp
    app = NautilusApp(store=store)
    try:
        app.run()
    finally:
        log.info("Dashboard closed")
        stop_ws.set()
        engine.stop()
        if cpp_client is not None:
            try: cpp_client.stop()
            except Exception: pass
        log.info("Shutdown complete")


def _print_backtest_results(console, store) -> None:
    from rich.table import Table
    from rich import box
    tbl = Table(
        title="[bold bright_cyan]BACKTEST RESULTS[/]",
        box=box.HEAVY_EDGE, show_header=True,
        header_style="bold bright_cyan",
    )
    tbl.add_column("Metric",          style="dim",     min_width=22)
    tbl.add_column("Buy & Hold",      justify="right", min_width=16)
    tbl.add_column("Regime Strategy", justify="right", min_width=16)
    bnh_d    = store.perf_bnh.to_dict()
    regime_d = store.perf_regime.to_dict()
    for k in bnh_d:
        if k != "Label":
            tbl.add_row(k, str(bnh_d[k]), str(regime_d[k]))
    console.print()
    console.print(tbl)
    console.print()


if __name__ == "__main__":
    main()
