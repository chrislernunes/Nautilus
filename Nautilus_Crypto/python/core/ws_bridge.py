"""
python/core/ws_bridge.py
─────────────────────────────────────────────────────────────────────────────
Pure-Python WebSocket bridge for Binance streams.

Role: Subscribe to Binance combined stream via websockets library, pump each
raw JSON frame directly into the C++ bar builder via nautilus_cpp.feed_message().

This module is the fallback when libwebsockets is not compiled into nautilus_cpp.
When libwebsockets IS compiled, ws_bridge is still used as the async supervisor
(it manages the asyncio loop and reconnect logic), while the C++ client handles
the actual socket I/O in its own OS thread.

Architecture:
  asyncio event loop (Python)
    └─ websockets.connect  → raw JSON → nautilus_cpp.feed_message()  (C++ thread)
         └─ bar builder → OHLCVBar callback → Python queue
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Callable, Optional
import websockets
from websockets.exceptions import ConnectionClosedError, WebSocketException

log = logging.getLogger("nautilus.ws_bridge")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

BINANCE_WS_BASE   = "wss://stream.binance.com:9443"
COMBINED_STREAM   = "/stream?streams=btcusdt@aggTrade/btcusdt@kline_1d"
COMBINED_STREAM_URL = BINANCE_WS_BASE + COMBINED_STREAM

INITIAL_BACKOFF   = 1.0    # seconds
MAX_BACKOFF       = 64.0
PING_INTERVAL     = 20     # seconds (keep-alive)
PING_TIMEOUT      = 10


class WSBridge:
    """
    Async WebSocket bridge.  Instantiate once, call await run() from your
    asyncio event loop.

    Parameters
    ----------
    on_message : Callable[[str], None]
        Raw JSON handler — should be nautilus_cpp_client.feed_message.
    on_connect : Optional[Callable[[], None]]
        Called each time a new connection is established.
    on_disconnect : Optional[Callable[[str], None]]
        Called on connection loss with reason string.
    """

    def __init__(
        self,
        on_message:    Callable[[str], None],
        on_connect:    Optional[Callable[[], None]]    = None,
        on_disconnect: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._on_message    = on_message
        self._on_connect    = on_connect
        self._on_disconnect = on_disconnect
        self._running       = False
        self._connected     = False
        self._msg_count     = 0
        self._last_msg_ts   = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Main loop — run forever with exponential backoff reconnection."""
        self._running = True
        backoff = INITIAL_BACKOFF

        while self._running:
            try:
                await self._connect_and_stream()
                backoff = INITIAL_BACKOFF   # reset on clean session
            except (ConnectionClosedError, WebSocketException, OSError) as exc:
                self._connected = False
                reason = str(exc)
                log.warning("WS disconnected: %s — retry in %.0fs", reason, backoff)
                if self._on_disconnect:
                    self._on_disconnect(reason)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)
            except asyncio.CancelledError:
                log.info("WS bridge cancelled")
                break
            except Exception as exc:
                log.exception("Unexpected WS error: %s", exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)

        self._running    = False
        self._connected  = False

    def stop(self) -> None:
        self._running = False

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def message_count(self) -> int:
        return self._msg_count

    @property
    def last_message_age_s(self) -> float:
        if self._last_msg_ts == 0:
            return float("inf")
        return time.monotonic() - self._last_msg_ts

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _connect_and_stream(self) -> None:
        log.info("Connecting to %s", COMBINED_STREAM_URL)

        async with websockets.connect(
            COMBINED_STREAM_URL,
            ping_interval = PING_INTERVAL,
            ping_timeout  = PING_TIMEOUT,
            max_size      = 2**20,           # 1 MiB per frame
            compression   = None,            # disable permessage-deflate for latency
        ) as ws:
            self._connected = True
            log.info("Connected to Binance WebSocket")
            if self._on_connect:
                self._on_connect()

            async for raw in ws:
                if not self._running:
                    break
                self._msg_count  += 1
                self._last_msg_ts = time.monotonic()
                # Hot path: hand raw JSON to C++ without parsing in Python
                self._on_message(raw)

        self._connected = False
