"""
src/nautilus/config.py
======================
Central configuration. Import anywhere with:

    from nautilus.config import CACHE_DIR, NIFTY_INDEX_TICKER, ...

No environment variables required — everything works out of the box.
"""
from __future__ import annotations

from pathlib import Path

# ── Root paths ─────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]   # repo root
DATA_DIR    = ROOT / "data"
CACHE_DIR   = DATA_DIR / "cache"
RESULTS_DIR = ROOT / "results"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Market tickers ─────────────────────────────────────────────────────────────
NIFTY_INDEX_TICKER = "^NSEI"
BOND_YIELD_TICKER  = "NIFTYGS10YR.NS"

# ── Nifty 50 universe ─────────────────────────────────────────────────────────
NIFTY50_TICKERS: list[str] = [
    "RELIANCE.NS",  "TCS.NS",       "HDFCBANK.NS",  "INFY.NS",      "ICICIBANK.NS",
    "HINDUNILVR.NS","ITC.NS",       "SBIN.NS",      "BAJFINANCE.NS","BHARTIARTL.NS",
    "KOTAKBANK.NS", "LT.NS",        "AXISBANK.NS",  "ASIANPAINT.NS","MARUTI.NS",
    "TITAN.NS",     "WIPRO.NS",     "HCLTECH.NS",   "SUNPHARMA.NS", "POWERGRID.NS",
    "ULTRACEMCO.NS","NTPC.NS",      "ONGC.NS",      "TATAMOTORS.NS","TECHM.NS",
    "JSWSTEEL.NS",  "ADANIPORTS.NS","GRASIM.NS",    "BAJAJFINSV.NS","HINDALCO.NS",
    "TATASTEEL.NS", "COALINDIA.NS", "DRREDDY.NS",   "DIVISLAB.NS",  "CIPLA.NS",
    "EICHERMOT.NS", "BPCL.NS",      "HEROMOTOCO.NS","APOLLOHOSP.NS","SBILIFE.NS",
    "HDFCLIFE.NS",  "M&M.NS",       "TATACONSUM.NS","BAJAJ-AUTO.NS","NESTLEIND.NS",
    "BRITANNIA.NS", "INDUSINDBK.NS","UPL.NS",       "VEDL.NS",      "SHREECEM.NS",
]

# ── Strategy defaults ──────────────────────────────────────────────────────────
DEFAULT_START_DATE = "2015-01-01"
DEFAULT_MA_WINDOW  = 45
DEFAULT_DMA_WINDOW = 200
MOMENTUM_LOOKBACK  = 252
MOMENTUM_SKIP      = 21

# ── HMM ───────────────────────────────────────────────────────────────────────
HMM_N_STATES = 5
HMM_N_ITER   = 200

# ── Cache ─────────────────────────────────────────────────────────────────────
CACHE_TTL_HOURS = 6
