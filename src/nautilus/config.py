"""
src/nautilus/config.py
======================
Central configuration.

    from nautilus.config import DATA_DIR, CACHE_DIR, NIFTY_INDEX_TICKER, ...

Data discovery order
--------------------
1. Walk up from CWD — covers ``streamlit run`` from the repo root.
2. Walk up from this file — covers editable (``pip install -e .``) installs.
3. Walk up from the entry-point script — covers scripts that live outside the repo.
4. Package-bundled data — covers non-editable PyPI installs where the CSVs live
   inside ``site-packages/nautilus/data/``.
5. Hard fallback to ``parents[3]`` of this file (the old behaviour).

Cache directory
---------------
Never write into the installed package directory — it may be read-only.
Resolution order:
  a. ``<repo_root>/data/cache/``  — when running from a dev checkout
  b. ``~/.cache/nautilus/``       — for pip / PyPI installs
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Data-root discovery
# ---------------------------------------------------------------------------

def _find_data_root() -> Path:
    """Return the directory that *contains* a ``data/`` sub-directory with
    ``india_10y_yield.csv`` inside it.  Falls back progressively so the
    function never raises.
    """
    def _has_data(p: Path) -> bool:
        return (p / "data" / "india_10y_yield.csv").exists()

    # 1. Walk up from CWD
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if _has_data(candidate):
            return candidate

    # 2. Walk up from this source file (editable install, src layout)
    for candidate in Path(__file__).resolve().parents:
        if _has_data(candidate):
            return candidate

    # 3. Walk up from the entry-point script
    if _sys.argv and _sys.argv[0]:
        for candidate in Path(_sys.argv[0]).resolve().parents:
            if _has_data(candidate):
                return candidate

    # 4. Package-bundled data (non-editable pip / PyPI install)
    #    CSVs live at  .../site-packages/nautilus/data/*.csv
    _pkg_data = Path(__file__).resolve().parent / "data"
    if (_pkg_data / "india_10y_yield.csv").exists():
        # Return the *parent* of ``data/`` so that DATA_DIR = ROOT / "data"
        # resolves correctly.
        return _pkg_data.parent

    # 5. Hard fallback
    return Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Public path constants
# ---------------------------------------------------------------------------

_DATA_ROOT  = _find_data_root()
DATA_DIR    = _DATA_ROOT / "data"

# Cache: write to a user-writable location when DATA_DIR is inside
# site-packages (i.e. a non-editable install).
def _cache_dir() -> Path:
    import site
    _site_dirs = {Path(p).resolve() for p in site.getsitepackages() if Path(p).exists()}
    _site_dirs.add(Path(site.getusersitepackages()).resolve())
    try:
        _resolved = DATA_DIR.resolve()
        _in_site  = any(
            str(_resolved).startswith(str(s)) for s in _site_dirs
        )
    except Exception:
        _in_site = False

    if _in_site:
        _cd = Path.home() / ".cache" / "nautilus"
    else:
        _cd = DATA_DIR / "cache"
    _cd.mkdir(parents=True, exist_ok=True)
    return _cd


CACHE_DIR   = _cache_dir()
RESULTS_DIR = _DATA_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Market tickers
# ---------------------------------------------------------------------------

NIFTY_INDEX_TICKER = "^NSEI"
BOND_YIELD_TICKER  = "NIFTYGS10YR.NS"


# ---------------------------------------------------------------------------
# Nifty 50 universe
# ---------------------------------------------------------------------------

NIFTY50_TICKERS: list[str] = [
    "RELIANCE.NS",   "TCS.NS",        "HDFCBANK.NS",   "INFY.NS",       "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS",        "SBIN.NS",        "BAJFINANCE.NS", "BHARTIARTL.NS",
    "KOTAKBANK.NS",  "LT.NS",         "AXISBANK.NS",    "ASIANPAINT.NS", "MARUTI.NS",
    "TITAN.NS",      "WIPRO.NS",      "HCLTECH.NS",     "SUNPHARMA.NS",  "POWERGRID.NS",
    "ULTRACEMCO.NS", "NTPC.NS",       "ONGC.NS",        "TATAMOTORS.NS", "TECHM.NS",
    "JSWSTEEL.NS",   "ADANIPORTS.NS", "GRASIM.NS",      "BAJAJFINSV.NS", "HINDALCO.NS",
    "TATASTEEL.NS",  "COALINDIA.NS",  "DRREDDY.NS",     "DIVISLAB.NS",   "CIPLA.NS",
    "EICHERMOT.NS",  "BPCL.NS",       "HEROMOTOCO.NS",  "APOLLOHOSP.NS", "SBILIFE.NS",
    "HDFCLIFE.NS",   "M&M.NS",        "TATACONSUM.NS",  "BAJAJ-AUTO.NS", "NESTLEIND.NS",
    "BRITANNIA.NS",  "INDUSINDBK.NS", "UPL.NS",         "VEDL.NS",       "SHREECEM.NS",
]


# ---------------------------------------------------------------------------
# Strategy defaults
# ---------------------------------------------------------------------------

DEFAULT_START_DATE = "2015-01-01"
DEFAULT_MA_WINDOW  = 45
DEFAULT_DMA_WINDOW = 200
MOMENTUM_LOOKBACK  = 252
MOMENTUM_SKIP      = 21


# ---------------------------------------------------------------------------
# HMM
# ---------------------------------------------------------------------------

HMM_N_STATES = 5
HMM_N_ITER   = 200


# ---------------------------------------------------------------------------
# Cache TTL
# ---------------------------------------------------------------------------

CACHE_TTL_HOURS = 6
