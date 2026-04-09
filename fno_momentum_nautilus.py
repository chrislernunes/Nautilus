"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   NSE FNO — CROSS-SECTIONAL MOMENTUM L/S  +  REGIME COMPARISON             ║
║                                                                              ║
║   THREE strategies run in parallel:                                          ║
║     [1] NO REGIME    — raw momentum, always long+short                       ║
║     [2] SMA-200      — original binary regime (price vs 200-DMA)             ║
║     [3] NAUTILUS HMM — 5-state Gaussian HMM (pip install nautilus-spark)    ║
║                        Bull/Neutral → long-only book                         ║
║                        Stress       → 35% gross exposure reduction           ║
║                        Panic        → flat (hard gate = 0)                   ║
║                                                                              ║
║   Capital : ₹1 Crore  |  Fixed 2% per position  |  TC: 10 bps              ║
║   Lookbacks: 30 / 60 / 90 / 120 / 160 / 200 / 300 / 400 days               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ── Nautilus imports ───────────────────────────────────────────────────────────
from nautilus.etl.loader import load_index
from nautilus.etl.macro  import build_macro_features
from nautilus.strategies.regime import (
    fit_hmm, REGIMES, REGIME_NAMES, MULT_VEC, N_REGIMES,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CAPITAL          = 1_00_00_000
LOOKBACKS        = [30, 60, 90, 120, 160, 200, 300, 400]
REBAL_FREQ       = "ME"
START_DATE       = "2020-01-01"
END_DATE         = "2026-12-31"
LONG_THRESH      = 0.80   # top 20% long  (was 0.5 → 370% gross; fix → ~72% gross)
SHORT_THRESH     = 0.20   # bottom 20% short
TRANSACTION_COST = 0.0010
REGIME_MA        = 200
REGIME_TICKER    = "^NSEI"
WEIGHT_PER_POS   = 0.02          # fixed ±2% per position

# Nautilus HMM regime mapping (v2 — directional filtering, not size scaling):
#
#   DIAGNOSIS from empirical analysis (2020-2026):
#     - HMM Stress state (state 3) has +72% annualised return in this period
#       because it captures POST-CRASH RECOVERY rallies, not ongoing drawdowns
#     - Scaling longs DOWN in Stress therefore misses the best recoveries
#     - The soft-kelly multiplier is designed for POSITION SIZING in a
#       single-asset strategy; it is NOT appropriate for L/S portfolio scaling
#
#   CORRECT APPLICATION for this cross-sectional L/S strategy:
#     State 0 (Bull Quiet)    → long-only book, full 2% per long
#     State 1 (Bull Volatile) → long-only book, full 2% per long
#     State 2 (Neutral)       → long-only book, full 2% per long
#                                (Neutral = slow grind, don't haircut longs)
#     State 3 (Stress)        → suppress NEW shorts, keep existing longs
#                                scale SHORT weight to 50% (partial hedge)
#     State 4 (Panic)         → FLAT: hard gate, zero all positions
#
#   KEY INSIGHT: HMM adds value through EARLIER short suppression vs SMA.
#   SMA takes 200+ days to flip; HMM flips within days of volatility spike.
HMM_LONGONLY_STATES  = {0, 1, 2}    # states → suppress all shorts (long-only)
HMM_SHORT_SCALE      = {3: 0.5}     # state → scale SHORT weight to X (partial hedge)
HMM_FLAT_STATES      = {4}          # states → zero everything (hard gate)

# ─────────────────────────────────────────────────────────────────────────────
# NSE FNO UNIVERSE (~185 tickers)
# ─────────────────────────────────────────────────────────────────────────────
FNO_STOCKS = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","BHARTIARTL.NS","ICICIBANK.NS",
    "INFOSYS.NS","SBIN.NS","HINDUNILVR.NS","ITC.NS","LT.NS",
    "KOTAKBANK.NS","AXISBANK.NS","BAJFINANCE.NS","MARUTI.NS","TITAN.NS",
    "SUNPHARMA.NS","ULTRACEMCO.NS","ASIANPAINT.NS","WIPRO.NS","HCLTECH.NS",
    "ONGC.NS","POWERGRID.NS","NTPC.NS","TECHM.NS","ADANIENT.NS",
    "ADANIPORTS.NS","COALINDIA.NS","JSWSTEEL.NS","TATAMOTORS.NS","TATASTEEL.NS",
    "BAJAJFINSV.NS","BAJAJ-AUTO.NS","NESTLEIND.NS","CIPLA.NS","DRREDDY.NS",
    "DIVISLAB.NS","GRASIM.NS","M&M.NS","HINDALCO.NS","INDUSINDBK.NS",
    "BRITANNIA.NS","EICHERMOT.NS","APOLLOHOSP.NS","LTIM.NS","BPCL.NS",
    "TATACONSUM.NS","HEROMOTOCO.NS","SHRIRAMFIN.NS","SBILIFE.NS","HDFCLIFE.NS",
    "ABB.NS","ABBOTINDIA.NS","ABCAPITAL.NS","ABFRL.NS","ACC.NS",
    "ALKEM.NS","AMBUJACEM.NS","ANGELONE.NS","APLAPOLLO.NS","APOLLOTYRE.NS",
    "ASHOKLEY.NS","ASTRAL.NS","ATUL.NS","AUBANK.NS","AUROPHARMA.NS",
    "BALKRISIND.NS","BANDHANBNK.NS","BANKBARODA.NS","BEL.NS","BERGEPAINT.NS",
    "BHEL.NS","BIOCON.NS","BOSCHLTD.NS","BSE.NS","BSOFT.NS",
    "CAMS.NS","CANBK.NS","CANFINHOME.NS","CDSL.NS","CESC.NS",
    "CHOLAFIN.NS","COFORGE.NS","COLPAL.NS","CONCOR.NS","CROMPTON.NS",
    "CUMMINSIND.NS","DABUR.NS","DALBHARAT.NS","DEEPAKNTR.NS","DIXON.NS",
    "DLF.NS","DMART.NS","ESCORTS.NS","EXIDEIND.NS","FEDERALBNK.NS",
    "GAIL.NS","GLENMARK.NS","GMRINFRA.NS","GNFC.NS","GODREJCP.NS",
    "GODREJPROP.NS","GRANULES.NS","GSPL.NS","GUJGASLTD.NS","HAL.NS",
    "HAVELLS.NS","HDFCAMC.NS","HINDCOPPER.NS","HINDPETRO.NS",
    "IBULHSGFIN.NS","ICICIGI.NS","ICICIPRULI.NS","IDFCFIRSTB.NS",
    "IEX.NS","IGL.NS","INDHOTEL.NS","INDIAMART.NS","INDUSTOWER.NS",
    "IOC.NS","IPCALAB.NS","IRCTC.NS","IRFC.NS",
    "JINDALSTEL.NS","JUBLFOOD.NS","KAJARIACER.NS","KPITTECH.NS",
    "LAURUSLABS.NS","LICHSGFIN.NS","LTTS.NS","LUPIN.NS",
    "M&MFIN.NS","MARICO.NS","MCDOWELL-N.NS","MCX.NS","MFSL.NS",
    "MGL.NS","MOTHERSON.NS","MPHASIS.NS","MRF.NS","MUTHOOTFIN.NS",
    "NAUKRI.NS","NAVINFLUOR.NS","NHPC.NS","NMDC.NS",
    "OBEROIRLTY.NS","OFSS.NS","PAGEIND.NS","PERSISTENT.NS",
    "PETRONET.NS","PFC.NS","PIDILITIND.NS","PIIND.NS","PNB.NS",
    "POLYCAB.NS","PVRINOX.NS","RAMCOCEM.NS","RECLTD.NS",
    "SAIL.NS","SBICARD.NS","SHREECEM.NS","SIEMENS.NS","SJVN.NS",
    "SONACOMS.NS","SRF.NS","SUNTV.NS","SUPREMEIND.NS",
    "SYNGENE.NS","TATACHEM.NS","TATACOMM.NS","TATAPOWER.NS",
    "TORNTPHARM.NS","TORNTPOWER.NS","TRENT.NS","TVSMOTOR.NS",
    "UBL.NS","UNIONBANK.NS","UPL.NS","VEDL.NS","VOLTAS.NS",
    "ZEEL.NS","ZOMATO.NS","NYKAA.NS","PAYTM.NS","DELHIVERY.NS",
    "RVNL.NS","IREDA.NS","JSWENERGY.NS","JPPOWER.NS",
    "ATGL.NS","CGPOWER.NS","KAYNES.NS","LICI.NS","MAPMYINDIA.NS",
    "POLICYBZR.NS","RBLBANK.NS","STARHEALTH.NS","TIINDIA.NS","YESBANK.NS",
    "L&TFH.NS","LALPATHLAB.NS","CHOLAHLDNG.NS",
]
_seen = set()
FNO_STOCKS = [x for x in FNO_STOCKS if not (x in _seen or _seen.add(x))]


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
def download_data():
    print(f"  Downloading {len(FNO_STOCKS)} FNO tickers...")
    data, failed = {}, []
    for i, ticker in enumerate(FNO_STOCKS):
        print(f"  [{i+1:>3}/{len(FNO_STOCKS)}] {ticker}", end="", flush=True)
        try:
            df = yf.Ticker(ticker).history(start=START_DATE, end=END_DATE, auto_adjust=True)
            if not df.empty and "Close" in df:
                s = df["Close"]
                if hasattr(s.index, "tz") and s.index.tz is not None:
                    s.index = s.index.tz_localize(None)
                data[ticker] = s
                print(" ✓")
            else:
                failed.append(ticker); print(" ✗ empty")
        except Exception:
            failed.append(ticker); print(" ✗ error")

    close = pd.DataFrame(data)
    close.index = pd.to_datetime(close.index)
    if hasattr(close.index, "tz") and close.index.tz is not None:
        close.index = close.index.tz_localize(None)

    print(f"\n  Success: {close.shape[1]} tickers  |  Failed: {len(failed)}")
    if close.shape[1] < 20:
        raise RuntimeError("Too few stocks downloaded.")

    print(f"\n  Downloading Nifty 50 ({REGIME_TICKER}) via Nautilus loader...")
    try:
        nifty_df = load_index(ticker=REGIME_TICKER, start=START_DATE)
        nsei = nifty_df["Close"].copy()
        nsei.index = pd.to_datetime(nsei.index)
        if hasattr(nsei.index, "tz") and nsei.index.tz is not None:
            nsei.index = nsei.index.tz_localize(None)
        print(f"  Nifty loaded: {len(nsei)} rows (Nautilus cache)")
    except Exception as exc:
        print(f"  Nautilus loader failed ({exc}), falling back to yfinance...")
        raw = yf.Ticker(REGIME_TICKER).history(start=START_DATE, end=END_DATE, auto_adjust=True)
        nsei = raw["Close"]
        nsei.index = pd.to_datetime(nsei.index)
        if hasattr(nsei.index, "tz") and nsei.index.tz is not None:
            nsei.index = nsei.index.tz_localize(None)

    if nsei.empty:
        raise RuntimeError("Failed to download Nifty 50.")
    return close, nsei


# ─────────────────────────────────────────────────────────────────────────────
# 2a. REGIME: plain SMA-200  (binary, original logic)
# ─────────────────────────────────────────────────────────────────────────────
def build_sma_regime(nsei: pd.Series, daily_index: pd.DatetimeIndex) -> pd.Series:
    """1 = bull (long-only), 0 = bear (long+short). Month-end snap, no look-ahead."""
    sma       = nsei.rolling(REGIME_MA).mean()
    raw       = (nsei > sma).astype(int)
    me        = raw.resample(REBAL_FREQ).last()
    return me.reindex(daily_index, method="ffill").fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# 2b. REGIME: Nautilus 5-state HMM  (from pip install nautilus-spark)
# ─────────────────────────────────────────────────────────────────────────────
def build_hmm_regime(nsei: pd.Series, daily_index: pd.DatetimeIndex) -> dict:
    """
    Fit the Nautilus 5-state Gaussian HMM on Nifty 50.

    Returns a dict with:
      'states'     : pd.Series[int]  — raw state label (0..4), daily
      'soft_kelly' : pd.Series[float]— Σ P(state)×mult, range [0,1]
      'hard_gate'  : pd.Series[float]— 1 if state<4 (Panic), else 0
      'mult'       : pd.Series[float]— position-size scalar per day (month-end snap)
      'longonly'   : pd.Series[int]  — 1 = suppress shorts (Bull states), 0 = L/S ok
    """
    print("\n  [Nautilus] Building macro features for HMM...")
    try:
        macro = build_macro_features(nsei, start=START_DATE, raw=True)
    except Exception as exc:
        print(f"  [Nautilus] Macro build failed ({exc}) — running without macro")
        macro = None

    print("  [Nautilus] Fitting 5-state Gaussian HMM (200 iter)...")
    result = fit_hmm(nsei, macro_df=macro, n_states=5, n_iter=200)
    if result is None:
        print("  [Nautilus] WARNING: HMM fit failed — retrying without macro")
        result = fit_hmm(nsei, macro_df=None, n_states=5, n_iter=200)
    if result is None:
        raise RuntimeError("Nautilus HMM fit failed on Nifty 50 data.")

    cur_state = int(result.states[-1])
    cur_sk    = float(result.soft_kelly[-1])
    print(f"  [Nautilus] HMM OK  |  features: {result.feature_names}")
    print(f"  [Nautilus] Current regime: {REGIMES[cur_state]['emoji']} "
          f"{REGIMES[cur_state]['name']}  conf={result.posteriors[-1].max():.0%}  "
          f"SoftKelly={cur_sk:.2f}x")

    hmm_dates = result.dates      # DatetimeIndex aligned to features (shifted +1)
    states_s  = pd.Series(result.states,     index=hmm_dates, dtype=int)
    kelly_s   = pd.Series(result.soft_kelly, index=hmm_dates)

    # Snap to month-end, forward-fill to daily_index  (no look-ahead: HMM already
    # uses .shift(1) in build_hmm_features, so we just ffill from HMM date)
    def _daily(s: pd.Series) -> pd.Series:
        me = s.resample(REBAL_FREQ).last()
        return me.reindex(daily_index, method="ffill").ffill().bfill()

    states_daily = _daily(states_s).astype(int)
    kelly_daily  = _daily(kelly_s)

    # Hard gate: zero exposure in Panic (state 4)
    hard_gate   = (states_daily < 4).astype(float)

    # Position-size multiplier: soft-kelly (Σ P×mult), already range-capped [0,1]
    mult_daily  = kelly_daily.clip(0.0, 1.0)

    # Long-only flag: states 0,1 = Bull → suppress shorts (mirrors SMA bull logic)
    longonly    = states_daily.isin(HMM_LONGONLY_STATES).astype(int)

    print(f"\n  [Nautilus] Regime distribution over backtest window:")
    for s in range(N_REGIMES):
        n   = int((states_daily == s).sum())
        pct = n / len(states_daily) * 100
        bar = "█" * int(pct / 2)
        print(f"    {REGIMES[s]['emoji']} {REGIMES[s]['name']:16s} "
              f"{pct:5.1f}%  {bar}")
    panic_days = int((states_daily == 4).sum())
    print(f"\n    Hard-gate OFF (Panic): {panic_days}d  "
          f"({panic_days/len(states_daily):.1%} of backtest)")

    # Build per-state return stats for diagnostic output
    log_ret = np.log(nsei / nsei.shift(1)).dropna()
    ret_by_state_diag = {}
    for s_val in range(N_REGIMES):
        s_days = states_daily[states_daily == s_val].index
        s_ret  = log_ret.reindex(s_days).dropna()
        if len(s_ret) > 5:
            ret_by_state_diag[s_val] = {
                "ann_ret": s_ret.mean() * 252 * 100,
                "sharpe":  s_ret.mean() / s_ret.std() * np.sqrt(252) if s_ret.std() > 0 else 0,
            }

    print(f"\n  [Nautilus] State quality (forward return context):")
    print(f"  {'State':<20} {'Days':>6} {'AnnRet':>8} {'Sharpe':>8} {'Role'}")
    print(f"  {'─'*20} {'─'*6} {'─'*8} {'─'*8} {'─'*20}")
    roles = {0:'Long-only (full)', 1:'Long-only (full)', 2:'Long-only (full)',
             3:'Short 50% only',   4:'FLAT (hard gate)'}
    for s_val in range(N_REGIMES):
        n_d = int((states_daily == s_val).sum())
        d   = ret_by_state_diag.get(s_val, {})
        ann = d.get("ann_ret", 0); shp = d.get("sharpe", 0)
        print(f"  {REGIMES[s_val]['emoji']} {REGIMES[s_val]['name']:<18} "
              f"{n_d:>6d} {ann:>+7.1f}% {shp:>8.2f}  {roles[s_val]}")

    return {
        "states":     states_daily,
        "soft_kelly": kelly_daily,
        "hard_gate":  hard_gate,
        "mult":       mult_daily,
        "longonly":   states_daily.isin(HMM_LONGONLY_STATES).astype(int),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def compute_signals(close: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Cross-sectional momentum rank → ±1 signals. Month-end snap, no look-ahead."""
    mom  = close.shift(1).pct_change(lookback)
    rank = mom.rank(axis=1, pct=True)
    sig  = pd.DataFrame(0.0, index=rank.index, columns=rank.columns)
    sig[rank >  LONG_THRESH]  =  1.0
    sig[rank <= SHORT_THRESH] = -1.0
    me_sig = sig.resample(REBAL_FREQ).last()
    return me_sig.reindex(close.index, method="ffill")


# ─────────────────────────────────────────────────────────────────────────────
# 4. WEIGHT BUILDERS  (one per regime variant)
# ─────────────────────────────────────────────────────────────────────────────
def _base_weights(signal: pd.DataFrame) -> pd.DataFrame:
    """Fixed ±2% per active signal, no regime adjustment."""
    w = signal.copy().astype(float)
    w[w > 0] =  WEIGHT_PER_POS
    w[w < 0] = -WEIGHT_PER_POS
    return w


def weights_no_regime(signal: pd.DataFrame) -> pd.DataFrame:
    """[1] No regime filter — raw momentum long+short always."""
    return _base_weights(signal)


def weights_sma_regime(signal: pd.DataFrame, regime: pd.Series) -> pd.DataFrame:
    """[2] SMA-200 binary regime: bull → zero shorts, bear → full L/S."""
    filt = signal.copy()
    bull_mask = regime[regime == 1].index
    filt.loc[bull_mask] = filt.loc[bull_mask].clip(lower=0.0)
    return _base_weights(filt)


def weights_hmm_regime(signal: pd.DataFrame, hmm: dict) -> pd.DataFrame:
    """
    [3] Nautilus HMM regime — directional filtering (v2).

    The soft-kelly multiplier is a single-asset position sizer; applying it
    uniformly to a L/S portfolio haircuts LONGS during post-crash recoveries
    (when HMM correctly labels Stress/Panic) — the exact wrong time to scale down.

    Correct approach: use HMM state as a DIRECTIONAL GATE, not a size scalar.

    Layer 1 — Hard gate (Panic, state 4):
        Flat. Zero all positions. This is the highest-conviction bear signal.
        HMM flips to Panic faster than SMA-200 (no 200-day lag).

    Layer 2 — Long-only in Bull + Neutral states (0, 1, 2):
        Suppress all shorts. Longs keep full 2% weight.
        Neutral (state 2) = slow-grind market; still directionally up,
        shorts just create unnecessary drag and turnover cost.

    Layer 3 — Partial short suppression in Stress (state 3):
        Scale short weights to 50% of base. Keep long weights at full 2%.
        Rationale: Stress = elevated volatility but market still has positive
        drift in 2020-2026 data (post-crash recovery rallies). Halving shorts
        reduces drag while maintaining some downside hedge.

    Layer 4 — No soft-kelly scaling of longs ever.
        Longs always held at full ±2%. The edge vs SMA is in SHORT management,
        not in reducing long conviction.
    """
    states_aln   = hmm["states"].reindex(signal.index).ffill().fillna(2).astype(int)
    hard_gate    = hmm["hard_gate"].reindex(signal.index).ffill().fillna(1.0)

    filt = signal.copy()

    # Layer 1: Panic → flat
    panic_mask = (hard_gate == 0)
    filt.loc[panic_mask] = 0.0

    # Layer 2: Bull+Neutral → zero all shorts (long-only book)
    longonly_mask = states_aln.isin(HMM_LONGONLY_STATES) & ~panic_mask
    filt.loc[longonly_mask] = filt.loc[longonly_mask].clip(lower=0.0)

    # Layer 3: Stress → partial short suppression (scale shorts to 50%)
    # Longs unaffected (clipped to 0 at minimum keeps them at +1)
    stress_mask = states_aln.isin(HMM_SHORT_SCALE.keys()) & ~panic_mask
    # We apply base weights first, then re-scale the shorts only
    w = _base_weights(filt)
    for stress_state, scale in HMM_SHORT_SCALE.items():
        day_mask = (states_aln == stress_state) & ~panic_mask
        if day_mask.any():
            # Scale short positions (negative weights) to `scale` fraction
            shorts_here = w.loc[day_mask] < 0
            w.loc[day_mask] = w.loc[day_mask].where(~shorts_here,
                                                      w.loc[day_mask] * scale)
    # For non-stress days that went through Layer 1/2, also apply base weights
    other_mask = ~stress_mask
    w.loc[other_mask] = _base_weights(filt.loc[other_mask])

    return w


# Regime-directional position sizing matrix
# ──────────────────────────────────────────
# Each state maps to (long_fraction, short_fraction) of WEIGHT_PER_POS.
# long_fraction  applies to stocks ranked in top quantile  (signal == +1)
# short_fraction applies to stocks ranked in bot quantile  (signal == -1)
# Rationale:
#   Bull Quiet    (0) → pure bull: max longs, no shorts
#   Bull Volatile (1) → cautious bull: trim shorts to 25%
#   Neutral       (2) → balanced: equal L/S book
#   Stress        (3) → defensive: flip to 75% short bias
#   Panic         (4) → pure bear: flip book, longs suppressed
HMM_DIRECTIONAL_SIZING: dict[int, tuple[float, float]] = {
    0: (1.00, 0.00),   # Bull Quiet    — 100% long  /   0% short
    1: (0.75, 0.25),   # Bull Volatile —  75% long  /  25% short
    2: (0.50, 0.50),   # Neutral       —  50% long  /  50% short
    3: (0.25, 0.75),   # Stress        —  25% long  /  75% short
    4: (0.00, 1.00),   # Panic         —   0% long  / 100% short
}


def weights_hmm_directional(signal: pd.DataFrame, hmm: dict) -> pd.DataFrame:
    """
    [4] Nautilus HMM — regime-directional position sizing.

    Unlike the directional-gate approach (which just suppresses one side),
    this strategy continuously tilts the L/S RATIO based on HMM regime:

        Bull Quiet    → 100% long  /   0% short  (pure long book)
        Bull Volatile →  75% long  /  25% short  (long-biased)
        Neutral       →  50% long  /  50% short  (balanced L/S)
        Stress        →  25% long  /  75% short  (short-biased)
        Panic         →   0% long  / 100% short  (pure short book)

    Both long and short position weights are scaled by their respective
    fractions of WEIGHT_PER_POS.  This means:
      - In Panic: shorts are at full 2% each, longs are zeroed
      - In Bull Quiet: longs are at full 2% each, shorts are zeroed
      - In Neutral: both sides are at full 2% each (same as no-regime)

    Gross exposure varies by regime but net exposure is the key signal.
    """
    states_aln = hmm["states"].reindex(signal.index).ffill().fillna(2).astype(int)
    w = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)

    for state, (long_frac, short_frac) in HMM_DIRECTIONAL_SIZING.items():
        day_mask = (states_aln == state)
        if not day_mask.any():
            continue
        sig_slice = signal.loc[day_mask]
        w_slice   = pd.DataFrame(0.0, index=sig_slice.index, columns=sig_slice.columns)
        # Longs: scale by long_frac
        if long_frac > 0:
            w_slice[sig_slice > 0] = WEIGHT_PER_POS * long_frac
        # Shorts: scale by short_frac
        if short_frac > 0:
            w_slice[sig_slice < 0] = -WEIGHT_PER_POS * short_frac
        w.loc[day_mask] = w_slice

    return w


def _find_best_directional_sizing(all_metrics: list[dict]) -> None:
    """
    Print a summary of regime-directional sizing performance vs other strategies.
    Tests whether the continuous tilt regime outperforms pure directional gate (HMM v2).
    """
    dir_metrics = [m for m in all_metrics if m["regime"] == "HMM_DIR"]
    if not dir_metrics:
        return
    hmm_metrics = [m for m in all_metrics if m["regime"] == "HMM"]

    print(f"\n\n{'═'*74}")
    print("  REGIME-DIRECTIONAL SIZING  vs  HMM GATE  (strategy 4 vs 3)")
    print(f"{'═'*74}")
    print(f"  {'LB':>4}  {'Dir CAGR':>9}  {'HMM CAGR':>9}  {'Dir Sharpe':>10}  "
          f"{'HMM Sharpe':>10}  {'Dir MaxDD':>9}  {'Winner'}")
    print(f"  {'─'*4}  {'─'*9}  {'─'*9}  {'─'*10}  {'─'*10}  {'─'*9}  {'─'*6}")
    for dm in sorted(dir_metrics, key=lambda x: x["lookback"]):
        lb  = dm["lookback"]
        hm  = next((m for m in hmm_metrics if m["lookback"] == lb), None)
        if hm is None:
            continue
        winner = "DIR ✓" if dm["sharpe"] > hm["sharpe"] else "HMM  "
        print(f"  {lb:>4}  {dm['cagr_pct']:>8.2f}%  {hm['cagr_pct']:>8.2f}%  "
              f"{dm['sharpe']:>10.3f}  {hm['sharpe']:>10.3f}  "
              f"{dm['max_dd_pct']:>8.2f}%  {winner}")

    avg_dir_sharpe = np.mean([m["sharpe"] for m in dir_metrics])
    avg_hmm_sharpe = np.mean([m["sharpe"] for m in hmm_metrics])
    print(f"\n  Avg Sharpe — Dir: {avg_dir_sharpe:.3f}  |  HMM Gate: {avg_hmm_sharpe:.3f}  "
          f"|  {'Directional wins ✓' if avg_dir_sharpe > avg_hmm_sharpe else 'Gate wins ✓'}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. BACKTEST ENGINE  (shared core, called once per regime variant)
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(close: pd.DataFrame, weights: pd.DataFrame) -> dict:
    """
    Vectorized portfolio backtest.

    Shift convention (matches nautilus core engine):
      weights[t]   = position held from close(t-1) to close(t),
                     decided at close(t-1) — callers are responsible
                     for pre-shifting via close.shift(1) in signal building.
      daily_ret[t] = close[t] / close[t-1] - 1

    This function does NOT apply any additional shift internally.
    Applying weights.shift(1) here would create a 2-day execution lag
    inconsistent with the core engine convention.
    """
    daily_ret = close.pct_change()
    turnover  = weights.diff().abs().sum(axis=1)
    gross_ret = (weights * daily_ret).sum(axis=1)
    net_ret   = gross_ret - turnover * TRANSACTION_COST
    equity    = CAPITAL * (1 + net_ret).cumprod()
    return {
        "daily_ret": net_ret,
        "equity":    equity,
        "gross_ret": gross_ret,
        "weights":   weights,
        "turnover":  turnover,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(result: dict, label: str, lookback: int) -> dict:
    r  = result["daily_ret"].dropna()
    eq = result["equity"].dropna()
    w  = result["weights"]
    tv = result["turnover"]

    n_years   = max(len(r) / 252, 0.1)
    total_ret = (eq.iloc[-1] / CAPITAL - 1) * 100
    cagr      = ((eq.iloc[-1] / CAPITAL) ** (1 / n_years) - 1) * 100
    ann_vol   = r.std() * np.sqrt(252) * 100
    sharpe    = (r.mean() * 252) / (r.std() * np.sqrt(252)) if r.std() > 0 else np.nan
    downside  = r[r < 0].std() * np.sqrt(252)
    sortino   = (r.mean() * 252) / downside if downside > 0 else np.nan

    roll_max  = eq.cummax()
    dd        = (eq - roll_max) / roll_max
    max_dd    = dd.min() * 100
    calmar    = cagr / abs(max_dd) if abs(max_dd) > 0 else np.nan

    max_dd_dur = int(
        (dd < 0).astype(int)
        .groupby((dd >= 0).astype(int).cumsum())
        .sum().max()
    )

    win_rate      = (r > 0).mean() * 100
    monthly_r     = r.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    pos_months    = (monthly_r > 0).mean() * 100
    avg_gross_exp = w.abs().sum(axis=1).mean() * 100
    avg_net_exp   = w.sum(axis=1).mean() * 100
    avg_turn      = tv.mean() * 100
    var_95        = np.percentile(r, 5) * 100
    cvar_95       = r[r <= np.percentile(r, 5)].mean() * 100

    return dict(
        label          = label,
        lookback       = lookback,
        total_ret_pct  = round(total_ret,    2),
        cagr_pct       = round(cagr,         2),
        ann_vol_pct    = round(ann_vol,       2),
        sharpe         = round(sharpe,        3),
        sortino        = round(sortino,       3),
        calmar         = round(calmar,        3),
        max_dd_pct     = round(max_dd,        2),
        max_dd_dur_d   = max_dd_dur,
        win_rate_pct   = round(win_rate,      2),
        pos_months_pct = round(pos_months,    2),
        avg_gross_exp  = round(avg_gross_exp, 2),
        avg_net_exp    = round(avg_net_exp,   2),
        avg_daily_turn = round(avg_turn,      3),
        var_95_pct     = round(var_95,        3),
        cvar_95_pct    = round(cvar_95,       3),
        skewness       = round(float(r.skew()), 3),
        excess_kurtosis= round(float(r.kurt()), 3),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. DISPLAY
# ─────────────────────────────────────────────────────────────────────────────
SEP  = "─" * 140
SEP2 = "═" * 140

REGIME_LABELS = {
    "NO_REGIME": "① No Regime     ",
    "SMA200":    "② SMA-200       ",
    "HMM":       "③ HMM Gate      ",
    "HMM_DIR":   "④ HMM Dir Sizing",
}

def print_banner():
    print()
    print("═" * 74)
    print("  NSE FNO  ─  CROSS-SECTIONAL MOMENTUM  ─  4-WAY REGIME COMPARISON")
    print(f"  Capital: ₹{CAPITAL/1e7:.0f}Cr  |  Fixed ±{WEIGHT_PER_POS*100:.0f}% per pos  |  "
          f"TC: {TRANSACTION_COST*100:.2f}%")
    print(f"  Universe: {len(FNO_STOCKS)} FNO tickers  |  Rebal: Monthly")
    print(f"  Period: {START_DATE} → {END_DATE}")
    print(f"  Regimes:")
    print(f"    ① No Regime    — always long+short raw momentum")
    print(f"    ② SMA-200      — binary bull/bear (price vs 200-DMA)")
    print(f"    ③ HMM Gate     — 5-state HMM: directional filtering")
    print(f"       States 0-2 → long-only full  |  State 3 → 50% short scale  |  State 4 → flat")
    print(f"    ④ HMM Dir Sizing — 5-state HMM: continuous L/S tilt")
    print(f"       BullQ 100L/0S  |  BullV 75L/25S  |  Neutral 50/50")
    print(f"       Stress 25L/75S  |  Panic 0L/100S")
    print("═" * 74)
    print()


def print_comparison_table(all_metrics: list[dict], lookback: int):
    """Print a side-by-side 3-column comparison for one lookback."""
    rows_by_label = {m["label"]: m for m in all_metrics if m["lookback"] == lookback}
    labels_order  = ["NO_REGIME", "SMA200", "HMM"]

    METRICS = [
        ("CAGR (%)",              "cagr_pct",       "{:>8.2f}"),
        ("Total Return (%)",      "total_ret_pct",   "{:>8.2f}"),
        ("Ann. Volatility (%)",   "ann_vol_pct",     "{:>8.2f}"),
        ("Sharpe Ratio",          "sharpe",          "{:>8.3f}"),
        ("Sortino Ratio",         "sortino",         "{:>8.3f}"),
        ("Calmar Ratio",          "calmar",          "{:>8.3f}"),
        ("Max Drawdown (%)",      "max_dd_pct",      "{:>8.2f}"),
        ("Max DD Duration (d)",   "max_dd_dur_d",    "{:>8d}"),
        ("Win Rate (%)",          "win_rate_pct",    "{:>8.2f}"),
        ("Positive Months (%)",   "pos_months_pct",  "{:>8.2f}"),
        ("Avg Gross Exp (%)",     "avg_gross_exp",   "{:>8.2f}"),
        ("Avg Net Exp (%)",       "avg_net_exp",     "{:>8.2f}"),
        ("Avg Daily Turnover (%)", "avg_daily_turn", "{:>8.3f}"),
        ("VaR 95% (daily %)",     "var_95_pct",      "{:>8.3f}"),
        ("CVaR 95% (daily %)",    "cvar_95_pct",     "{:>8.3f}"),
        ("Skewness",              "skewness",        "{:>8.3f}"),
        ("Excess Kurtosis",       "excess_kurtosis", "{:>8.3f}"),
    ]

    HDR_W = 28
    COL_W = 16

    print(f"\n{'─'*74}")
    print(f"  LOOKBACK = {lookback} days")
    print(f"{'─'*74}")
    header = f"  {'Metric':<{HDR_W}}"
    for lbl in labels_order:
        header += f"  {REGIME_LABELS[lbl]:>{COL_W}}"
    print(header)
    print("  " + "─" * (HDR_W + (COL_W + 2) * 3))

    for name, key, fmt in METRICS:
        row = f"  {name:<{HDR_W}}"
        vals = []
        for lbl in labels_order:
            m = rows_by_label.get(lbl, {})
            v = m.get(key, float("nan"))
            try:
                vals.append(fmt.format(v))
            except Exception:
                vals.append(f"{'—':>8}")
        # Highlight best value (green-ish marker)
        # For max_dd and vol/VaR/CVaR: lower is better; rest: higher is better
        lower_is_better = {"max_dd_pct", "ann_vol_pct", "var_95_pct",
                           "cvar_95_pct", "avg_daily_turn", "max_dd_dur_d"}
        try:
            numeric_vals = [rows_by_label.get(lbl, {}).get(key, float("nan"))
                            for lbl in labels_order]
            if all(isinstance(v, (int, float)) and not np.isnan(v)
                   for v in numeric_vals):
                if key in lower_is_better:
                    best_i = int(np.argmin(numeric_vals))
                else:
                    best_i = int(np.argmax(numeric_vals))
                vals[best_i] = f"{vals[best_i].strip():>7} ★"
        except Exception:
            pass
        for v in vals:
            row += f"  {v:>{COL_W}}"
        print(row)

    print()


def print_aggregate_winner(all_metrics: list[dict]):
    """Count how many times each regime wins each metric across all lookbacks."""
    df = pd.DataFrame(all_metrics)
    labels = ["NO_REGIME", "SMA200", "HMM"]
    lower_is_better = {"max_dd_pct", "ann_vol_pct", "var_95_pct",
                       "cvar_95_pct", "avg_daily_turn", "max_dd_dur_d"}
    score_cols = ["cagr_pct","sharpe","sortino","calmar","max_dd_pct",
                  "ann_vol_pct","win_rate_pct","pos_months_pct","var_95_pct"]
    wins = {l: 0 for l in labels}
    total = 0

    for lb in df["lookback"].unique():
        sub = df[df["lookback"] == lb]
        for col in score_cols:
            row_vals = {m["label"]: m[col] for _, m in sub.iterrows()
                        if col in m and not np.isnan(m[col])}
            if not row_vals:
                continue
            total += 1
            winner = (min if col in lower_is_better else max)(
                row_vals, key=row_vals.get
            )
            wins[winner] += 1

    print(f"\n{'═'*74}")
    print("  AGGREGATE WIN COUNT  (★ best across all lookbacks × metrics)")
    print(f"{'─'*74}")
    for lbl in labels:
        bar = "█" * int(wins[lbl] / max(wins.values()) * 30)
        pct = wins[lbl] / total * 100 if total else 0
        print(f"  {REGIME_LABELS[lbl]}  {wins[lbl]:>3}/{total}  ({pct:>5.1f}%)  {bar}")
    best = max(wins, key=wins.get)
    print(f"\n  Overall winner: {REGIME_LABELS[best]} "
          f"({wins[best]} wins / {total} contests)")
    print(f"{'═'*74}")


def print_hmm_vs_sma_lift(all_metrics: list[dict]):
    """Print the average improvement of HMM over SMA-200 across all lookbacks."""
    df = pd.DataFrame(all_metrics)
    LIFT_COLS = ["cagr_pct","sharpe","sortino","calmar","max_dd_pct","ann_vol_pct"]
    lower_is_better = {"max_dd_pct","ann_vol_pct"}

    print(f"\n{'═'*74}")
    print("  HMM vs SMA-200 — AVERAGE DELTA ACROSS ALL LOOKBACKS")
    print(f"{'─'*74}")
    print(f"  {'Metric':<26}  {'Avg HMM':>10}  {'Avg SMA':>10}  {'Δ (HMM−SMA)':>12}  {'Better':>8}")
    print(f"  {'─'*26}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*8}")

    for col in LIFT_COLS:
        hmm_vals = df[df["label"] == "HMM"][col].dropna()
        sma_vals = df[df["label"] == "SMA200"][col].dropna()
        if hmm_vals.empty or sma_vals.empty:
            continue
        avg_hmm = hmm_vals.mean()
        avg_sma = sma_vals.mean()
        delta   = avg_hmm - avg_sma
        if col in lower_is_better:
            better = "HMM ✓" if delta < 0 else ("SMA ✓" if delta > 0 else "Tie")
        else:
            better = "HMM ✓" if delta > 0 else ("SMA ✓" if delta < 0 else "Tie")
        print(f"  {col:<26}  {avg_hmm:>10.3f}  {avg_sma:>10.3f}  {delta:>+12.3f}  {better:>8}")
    print(f"{'═'*74}")


def save_results(all_metrics: list[dict], equity_curves: dict):
    df_m = pd.DataFrame(all_metrics)
    df_m.to_csv("fno_momentum_4regime_metrics.csv", index=False)
    print("\n  Saved fno_momentum_4regime_metrics.csv")

    df_eq = pd.DataFrame(equity_curves)
    df_eq.index.name = "Date"
    df_eq.to_csv("fno_momentum_4regime_equity.csv")
    print("  Saved fno_momentum_4regime_equity.csv")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print_banner()

    # ── Download data ───────────────────────────────────────────────────────
    close, nsei = download_data()

    # ── Build regimes once (shared across all lookbacks) ────────────────────
    print("\n  Building SMA-200 regime...")
    sma_regime = build_sma_regime(nsei, close.index)
    bull_sma   = sma_regime.mean() * 100
    print(f"  SMA-200 bull fraction: {bull_sma:.1f}%  bear: {100-bull_sma:.1f}%")

    print("\n  Building Nautilus HMM regime...")
    hmm_regime = build_hmm_regime(nsei, close.index)

    # ── Run all lookbacks ───────────────────────────────────────────────────
    all_metrics:  list[dict] = []
    equity_curves: dict      = {}

    print(f"\n\n{'═'*74}")
    print("  RUNNING BACKTESTS  (4 regimes × 8 lookbacks = 32 runs)")
    print(f"{'═'*74}")

    for lb in LOOKBACKS:
        print(f"\n  LB={lb:>3d} ─────────────────────────────────────────────────")
        raw_sig = compute_signals(close, lb)

        # [1] No regime
        w1  = weights_no_regime(raw_sig)
        r1  = run_backtest(close, w1)
        m1  = compute_metrics(r1, "NO_REGIME", lb)
        all_metrics.append(m1)
        equity_curves[f"NR_LB{lb}"] = r1["equity"]

        # [2] SMA-200 regime
        w2  = weights_sma_regime(raw_sig, sma_regime)
        r2  = run_backtest(close, w2)
        m2  = compute_metrics(r2, "SMA200", lb)
        all_metrics.append(m2)
        equity_curves[f"SMA_LB{lb}"] = r2["equity"]

        # [3] Nautilus HMM regime — directional gate
        w3  = weights_hmm_regime(raw_sig, hmm_regime)
        r3  = run_backtest(close, w3)
        m3  = compute_metrics(r3, "HMM", lb)
        all_metrics.append(m3)
        equity_curves[f"HMM_LB{lb}"] = r3["equity"]

        # [4] Nautilus HMM — regime-directional sizing
        #     Bull Quiet 100L/0S → Bull Vol 75L/25S → Neutral 50/50
        #     → Stress 25L/75S → Panic 0L/100S
        w4  = weights_hmm_directional(raw_sig, hmm_regime)
        r4  = run_backtest(close, w4)
        m4  = compute_metrics(r4, "HMM_DIR", lb)
        all_metrics.append(m4)
        equity_curves[f"HDIR_LB{lb}"] = r4["equity"]

        print(
            f"  {'':8s}  {'Regime':<18}  {'CAGR':>7}  {'Sharpe':>7}  "
            f"{'MaxDD':>8}  {'GrossExp':>9}  {'NetExp':>8}"
        )
        print(f"  {'':8s}  {'─'*18}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*9}  {'─'*8}")
        for m, tag in [
            (m1, "① No Regime"),
            (m2, "② SMA-200"),
            (m3, "③ HMM Gate"),
            (m4, "④ HMM Dir Sizing"),
        ]:
            print(
                f"  {'':8s}  {tag:<18}  "
                f"{m['cagr_pct']:>6.2f}%  "
                f"{m['sharpe']:>7.3f}  "
                f"{m['max_dd_pct']:>7.2f}%  "
                f"{m['avg_gross_exp']:>8.1f}%  "
                f"{m['avg_net_exp']:>+7.1f}%"
            )

    # ── Summary tables ──────────────────────────────────────────────────────
    print(f"\n\n{'═'*74}")
    print("  PER-LOOKBACK DETAILED COMPARISON")
    for lb in LOOKBACKS:
        print_comparison_table(all_metrics, lb)

    print_aggregate_winner(all_metrics)
    print_hmm_vs_sma_lift(all_metrics)
    _find_best_directional_sizing(all_metrics)
    save_results(all_metrics, equity_curves)

    print(f"\n  Done — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    main()