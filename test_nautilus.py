"""
run_results.py — Nautilus terminal results
Mirrors dashboard logic exactly: 2018 start, v5 hard gate, EWM(5) Kelly.

Usage (from nautilus_v6/):
    python run_results.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from nautilus import __version__
from nautilus.config import DEFAULT_MA_WINDOW, DEFAULT_DMA_WINDOW, HMM_N_ITER, HMM_N_STATES
from nautilus.etl.loader import load_index
from nautilus.etl.macro import build_macro_features, load_bond_yield, load_rbi_repo_rate
from nautilus.strategies.momentum import compute_price_above_ma
from nautilus.strategies.regime import fit_hmm, REGIMES, N_REGIMES
from nautilus.backtests.engine import run_backtest

# ── Config — match dashboard defaults ─────────────────────────────────────────
START      = "2018-01-01"   # dashboard default start
MA_WIN     = DEFAULT_MA_WINDOW   # 45
DMA_WIN    = DEFAULT_DMA_WINDOW  # 200
TX_BPS     = 10
# v5 Hard Gate — exactly as in dashboard
HARD_GATE  = {0: 1.00, 1: 1.00, 2: 0.75, 3: 0.00, 4: 0.00}

SEP  = "─" * 62
SEP2 = "━" * 62

def fmt(v):
    if isinstance(v, float) and np.isnan(v): return "—"
    return str(v)

print(f"\n{SEP2}")
print(f"  NAUTILUS  v{__version__}   Terminal Results")
print(f"{SEP2}\n")

# ── 1. Load price ──────────────────────────────────────────────────────────────
print("[ 1/4 ]  Loading market data...")
nifty_full = load_index()
nifty_sl   = nifty_full.loc[START:].copy()
price      = nifty_sl["Close"].copy()
price.index = pd.to_datetime(price.index)

log_ret  = np.log(price / price.shift(1))
vol_21d  = log_ret.rolling(21, min_periods=10).std() * np.sqrt(252) * 100
dma      = price.rolling(DMA_WIN, min_periods=DMA_WIN // 2).mean()
cur_dma  = float(dma.dropna().iloc[-1])
dma_pct  = (float(price.iloc[-1]) / cur_dma - 1) * 100

print(f"  Nifty 50 : {len(price)} days  │  {price.index[0].date()} → {price.index[-1].date()}")
print(f"  Close    : {price.iloc[-1]:>10,.0f}")
print(f"  {DMA_WIN}D MA   : {cur_dma:>10,.0f}  ({dma_pct:+.1f}% spread)")
print(f"  21D Vol  : {vol_21d.dropna().iloc[-1]:.1f}% ann.\n")

# ── 2. Load macro ──────────────────────────────────────────────────────────────
print("[ 2/4 ]  Loading macro data...")
try:
    bond_s = load_bond_yield(start=START)
    bond_s.index = pd.to_datetime(bond_s.index).tz_localize(None)
    bond_last = float(bond_s.dropna().iloc[-1]) if not bond_s.dropna().empty else float("nan")
except Exception:
    bond_last = float("nan")

try:
    repo_s = load_rbi_repo_rate(start=START)
    repo_s.index = pd.to_datetime(repo_s.index).tz_localize(None)
    repo_last = float(repo_s.dropna().iloc[-1]) if not repo_s.dropna().empty else float("nan")
except Exception:
    repo_last = float("nan")

spread = bond_last - repo_last if not (np.isnan(bond_last) or np.isnan(repo_last)) else float("nan")
print(f"  India 10Y G-Sec : {bond_last:.3f}%" if not np.isnan(bond_last) else "  India 10Y G-Sec : unavailable")
print(f"  RBI Repo Rate   : {repo_last:.2f}%"  if not np.isnan(repo_last) else "  RBI Repo Rate   : unavailable")
print(f"  Term Spread     : {spread:+.2f}pp\n"  if not np.isnan(spread)   else "  Term Spread     : unavailable\n")

# ── 3. Fit HMM ────────────────────────────────────────────────────────────────
print(f"[ 3/4 ]  Fitting {HMM_N_STATES}-state Gaussian HMM  ({HMM_N_ITER} iterations)...")
try:
    macro_df = build_macro_features(price, start=START)
    result   = fit_hmm(price, macro_df=macro_df, n_states=HMM_N_STATES, n_iter=HMM_N_ITER)
    hmm_ok   = True
except Exception as e:
    print(f"  ⚠  HMM failed: {e}")
    result = None
    hmm_ok = False

if hmm_ok:
    states_s = pd.Series(result.states,     index=pd.to_datetime(result.dates), dtype=int)
    kelly_s  = pd.Series(result.soft_kelly, index=pd.to_datetime(result.dates))
    probs    = np.array(result.posteriors)
    cur_st   = int(result.states[-1])
    cur_sk   = float(result.soft_kelly[-1])
    # EWM(5) sharpened kelly — matches dashboard
    sk_ewm   = kelly_s.ewm(span=5, min_periods=3).mean()
    cur_sk_ewm = float(sk_ewm.iloc[-1])

    print(f"\n  Current Regime   : {REGIMES[cur_st]['emoji']}  {REGIMES[cur_st]['name']}  "
          f"({probs[-1].max():.0%} confidence)")
    print(f"  Raw Soft Kelly   : {cur_sk:.3f}×")
    print(f"  EWM(5) Kelly     : {cur_sk_ewm:.3f}×")
    print(f"  Hard Gate (v5)   : {HARD_GATE[cur_st]:.2f}×\n")

    print(f"  {'Regime':<18} {'Days':>6} {'% Time':>8} {'Gate':>7}")
    print(f"  {SEP}")
    sc = pd.Series(result.states).value_counts().sort_index()
    tot = len(result.states)
    for rid, rinfo in REGIMES.items():
        days = sc.get(rid, 0)
        pct  = days / tot * 100
        gate = HARD_GATE[rid]
        bar  = "█" * int(pct / 3)
        print(f"  {rinfo['emoji']} {rinfo['name']:<16} {days:>6}  {pct:>6.1f}%  {gate:>5.2f}×  {bar}")

    # Transition matrix
    A = np.array(result.trans_matrix)
    print(f"\n  Transition Matrix (row=from, col=to):")
    names = [r["name"][:8] for r in REGIMES.values()]
    header = "  " + " " * 16 + "".join(f"{n:>10}" for n in names)
    print(header)
    for i, rinfo in REGIMES.items():
        row_str = f"  {rinfo['emoji']} {rinfo['name']:<14}" + "".join(f"{A[i,j]:>10.2f}" for j in range(N_REGIMES))
        print(row_str)

# ── 4. Backtest ───────────────────────────────────────────────────────────────
print(f"\n[ 4/4 ]  Running backtest  (MA={MA_WIN}, cost={TX_BPS}bps/side)...")

signal_naive = compute_price_above_ma(price, window=MA_WIN)

if hmm_ok:
    state_aln        = states_s.reindex(price.index).ffill()
    hard_gate_s      = state_aln.map(HARD_GATE).fillna(1.0)
    hard_gate_shifted = hard_gate_s.shift(1).fillna(1.0)
    signal_full      = (signal_naive * hard_gate_shifted).clip(0.0, 1.0)
else:
    signal_full = signal_naive.copy()

bt_bh    = run_backtest(price, pd.Series(1.0, index=price.index).shift(1).fillna(0.0),
                        cost_bps=0, name="Buy & Hold")
bt_naive = run_backtest(price, signal_naive, cost_bps=TX_BPS, name=f"MA({MA_WIN})")
bt_full  = run_backtest(price, signal_full,  cost_bps=TX_BPS, name="+ Regime Gate (v5)")

metrics = ["Total Return", "CAGR", "Sharpe", "Sortino", "Max DD", "Calmar", "Win Rate"]

print(f"\n  {SEP2}")
print(f"  {'Metric':<18}  {'Buy & Hold':>13}  {f'MA({MA_WIN})':>13}  {'+ Regime Gate':>14}")
print(f"  {SEP}")
for m in metrics:
    v0 = fmt(bt_bh.metrics.get(m, "—"))
    v1 = fmt(bt_naive.metrics.get(m, "—"))
    v2 = fmt(bt_full.metrics.get(m, "—"))
    print(f"  {m:<18}  {v0:>13}  {v1:>13}  {v2:>14}")
print(f"  {SEP2}")

# Regime attribution
if hmm_ok:
    print(f"\n  Regime Attribution  (start={START})")
    print(f"  {'Regime':<18} {'Ann.Ret':>9} {'Sharpe':>8} {'Max DD':>9} {'Days':>6}")
    print(f"  {SEP}")
    dr = price.pct_change().dropna()
    for rid, rinfo in REGIMES.items():
        mask  = (state_aln == rid)
        r_sub = dr.reindex(state_aln[mask].index).dropna()
        if len(r_sub) > 1:
            ann = ((1 + r_sub).prod() ** (252 / len(r_sub)) - 1)
            sr  = r_sub.mean() / r_sub.std() * np.sqrt(252)
            mdd = float(((1 + r_sub).cumprod() / (1 + r_sub).cumprod().cummax() - 1).min() * 100)
        else:
            ann = sr = float("nan"); mdd = float("nan")
        days = int(mask.sum())
        ann_s = f"{ann:.1%}" if not np.isnan(ann) else "—"
        sr_s  = f"{sr:.2f}"  if not np.isnan(sr)  else "—"
        mdd_s = f"{mdd:.1f}%" if not np.isnan(mdd) else "—"
        print(f"  {rinfo['emoji']} {rinfo['name']:<16} {ann_s:>9} {sr_s:>8} {mdd_s:>9} {days:>6}")

print(f"\n  Data: yfinance  ·  RBI CSV  ·  Start {START}  ·  Tx cost {TX_BPS}bps/side")
print(f"  Not financial advice.\n")