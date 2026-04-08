"""
nautilus/dashboard/regime_dashboard.py
========================================
Nautilus v5 — India Macro-Regime Research Dashboard
"""
from __future__ import annotations
import logging, time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from nautilus import __version__
from nautilus.config import (
    DEFAULT_DMA_WINDOW, DEFAULT_MA_WINDOW, DEFAULT_START_DATE,
    HMM_N_ITER, HMM_N_STATES, NIFTY_INDEX_TICKER,
)
from nautilus.etl.loader import load_index
from nautilus.etl.macro import build_macro_features, load_bond_yield, load_rbi_repo_rate
from nautilus.strategies.momentum import compute_price_above_ma, compute_price_regime
from nautilus.strategies.regime import (
    MULT_VEC, N_REGIMES, REGIME_COLS, REGIME_NAMES, REGIMES, fit_hmm, markov_forecast,
)
from nautilus.backtests.engine import compute_metrics, run_backtest

logger = logging.getLogger(__name__)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NAUTILUS | Macro-Regime",
    page_icon="🧭", layout="wide", initial_sidebar_state="expanded",
)
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, .stApp { background-color:#080c12 !important; color:#c9d1d9; font-family:'Inter',sans-serif; }
  .main .block-container { padding-top:1rem; padding-bottom:2rem; max-width:100%; }

  [data-testid="stSidebar"] {
      background: linear-gradient(180deg,#0d1117 0%,#080c12 100%) !important;
      border-right:1px solid #1a2030 !important;
  }
  [data-testid="stSidebar"] .stMarkdown h2 {
      font-size:1.05rem; letter-spacing:0.12em; text-transform:uppercase;
      color:#58a6ff; font-weight:700;
  }
  [data-testid="stSidebar"] .stMarkdown h3 {
      font-size:0.68rem; letter-spacing:0.15em; text-transform:uppercase;
      color:#484f58; font-weight:600; margin-top:1.2rem;
  }

  [data-testid="metric-container"] {
      background: linear-gradient(135deg,#0d1117 0%,#0f1520 100%);
      border:1px solid #1a2030;
      border-radius:8px; padding:14px 18px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.4), inset 0 1px 0 rgba(88,166,255,0.04);
  }
  [data-testid="metric-container"]:hover { border-color:#2d3748; }
  [data-testid="metric-container"] label {
      font-size:0.65rem !important; letter-spacing:0.14em; text-transform:uppercase;
      color:#484f58 !important; font-weight:600;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
      font-family:'JetBrains Mono',monospace !important; font-size:1.3rem !important;
      font-weight:600; color:#e6edf3;
  }
  [data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size:0.7rem; }

  .section-title {
      font-size:0.68rem; font-weight:700; letter-spacing:0.2em; text-transform:uppercase;
      color:#3d4a5c; margin:1.8rem 0 0.6rem 0;
      padding:0 0 7px 10px;
      border-left:2px solid #1e3a5f;
      border-bottom:1px solid #0f1520;
  }

  hr { border-color:#1a2030 !important; margin:1.2rem 0; }

  .stDataFrame { font-size:0.75rem; font-family:'JetBrains Mono',monospace; }
  [data-testid="stDataFrame"] > div {
      background:#080c12; border:1px solid #1a2030; border-radius:6px;
  }

  .stButton > button {
      background:linear-gradient(135deg,#0f1e30,#162740);
      border:1px solid #1e3a5f; color:#4d9de0; font-weight:600;
      font-size:0.75rem; letter-spacing:0.1em; border-radius:6px;
  }
  .stButton > button:hover {
      background:linear-gradient(135deg,#162740,#1e3a5f);
      border-color:#2d5a8e; color:#58a6ff;
  }

  [data-testid="stPlotlyChart"] {
      border:1px solid #1a2030; border-radius:8px; overflow:hidden;
      box-shadow:0 4px 28px rgba(0,0,0,0.6);
  }

  .stMarkdown small, .stCaption, [data-testid="stCaptionContainer"] p {
      font-size:0.7rem !important; color:#3d4a5c !important;
  }
  [data-testid="stExpander"] {
      background:#080c12; border:1px solid #1a2030; border-radius:8px;
  }
  [data-testid="stExpander"] summary { font-size:0.8rem; color:#6e7681; }

  ::-webkit-scrollbar { width:4px; height:4px; }
  ::-webkit-scrollbar-track { background:#080c12; }
  ::-webkit-scrollbar-thumb { background:#1a2030; border-radius:2px; }
</style>
""", unsafe_allow_html=True)

_BG = "#080c12"; _GRID = "#0f1520"; _LINE = "#1a2030"; _FONT = "#6e7681"

def _layout(**kw):
    return dict(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        font=dict(color=_FONT, size=12),
        hovermode="x unified",
        margin=dict(l=65, r=20, t=30, b=40),
        legend=dict(bgcolor="rgba(22,27,34,0.9)", bordercolor=_LINE, borderwidth=1, font=dict(size=11)),
        xaxis=dict(showgrid=True, gridcolor=_GRID, zeroline=False, showline=True,
                   linecolor=_LINE, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor=_GRID, zeroline=False, tickfont=dict(size=10)),
        **kw,
    )

def _rgba(h, a):
    h = h.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"

def _spans(s, val):
    spans, in_s, t0 = [], False, None
    for dt, v in s.items():
        if v == val and not in_s: t0, in_s = dt, True
        elif v != val and in_s: spans.append((t0, dt)); in_s = False
    if in_s and t0 is not None: spans.append((t0, s.index[-1]))
    return spans

def _fmt(v, fmt):
    return format(v, fmt) if not (isinstance(v, float) and np.isnan(v)) else "\u2014"

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
<div style="padding:6px 0 14px 0;border-bottom:1px solid #1a2030;margin-bottom:6px">
  <div style="font-size:1.1rem;font-weight:800;letter-spacing:0.08em;color:#e6edf3;font-family:'Inter',sans-serif;">NAUTILUS</div>
  <div style="font-size:0.62rem;letter-spacing:0.2em;color:#3d4a5c;text-transform:uppercase;margin-top:2px;">Macro-Regime · v{__version__}</div>
</div>
""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### ⏱ Date Range")
    start_date = st.date_input("From", value=pd.Timestamp("2018-01-01"),
                               min_value=pd.Timestamp("2018-01-01"), max_value=pd.Timestamp.today())
    end_date   = st.date_input("To",   value=pd.Timestamp.today(),
                               min_value=pd.Timestamp("2018-01-02"), max_value=pd.Timestamp.today())
    st.markdown("### \U0001f4ca Regime Model")
    dma_window    = st.slider("DMA Window (days)",      50, 300, DEFAULT_DMA_WINDOW, 10)
    hmm_n_iter    = st.slider("HMM Iterations",         50, 500, HMM_N_ITER, 50)
    forecast_days = st.slider("Forecast Horizon (days)", 5,  60, 20, 5)
    use_macro_feat = st.toggle("Macro features in HMM", value=True,
                               help="10Y yield \u039421D, yield spread, RBI easing, 200-DMA ratio")
    st.markdown("### \U0001f4c8 Strategy")
    ma_window = st.slider("MA Window (days)",        5, 120, DEFAULT_MA_WINDOW, 5)
    tx_cost   = st.slider("Transaction Cost (bps)",  0,  50, 10, 5) / 10_000
    st.markdown("### \U0001f5a5 Panels")
    show_bands    = st.toggle("Regime Shading",      value=True)
    show_dma      = st.toggle("DMA Line",            value=True)
    show_macro    = st.toggle("Macro Panel",         value=True)
    show_yc       = st.toggle("Yield Curve Panel",   value=True)
    show_vol      = st.toggle("Volatility Panel",    value=True)
    show_drawdown = st.toggle("Drawdown Panel",      value=False)
    live_mode     = st.toggle("\U0001f534 Live Mode (60s)", value=False)
    st.markdown("---")
    refresh = st.button("\U0001f504 Refresh Data", type="primary", use_container_width=True)
    st.caption("Fetches latest Nifty 50 from yfinance.")
    st.markdown("---")
    st.markdown("**v5 Regime Multipliers**")
    for r in REGIMES.values():
        st.markdown(
            f'<span style="color:{r["color"]};font-weight:600">{r["emoji"]} {r["name"]}</span>'
            f'<span style="color:#484f58"> \u2192 {r["mult"]:.2f}\u00d7</span>',
            unsafe_allow_html=True,
        )
    st.caption("Bull states = full exposure. Panic = cash.")

if live_mode:
    st.info("\U0001f534 **Live Mode** \u2014 refreshes every 60 seconds.")
    time.sleep(60); st.rerun()

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=3_600, show_spinner=False)
def _load_nifty(force):
    return load_index(ticker=NIFTY_INDEX_TICKER, start=DEFAULT_START_DATE, force_refresh=force)

@st.cache_data(ttl=3_600, show_spinner=False)
def _load_macro_cached(force, start):
    try:
        nifty = load_index(force_refresh=force)
        df = build_macro_features(nifty["Close"], start=start, force_refresh=force)
        return df.to_dict()
    except Exception as exc:
        logger.error("Macro load failed: %s", exc); return None

@st.cache_data(ttl=3_600, show_spinner=False)
def _load_bond(force, start):
    s = load_bond_yield(start=start, force_refresh=force)
    if s.empty: return {}
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return {str(k): v for k, v in s.sort_index().items()}

@st.cache_data(ttl=86_400, show_spinner=False)
def _load_repo(force, start):
    s = load_rbi_repo_rate(start=start, force_refresh=force)
    if s.empty: return {}
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return {str(k): v for k, v in s.sort_index().items()}

@st.cache_data(ttl=3_600, show_spinner=False)
def _load_us_yields(force, start):
    """
    Load US Treasury yields via yfinance.
      ^TNX = CBOE 10-Year Treasury Note Yield (annualised %)
      ^IRX = CBOE 13-Week T-Bill Yield        (annualised %, short-rate proxy)
    Returns {"us10y": {date_str: float}, "us3m": {date_str: float}}.
    """
    try:
        import yfinance as yf
        raw = yf.download(["^TNX", "^IRX"], start=start, auto_adjust=True,
                          progress=False, threads=False)
        if raw.empty:
            raise ValueError("Empty yfinance response for ^TNX / ^IRX")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = ["_".join(c).strip() for c in raw.columns]
        raw.index = pd.to_datetime(raw.index).tz_localize(None)

        def _col(tag):
            hits = [c for c in raw.columns if tag in c and "Close" in c]
            if not hits:
                hits = [c for c in raw.columns if tag in c]
            return raw[hits[0]].dropna().sort_index() if hits else pd.Series(dtype=float)

        out = {}
        for name, tag in [("us10y", "TNX"), ("us3m", "IRX")]:
            s = _col(tag)
            if not s.empty:
                out[name] = {str(k): float(v) for k, v in s.items()}
        return out
    except Exception as exc:
        logger.warning("US yields fetch failed: %s", exc); return {}

with st.spinner("Loading market data\u2026"):
    try: nifty_df = _load_nifty(refresh)
    except Exception as exc:
        st.error(f"\u274c Failed to load Nifty 50: {exc}"); st.stop()

start_ts = pd.Timestamp(start_date)
end_ts   = pd.Timestamp(end_date)
nifty_sl = nifty_df.loc[start_ts:end_ts].copy()
if nifty_sl.empty: st.error("No data for selected date range."); st.stop()

price = nifty_sl["Close"].copy()
price.index = pd.to_datetime(price.index)

def _rebuild(d, name):
    if not d: return pd.Series(dtype=float, name=name)
    s = pd.Series(d, dtype=float)
    s.index = pd.to_datetime(s.index)
    return s.sort_index().loc[start_ts:end_ts]

macro_dict = _load_macro_cached(refresh, DEFAULT_START_DATE)
macro_df   = pd.DataFrame(macro_dict) if macro_dict else None
if macro_df is not None:
    macro_df.index = pd.to_datetime(macro_df.index)
    macro_df = macro_df.loc[start_ts:end_ts]

bond_s    = _rebuild(_load_bond(refresh, DEFAULT_START_DATE),   "bond_yield_10y")
repo_s    = _rebuild(_load_repo(refresh, DEFAULT_START_DATE),   "repo_rate")

# US Treasury yields (^TNX / ^IRX via yfinance) for the US Yield Monitor panel
_us_raw   = _load_us_yields(refresh, DEFAULT_START_DATE)
def _rebuild_us(d_all, key, name):
    d = d_all.get(key, {})
    if not d: return pd.Series(dtype=float, name=name)
    s = pd.Series(d, dtype=float); s.index = pd.to_datetime(s.index)
    return s.sort_index().loc[start_ts:end_ts].rename(name)
us10y_s = _rebuild_us(_us_raw, "us10y", "us_10y")
us3m_s  = _rebuild_us(_us_raw, "us3m",  "us_3m")

# ── Derived series ─────────────────────────────────────────────────────────────
log_ret   = np.log(price / price.shift(1))
vol_21d   = log_ret.rolling(21, min_periods=10).std() * np.sqrt(252) * 100
vol_63d   = log_ret.rolling(63, min_periods=30).std() * np.sqrt(252) * 100
vol_ratio = (vol_21d / vol_63d.replace(0, np.nan)).clip(0.3, 3.0)
dma_s     = price.rolling(dma_window, min_periods=dma_window//2).mean()
p_regime  = compute_price_regime(price, dma_window)

# ── HMM ───────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _fit_hmm_cached(pvals, pidx, macro_d, n_iter, use_macro):
    p = pd.Series(pvals, index=pd.to_datetime(pidx))
    m = None
    if use_macro and macro_d:
        m = pd.DataFrame(macro_d); m.index = pd.to_datetime(m.index)
    try:
        result = fit_hmm(p, macro_df=m, n_states=HMM_N_STATES, n_iter=n_iter)
    except RuntimeError as exc: return {"_error": str(exc)}
    except Exception as exc: return {"_error": f"{type(exc).__name__}: {exc}"}
    if result is None: return {"_error": "fit_hmm returned None \u2014 check data or hmmlearn install"}
    return {
        "posteriors":    result.posteriors.tolist(),
        "states":        result.states.tolist(),
        "trans_matrix":  result.trans_matrix.tolist(),
        "soft_kelly":    result.soft_kelly.tolist(),
        "dates":         result.dates.astype(str).tolist(),
        "feature_names": result.feature_names,
    }

with st.spinner("Fitting 5-state Gaussian HMM\u2026"):
    hmm_raw = _fit_hmm_cached(
        price.tolist(), price.index.astype(str).tolist(),
        macro_dict, hmm_n_iter, use_macro_feat,
    )

_hmm_error_msg = None
if "_error" in hmm_raw:
    _hmm_error_msg = hmm_raw["_error"]; hmm_raw = None
hmm_ok = hmm_raw is not None

if hmm_ok:
    _dates       = pd.to_datetime(hmm_raw["dates"])
    hmm_prob_df  = pd.DataFrame(hmm_raw["posteriors"], index=_dates, columns=REGIME_NAMES)
    hmm_state_s  = pd.Series(hmm_raw["states"],        index=_dates, dtype=int)
    soft_kelly_s = pd.Series(hmm_raw["soft_kelly"],    index=_dates, name="soft_kelly")
    hmm_trans_A  = np.array(hmm_raw["trans_matrix"])
    cur_probs    = np.array(hmm_raw["posteriors"][-1])
    cur_state    = int(hmm_raw["states"][-1])
    cur_sk       = float(hmm_raw["soft_kelly"][-1])
    cur_cap      = REGIMES[cur_state]["cap"]
else:
    cur_state = 0; cur_sk = 1.0; cur_cap = 1.5
    cur_probs = np.array([1.0] + [0.0]*(N_REGIMES-1))

# ── Scalar metrics ─────────────────────────────────────────────────────────────
cur_price  = float(price.iloc[-1])
prev_price = float(price.iloc[-2]) if len(price) > 1 else cur_price
day_ret    = (cur_price/prev_price - 1) * 100
cur_dma    = float(dma_s.dropna().iloc[-1]) if not dma_s.dropna().empty else np.nan
dma_pct    = (cur_price/cur_dma - 1)*100 if not np.isnan(cur_dma) else 0.0
bond_last  = float(bond_s.dropna().iloc[-1]) if not bond_s.dropna().empty else np.nan
bond_prev  = float(bond_s.dropna().iloc[-2]) if len(bond_s.dropna()) > 1 else bond_last
repo_last  = float(repo_s.dropna().iloc[-1]) if not repo_s.dropna().empty else np.nan
spread     = bond_last - repo_last if not (np.isnan(bond_last) or np.isnan(repo_last)) else np.nan
spread_prev = (float(bond_s.dropna().iloc[-2]) - repo_last if len(bond_s.dropna()) > 1 else spread)
cur_vol    = float(vol_21d.dropna().iloc[-1]) if not vol_21d.dropna().empty else np.nan
cv63       = float(vol_63d.dropna().iloc[-1]) if not vol_63d.dropna().empty else np.nan

# US 10Y last value (for the KPI row — distinct from Indian G-Sec spread)
_us10y_last = float(us10y_s.dropna().iloc[-1]) if not us10y_s.dropna().empty else np.nan
_us3m_last  = float(us3m_s.dropna().iloc[-1])  if not us3m_s.dropna().empty  else np.nan
yc_spread   = _us10y_last - _us3m_last if not (np.isnan(_us10y_last) or np.isnan(_us3m_last)) else np.nan

hmm_label  = (
    f"{REGIMES[cur_state]['emoji']} {REGIMES[cur_state]['name']} ({cur_probs.max():.0%})"
    if hmm_ok else ("\u26a0\ufe0f HMM Error" if _hmm_error_msg else "HMM Disabled")
)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
            border-bottom:1px solid #1a2030;padding-bottom:16px;margin-bottom:8px;">
  <div>
    <div style="display:flex;align-items:baseline;gap:14px">
      <span style="font-size:2rem;font-weight:800;letter-spacing:-1px;color:#e6edf3;
                   font-family:'Inter',sans-serif;line-height:1">NAUTILUS</span>
      <span style="font-size:0.65rem;font-weight:600;letter-spacing:0.2em;color:#58a6ff;
                   text-transform:uppercase;background:rgba(88,166,255,0.07);
                   border:1px solid rgba(88,166,255,0.18);border-radius:4px;
                   padding:2px 8px;font-family:'JetBrains Mono',monospace;">v{__version__}</span>
    </div>
    <div style="font-size:0.65rem;letter-spacing:0.14em;text-transform:uppercase;
                color:#3d4a5c;margin-top:4px;font-family:'Inter',sans-serif;">
      Macro-Regime Intelligence &nbsp;·&nbsp; 5-State HMM &nbsp;·&nbsp; No Look-Ahead &nbsp;·&nbsp; Research Use Only
    </div>
  </div>
  <div style="text-align:right;font-size:0.65rem;color:#1a2030;
              letter-spacing:0.1em;font-family:'JetBrains Mono',monospace;">
    NIFTY&nbsp;50 &nbsp;·&nbsp; ^GSPC &nbsp;·&nbsp; ^TNX &nbsp;·&nbsp; ^IRX
  </div>
</div>
""", unsafe_allow_html=True)

if _hmm_error_msg:
    st.error(f"\u274c **HMM Error** \u2014 {_hmm_error_msg}", icon="\U0001f6a8")

c0,c1,c2,c3,c4,c5,c6 = st.columns(7)
c0.metric("Nifty 50",          f"{cur_price:,.0f}",  delta=f"{day_ret:+.2f}% 1D")
c1.metric(f"{dma_window}D MA", f"{cur_dma:,.0f}",    delta=f"{dma_pct:+.1f}% spread")
c2.metric("HMM Regime",        hmm_label)
c3.metric("Soft Kelly",        f"{cur_sk:.2f}\u00d7",
          help="\u03a3 P(state)\u00d7mult. Bull=1.0, Neutral=0.75, Stress=0.35, Panic=0.0")
c4.metric("10Y G-Sec",
          f"{bond_last:.3f}%" if not np.isnan(bond_last) else "\u2014",
          delta=f"{(bond_last-bond_prev)*100:+.0f}bps 1D" if not (np.isnan(bond_last) or np.isnan(bond_prev)) else None)
c5.metric("RBI Repo Rate",     f"{repo_last:.2f}%" if not np.isnan(repo_last) else "\u2014")
c6.metric("US 10Y\u22123M Spread",
          f"{yc_spread:+.2f}pp" if not np.isnan(yc_spread) else "\u2014",
          delta="\u26a0\ufe0f Inverted" if (not np.isnan(yc_spread) and yc_spread < 0) else None,
          help="US 10Y Treasury \u2212 3M T-Bill. Negative = inverted US curve (recession signal).")

# Regime distribution bar
if hmm_ok:
    _sc = pd.Series(hmm_raw["states"]).value_counts().sort_index()
    _tot = len(hmm_raw["states"])
    _bar = '<div style="display:flex;height:8px;border-radius:4px;overflow:hidden;margin:6px 0 4px 0">'
    for rid in range(N_REGIMES):
        _p = _sc.get(rid, 0) / _tot * 100
        if _p > 0:
            _bar += f'<div style="width:{_p:.1f}%;background:{REGIMES[rid]["color"]};opacity:0.7"></div>'
    _bar += "</div>"
    _legend = " &nbsp;\u00b7&nbsp; ".join(
        f'<span style="color:{REGIMES[r]["color"]}">{REGIMES[r]["emoji"]} '
        f'{REGIMES[r]["name"]} {_sc.get(r,0)/_tot*100:.0f}%</span>'
        for r in range(N_REGIMES)
    )
    st.markdown(_bar + f"<p style='color:#484f58;font-size:0.75rem;margin:0'>{_legend}</p>",
                unsafe_allow_html=True)

st.markdown("---")

# ── ROW 1: Price chart ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Nifty 50 \u00b7 Regime Shading \u00b7 Soft Kelly</div>', unsafe_allow_html=True)

def _price_chart():
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if show_bands:
        shapes = []
        src = hmm_state_s if hmm_ok else p_regime.map({1:0, 0:4})
        for rid in (range(N_REGIMES) if hmm_ok else [0, 4]):
            for s, e in _spans(src, rid):
                shapes.append(dict(type="rect", xref="x", yref="paper", x0=s, x1=e, y0=0, y1=1,
                    fillcolor=_rgba(REGIMES[rid]["color"], 0.10), line_width=0, layer="below"))
        fig.update_layout(shapes=shapes)
    fig.add_trace(go.Scatter(x=price.index, y=price.values, mode="lines", name="Nifty 50",
        line=dict(color="#58a6ff", width=2),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Close: \u20b9%{y:,.0f}<extra></extra>"), secondary_y=False)
    if show_dma:
        fig.add_trace(go.Scatter(x=dma_s.index, y=dma_s.values, mode="lines", name=f"{dma_window}D MA",
            line=dict(color="#d29922", width=1.4, dash="dash"),
            hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{dma_window}D MA: \u20b9%{{y:,.0f}}<extra></extra>"),
            secondary_y=False)
    if hmm_ok:
        sk_aln = soft_kelly_s.reindex(price.index).ffill()
        # Sharpened: EWM(5) cuts lag from ~10 days (MA21) to ~3 days
        sk_sharp = sk_aln.ewm(span=5, min_periods=3).mean()
        fig.add_trace(go.Scatter(x=sk_sharp.index, y=sk_sharp.values, mode="lines",
            name="Kelly EWM(5)",
            line=dict(color="#FF7F0E", width=1.5, dash="dot"),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Kelly: %{y:.2f}\u00d7<extra></extra>"), secondary_y=True)
        fig.add_hline(y=1.0, line_dash="dot", line_color="#484f58", line_width=1, secondary_y=True)
    lo = _layout(height=420)
    lo["margin"] = dict(l=65, r=75, t=10, b=40)
    lo["legend"]["orientation"] = "h"; lo["legend"]["y"] = 1.06
    fig.update_layout(**lo)
    fig.update_yaxes(title_text="Nifty 50", tickprefix="\u20b9", tickformat=",.0f", secondary_y=False)
    fig.update_yaxes(title_text="Soft Kelly (\u00d7)", range=[0, 1.6], tickformat=".2f", secondary_y=True)
    return fig

st.plotly_chart(_price_chart(), use_container_width=True)

# ── ROW 2: Regime stack + Forecast ────────────────────────────────────────────
col_l, col_r = st.columns([3, 2])

with col_l:
    st.markdown('<div class="section-title">Regime Probability Stack</div>', unsafe_allow_html=True)
    if hmm_ok:
        fig_p = go.Figure()
        for rid in range(N_REGIMES):
            nm = REGIME_NAMES[rid]; col = REGIME_COLS[rid]
            fig_p.add_trace(go.Scatter(x=hmm_prob_df.index, y=hmm_prob_df[nm],
                mode="lines", name=nm, stackgroup="one",
                fillcolor=_rgba(col, 0.60), line=dict(color=col, width=0.5),
                hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{nm}: %{{y:.1%}}<extra></extra>"))
        lo_p = _layout(height=300)
        lo_p["yaxis"].update(range=[0,1], tickformat=".0%")
        lo_p["margin"] = dict(l=50, r=10, t=10, b=40)
        lo_p["legend"] = dict(bgcolor="rgba(22,27,34,0.9)", bordercolor=_LINE,
                              borderwidth=1, orientation="v", x=1.01, y=1, font=dict(size=10))
        fig_p.update_layout(**lo_p)
        st.plotly_chart(fig_p, use_container_width=True)
    else:
        st.error(f"\u274c {_hmm_error_msg}" if _hmm_error_msg else "HMM disabled \u2014 install `hmmlearn`")

with col_r:
    st.markdown('<div class="section-title">Regime Forward Forecast</div>', unsafe_allow_html=True)
    if hmm_ok:
        paths = markov_forecast(hmm_trans_A, cur_probs, forecast_days)
        labels = REGIME_NAMES; colors = REGIME_COLS
    else:
        reg = p_regime.values; A2 = np.zeros((2,2))
        for i in range(len(reg)-1): A2[int(reg[i]), int(reg[i+1])] += 1
        rs = A2.sum(axis=1, keepdims=True)
        A2 = np.where(rs>0, A2/rs, 0.5)
        cur_r = int(p_regime.iloc[-1]) if not p_regime.empty else 0
        paths = markov_forecast(A2, np.array([1-cur_r, float(cur_r)]), forecast_days)
        labels = ["Risk-OFF", "Risk-ON"]; colors = ["#E74C3C", "#2ECC71"]
    fig_fc = go.Figure()
    for i, (lbl, col) in enumerate(zip(labels, colors)):
        if i < paths.shape[1]:
            fig_fc.add_trace(go.Scatter(
                x=list(range(forecast_days+1)), y=paths[:,i].tolist(),
                mode="lines+markers", name=lbl,
                line=dict(color=col, width=2), marker=dict(size=4),
                hovertemplate=f"Day %{{x}}<br>{lbl}: %{{y:.1%}}<extra></extra>"))
    lo_fc = _layout(height=300)
    lo_fc["xaxis"]["title"] = "Trading Days Ahead"
    lo_fc["yaxis"].update(title="Probability", range=[0,1], tickformat=".0%")
    lo_fc["margin"] = dict(l=60, r=10, t=30, b=50)
    fig_fc.update_layout(**lo_fc)
    st.plotly_chart(fig_fc, use_container_width=True)

# ── ROW 3: Soft Kelly (full width, sharpened) ────────────────────────────────
# Pre-compute ISO calendar for weekly resamples still used by Monthly Heatmap
if hmm_ok:
    _wk_prob  = hmm_prob_df.resample("W").mean()
    _wk_dom   = _wk_prob.values.argmax(axis=1)
    _wk_dates = _wk_prob.index
    _iso_cal  = _wk_dates.isocalendar()
    _wk_df    = pd.DataFrame({
        "date":   _wk_dates,
        "year":   _iso_cal.year.values,
        "week":   _iso_cal.week.values,
        "regime": _wk_dom,
    })
    _pivot_wk = _wk_df.pivot(index="year", columns="week", values="regime")
    _name_map = {i: REGIMES[i]["name"] for i in range(N_REGIMES)}
st.markdown('<div class="section-title">Soft Kelly — Sharpened Signal History</div>', unsafe_allow_html=True)
if hmm_ok:
    # ── Compute hard gate here (also used by backtest section below) ──────
    if "_hard_gate_shifted" not in dir():
        _state_aln_early  = hmm_state_s.reindex(price.index).ffill()
        _hard_gate_early  = _state_aln_early.map({0:1.0, 1:1.0, 2:0.75, 3:0.0, 4:0.0}).fillna(1.0)
        _hard_gate_shifted = _hard_gate_early.shift(1).fillna(1.0)
    # Sharpened Kelly: EWM span=5 instead of 21D rolling → ~3–5 day lag vs 10+ day lag
    sk_ewm   = soft_kelly_s.ewm(span=5, min_periods=3).mean()
    sk_roll  = soft_kelly_s.rolling(21, min_periods=5).mean()  # kept for reference
    # Hard gate overlay on same chart
    _hg_aln  = _hard_gate_shifted.reindex(soft_kelly_s.index).ffill()

    fig_sk = make_subplots(specs=[[{"secondary_y": True}]])
    # Raw daily (faint background)
    fig_sk.add_trace(go.Scatter(x=soft_kelly_s.index, y=soft_kelly_s.values, mode="lines",
        name="Daily Kelly", line=dict(color="#58a6ff", width=0.6, dash="dot"), opacity=0.30),
        secondary_y=False)
    # EWM(5) — sharpened, responsive
    fig_sk.add_trace(go.Scatter(x=sk_ewm.index, y=sk_ewm.values, mode="lines",
        name="EWM(5) Kelly ← sharpened", line=dict(color="#2ECC71", width=2.4)),
        secondary_y=False)
    # 21D MA — reference slow signal
    fig_sk.add_trace(go.Scatter(x=sk_roll.index, y=sk_roll.values, mode="lines",
        name="MA(21) Kelly", line=dict(color="#d29922", width=1.2, dash="dash"), opacity=0.60),
        secondary_y=False)
    # Hard gate (step function on right axis)
    fig_sk.add_trace(go.Scatter(x=_hg_aln.index, y=_hg_aln.values, mode="lines",
        name="Hard Gate (t-1)", line=dict(color="#E74C3C", width=1.2, dash="dot"),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Gate: %{y:.2f}×<extra></extra>"),
        secondary_y=True)
    for yv, lbl, col in [(1.00,"1.0×","#484f58"),(0.75,"0.75×","#F1C40F"),(0.35,"0.35×","#E67E22")]:
        fig_sk.add_hline(y=yv, line_dash="dot", line_color=col, line_width=1,
                         annotation_text=lbl, annotation_font_color=col, annotation_font_size=9,
                         secondary_y=False)
    lo_sk = _layout(height=280)
    lo_sk["margin"] = dict(l=65, r=75, t=10, b=40)
    lo_sk["legend"]["orientation"] = "h"; lo_sk["legend"]["y"] = 1.10
    fig_sk.update_layout(**lo_sk)
    fig_sk.update_yaxes(title_text="Soft Kelly (×)", range=[-0.05, 1.30], tickformat=".2f", secondary_y=False)
    fig_sk.update_yaxes(title_text="Hard Gate (×)", range=[-0.05, 1.30], showgrid=False, tickformat=".2f", secondary_y=True)
    st.plotly_chart(fig_sk, use_container_width=True)
    st.caption(
        "**EWM(5)** = exponentially-weighted Kelly with 5-day half-life — ~2× faster response vs MA(21). "
        "Hard Gate (right axis, dotted red) = binary 1.0/0.75/0 from dominant HMM state, pre-shifted +1D."
    )

# ── Monthly Returns Heatmap + BRICS FX ────────────────────────────────────────
_month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

st.markdown('<div class="section-title">Monthly Returns Heatmap · BRICS FX Monitor</div>', unsafe_allow_html=True)
try: _monthly_ret = price.resample("ME").last().pct_change().dropna() * 100
except Exception: _monthly_ret = price.resample("M").last().pct_change().dropna() * 100

_mr_df = pd.DataFrame({"year":_monthly_ret.index.year,"month":_monthly_ret.index.month,"ret":_monthly_ret.values})
_pivot = _mr_df.pivot(index="year", columns="month", values="ret")
_pivot.columns = [_month_labels[m-1] for m in _pivot.columns]
_pivot["Annual"] = _mr_df.groupby("year")["ret"].apply(lambda x: (np.prod(1+x/100)-1)*100)

fig_mr = go.Figure(go.Heatmap(
    z=_pivot.values, x=_pivot.columns.tolist(), y=_pivot.index.tolist(),
    colorscale=[[0,"#7f1d1d"],[0.3,"#E74C3C"],[0.45,"#2d333b"],[0.5,"#21262d"],[0.55,"#162512"],[0.7,"#2ECC71"],[1,"#145a32"]],
    zmid=0,
    text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in _pivot.values],
    texttemplate="%{text}", textfont=dict(size=9, color="white"),
    colorbar=dict(title=dict(text="Ret%", font=dict(color=_FONT, size=10)),
                  tickfont=dict(color=_FONT, size=9), ticksuffix="%"),
    hovertemplate="<b>%{y} %{x}</b><br>Return: %{z:.2f}%<extra></extra>",
))
lo_mr = _layout(height=max(220, len(_pivot)*30+80))
lo_mr["margin"] = dict(l=55, r=80, t=40, b=10)
lo_mr["xaxis"]["side"] = "top"
fig_mr.update_layout(**lo_mr)

# BRICS FX loader (shared with standalone panel below if shown)
_BRICS_SERIES = {
    "INR": ("CCUSMA02INM618N", "#F0A500", "India (INR)"),
    "CNY": ("CCUSMA02CNM618N", "#58a6ff", "China (CNY)"),
    "BRL": ("CCUSMA02BRM618N", "#3fb950", "Brazil (BRL)"),
    "ZAR": ("CCUSMA02ZAM618N", "#d29922", "S.Africa (ZAR)"),
    "RUB": ("CCUSMA02RUM618N", "#E74C3C", "Russia (RUB)"),
}

@st.cache_data(ttl=86_400, show_spinner=False)
def _load_brics_fx(force: bool) -> dict:
    """Fetch BRICS FX monthly series from FRED public CSV endpoint (no API key)."""
    import urllib.request
    out = {}
    for ccy, (series_id, _, _label) in _BRICS_SERIES.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                lines = resp.read().decode().strip().splitlines()
            vals = {}
            for row in lines[1:]:
                parts = row.split(",")
                if len(parts) == 2 and parts[1].strip() != ".":
                    try:
                        vals[parts[0].strip()] = float(parts[1].strip())
                    except ValueError:
                        pass
            out[ccy] = vals
        except Exception as exc:
            logger.warning("FRED fetch failed for %s: %s", ccy, exc)
            out[ccy] = {}
    return out

_brics_raw = _load_brics_fx(refresh)

def _build_brics_frames(raw: dict, s_ts, e_ts) -> dict:
    frames = {}
    for ccy, vals in raw.items():
        if vals:
            s = pd.Series(vals, dtype=float)
            s.index = pd.to_datetime(s.index)
            s = s.sort_index().loc[s_ts:e_ts]
            if not s.empty:
                frames[ccy] = s
    return frames

_brics_frames = _build_brics_frames(_brics_raw, start_ts, end_ts)

# Side-by-side layout
_col_hm, _col_fx = st.columns([3, 2])

with _col_hm:
    st.plotly_chart(fig_mr, use_container_width=True)

with _col_fx:
    if _brics_frames:
        _norm_start = max(s.index[0] for s in _brics_frames.values())
        fig_brics_sm = go.Figure()
        for ccy, s in _brics_frames.items():
            _, color, label = _BRICS_SERIES[ccy]
            _base_slice = s.loc[s.index >= _norm_start]
            if _base_slice.empty:
                continue
            _base = float(_base_slice.iloc[0])
            s_norm = s / _base * 100
            fig_brics_sm.add_trace(go.Scatter(
                x=s_norm.index, y=s_norm.values,
                mode="lines", name=label,
                line=dict(color=color, width=1.6),
                hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{label}: %{{y:.1f}}<extra></extra>",
            ))
            # latest label
            _lv = float(s_norm.iloc[-1])
            fig_brics_sm.add_annotation(
                x=s_norm.index[-1], y=_lv,
                text=f"  {_lv:.0f}",
                showarrow=False, xanchor="left",
                font=dict(color=color, size=8, family="JetBrains Mono, monospace"),
            )
        lo_sm = _layout(height=max(220, len(_pivot)*30+80))
        lo_sm["margin"] = dict(l=50, r=45, t=40, b=10)
        lo_sm["hovermode"] = "x unified"
        lo_sm["legend"] = dict(
            bgcolor="rgba(22,27,34,0.85)", bordercolor=_LINE, borderwidth=1,
            font=dict(size=9), x=0.01, y=0.99,
        )
        lo_sm["title"] = dict(
            text="BRICS FX vs USD (norm. 100)",
            font=dict(color=_FONT, size=11),
            x=0.03, y=0.97,
        )
        # ── High-uncertainty red patch (COVID shock: Mar–Jun 2020) ──────────
        _unc_spans = [
            ("2020-02-20", "2020-06-30", "COVID-19 Shock"),
        ]
        for _u0, _u1, _ulbl in _unc_spans:
            _ustart = pd.Timestamp(_u0)
            _uend   = pd.Timestamp(_u1)
            if _ustart >= _norm_start or _uend >= _norm_start:
                fig_brics_sm.add_vrect(
                    x0=_ustart, x1=_uend,
                    fillcolor=_rgba("#E74C3C", 0.13),
                    line_width=0.8, line_color=_rgba("#E74C3C", 0.35),
                    layer="below",
                )
                fig_brics_sm.add_annotation(
                    x=_ustart + (_uend - _ustart) / 2,
                    y=1.0, yref="paper",
                    text=_ulbl,
                    showarrow=False,
                    font=dict(color=_rgba("#E74C3C", 0.70), size=8,
                              family="JetBrains Mono, monospace"),
                    xanchor="center", yanchor="top",
                )

        fig_brics_sm.update_yaxes(
            title_text="Index (100 = base)", showgrid=True,
            gridcolor=_GRID, tickfont=dict(size=9),
        )
        fig_brics_sm.update_layout(**lo_sm)
        st.plotly_chart(fig_brics_sm, use_container_width=True)
        st.caption("Monthly avg USD FX · OECD via FRED · higher = weaker local ccy")
    else:
        st.warning("⚠️ BRICS FX unavailable (fred.stlouisfed.org)")

# ── Macro Panel ───────────────────────────────────────────────────────────────
if show_macro:
    st.markdown('<div class="section-title">Macro Panel \u2014 RBI Repo Rate & 10Y G-Sec Yield</div>', unsafe_allow_html=True)
    _bond_ok = not bond_s.empty; _repo_ok = not repo_s.empty
    if not _bond_ok and not _repo_ok:
        st.warning("Macro data unavailable. Check bundled CSVs in data/.")
    else:
        bond_aln = bond_s.reindex(price.index).ffill() if _bond_ok else pd.Series(dtype=float)
        repo_aln = repo_s.reindex(price.index).ffill() if _repo_ok else pd.Series(dtype=float)

        # Both on single chart but on SEPARATE Y-AXES (secondary_y)
        fig_m = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            subplot_titles=["10Y G-Sec Yield & RBI Repo Rate (separate axes)",
                            "Yield Spread \u2014 Term Premium (pp)"],
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        )

        if _bond_ok:
            fig_m.add_trace(go.Scatter(x=bond_aln.index, y=bond_aln.values, mode="lines",
                name="10Y G-Sec", line=dict(color="#58a6ff", width=1.8),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>10Y: %{y:.3f}%<extra></extra>"),
                row=1, col=1, secondary_y=False)

        if _repo_ok:
            fig_m.add_trace(go.Scatter(x=repo_aln.index, y=repo_aln.values, mode="lines",
                name="RBI Repo Rate", line=dict(color="#f0883e", width=2.0, dash="dash"),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Repo: %{y:.2f}%<extra></extra>"),
                row=1, col=1, secondary_y=True)

        if _bond_ok and _repo_ok:
            spr = (bond_aln - repo_aln).rename("spread")
            spr_color = ["#2ECC71" if v>=0 else "#E74C3C" for v in spr.fillna(0)]
            fig_m.add_trace(go.Bar(x=spr.index, y=spr.values, name="Yield Spread",
                marker_color=spr_color, opacity=0.80,
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Spread: %{y:+.2f}pp<extra></extra>"),
                row=2, col=1)
            fig_m.add_hline(y=0, line_color=_LINE, line_width=1, row=2, col=1)

        lo_m = _layout(height=440)
        lo_m.update(showlegend=True)
        lo_m["legend"]["orientation"] = "h"; lo_m["legend"]["y"] = 1.03
        lo_m["margin"] = dict(l=65, r=65, t=50, b=40)

        # Primary y-axis (10Y) on left
        fig_m.update_yaxes(title_text="10Y G-Sec (%)", ticksuffix="%",
                            showgrid=True, gridcolor=_GRID, tickfont=dict(size=10, color="#58a6ff"),
                            row=1, col=1, secondary_y=False)
        # Secondary y-axis (Repo) on right
        fig_m.update_yaxes(title_text="RBI Repo Rate (%)", ticksuffix="%",
                            showgrid=False, tickfont=dict(size=10, color="#f0883e"),
                            row=1, col=1, secondary_y=True)
        # Spread axis
        fig_m.update_yaxes(title_text="Spread (pp)", showgrid=True, gridcolor=_GRID,
                            zeroline=True, zerolinecolor=_LINE, tickfont=dict(size=10),
                            row=2, col=1)

        fig_m.update_layout(**lo_m)
        fig_m.update_annotations(font_color=_FONT, font_size=11)
        st.plotly_chart(fig_m, use_container_width=True)

        _rbi = {
            "2020-03-27":"COVID cut \u221275bps","2020-05-22":"COVID cut \u221240bps",
            "2022-05-04":"Hike +40bps","2022-06-08":"Hike +50bps",
            "2022-08-05":"Hike +50bps","2023-02-08":"Final hike +25bps",
            "2025-02-07":"Cut \u221225bps","2025-04-09":"Cut \u221225bps",
        }
        _visible = {d:l for d,l in _rbi.items() if start_ts <= pd.Timestamp(d) <= end_ts}
        if _visible:
            st.caption("\U0001f3db\ufe0f **RBI Policy Events:** " +
                       " \u00b7 ".join(f"`{d}` {l}" for d,l in _visible.items()))

# ── US Macro Monitor — SPX / 10Y-3M Spread / Fed Funds Rate ─────────────────
if show_yc:
    st.markdown(
        '<div class="section-title">US Macro Monitor'
        ' — S&amp;P 500 · 10Y−3M Treasury Spread · Fed Funds Rate</div>',
        unsafe_allow_html=True,
    )

    _us10y_ok = not us10y_s.empty
    _us3m_ok  = not us3m_s.empty

    # ── Load SPX (^GSPC) ─────────────────────────────────────────────────
    @st.cache_data(ttl=3_600, show_spinner=False)
    def _load_spx(force, start):
        try:
            import yfinance as yf
            raw = yf.download("^GSPC", start=start, auto_adjust=True,
                              progress=False, threads=False)
            if raw.empty:
                return {}
            # Flatten MultiIndex that yfinance >=0.2.x returns for single tickers
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = ["_".join(str(c) for c in col).strip() for col in raw.columns]
            raw.index = pd.to_datetime(raw.index).tz_localize(None)
            close_hits = [c for c in raw.columns if "Close" in c or "close" in c]
            col = close_hits[0] if close_hits else raw.columns[0]
            s = raw[col].dropna().sort_index()
            return {str(k): float(v) for k, v in s.items()}
        except Exception as exc:
            logger.warning("SPX fetch failed: %s", exc)
            return {}

    _spx_raw = _load_spx(refresh, DEFAULT_START_DATE)
    spx_s = pd.Series(_spx_raw, dtype=float) if _spx_raw else pd.Series(dtype=float)
    if not spx_s.empty:
        spx_s.index = pd.to_datetime(spx_s.index)
        spx_s = spx_s.sort_index().loc[start_ts:end_ts]

    # ── Fed Funds Rate — hardcoded policy-decision anchors (step series) ─
    _fed_rate_anchors = {
        "2018-01-01": 1.50, "2018-06-14": 2.00, "2018-09-27": 2.25, "2018-12-20": 2.50,
        "2019-08-01": 2.25, "2019-09-19": 2.00, "2019-10-31": 1.75,
        "2020-03-04": 1.25, "2020-03-16": 0.25,
        "2022-03-17": 0.50, "2022-05-05": 1.00, "2022-06-16": 1.75,
        "2022-07-28": 2.50, "2022-09-22": 3.25, "2022-11-03": 4.00, "2022-12-15": 4.50,
        "2023-02-02": 4.75, "2023-03-23": 5.00, "2023-05-04": 5.25, "2023-07-27": 5.50,
        "2024-09-19": 5.00, "2024-11-08": 4.75, "2024-12-19": 4.50,
        "2025-01-30": 4.50, "2025-04-01": 4.50,
    }
    _fed_idx = pd.date_range(start_ts, end_ts, freq="B")
    _anch = pd.Series(_fed_rate_anchors)
    _anch.index = pd.to_datetime(_anch.index)
    fedfunds_full = (_anch
                     .reindex(_fed_idx.union(_anch.index))
                     .sort_index().ffill()
                     .loc[start_ts:end_ts])

    # ── Align US yields to business-day grid ─────────────────────────────
    _bdays = pd.date_range(start_ts, end_ts, freq="B")
    us10y_full = us10y_s.reindex(_bdays).ffill() if _us10y_ok else pd.Series(np.nan, index=_bdays)
    us3m_full  = us3m_s.reindex(_bdays).ffill()  if _us3m_ok  else pd.Series(np.nan, index=_bdays)
    spx_full   = spx_s.reindex(_bdays).ffill()   if not spx_s.empty else pd.Series(np.nan, index=_bdays)
    us_yc_spr  = (us10y_full - us3m_full).rename("us10y3m_spread")

    # Inversion spans — cross-panel recession shading
    _inv_mask        = (us_yc_spr < 0).astype(int) if (_us10y_ok and _us3m_ok) else pd.Series(0, index=_bdays)
    _inversion_spans = _spans(_inv_mask, 1)

    fig_yc = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.50, 0.27, 0.23],
        subplot_titles=["S&P 500 (^GSPC)", "US 10Y − 3M Treasury Spread", "Fed Funds Rate (%)"],
    )

    # ── Panel 1: SPX ─────────────────────────────────────────────────────
    # Inversion shading — matches spread and FFR panels exactly
    for s_, e_ in _inversion_spans:
        fig_yc.add_vrect(x0=s_, x1=e_,
            fillcolor=_rgba("#E74C3C", 0.07),
            line_width=0.5, line_color=_rgba("#E74C3C", 0.18),
            layer="below", row=1, col=1)

    _spx_ok = not spx_full.dropna().empty
    if _spx_ok:
        fig_yc.add_trace(go.Scatter(x=spx_full.index, y=spx_full.values,
            mode="lines", name="S&P 500",
            line=dict(color="#c9d1d9", width=1.8),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>SPX: %{y:,.0f}<extra></extra>"),
            row=1, col=1)
        # Drawdown event annotations
        _spx_events = {
            "2020-03-23": ("−35%", "#E74C3C"),
            "2022-10-13": ("−27%", "#E74C3C"),
        }
        for dt_str, (lbl, col_ann) in _spx_events.items():
            _dt = pd.Timestamp(dt_str)
            if start_ts <= _dt <= end_ts:
                _idx = spx_full.index.get_indexer([_dt], method="nearest")[0]
                _yval = float(spx_full.iloc[_idx])
                fig_yc.add_annotation(
                    x=spx_full.index[_idx], y=_yval * 0.985,
                    text=f"▼ {lbl}", showarrow=False,
                    xanchor="center", yanchor="top",
                    font=dict(color=col_ann, size=9,
                              family="JetBrains Mono, monospace"),
                    row=1, col=1)
    else:
        st.warning("⚠️ S&P 500 (^GSPC) unavailable via yfinance.")

    # ── Panel 2: 10Y−3M Spread — teal/red fill ───────────────────────────
    if _us10y_ok and _us3m_ok:
        _spr_pos = us_yc_spr.clip(lower=0)
        _spr_neg = us_yc_spr.clip(upper=0)
        fig_yc.add_trace(go.Scatter(
            x=us_yc_spr.index, y=_spr_pos.values, mode="lines", fill="tozeroy",
            fillcolor=_rgba("#3fb950", 0.35), line=dict(color="#3fb950", width=0.5),
            name="Spread ≥ 0 (normal)", showlegend=True,
            hovertemplate="<b>%{x|%d %b %Y}</b><br>10Y−3M: %{y:+.2f}pp<extra></extra>"),
            row=2, col=1)
        fig_yc.add_trace(go.Scatter(
            x=us_yc_spr.index, y=_spr_neg.values, mode="lines", fill="tozeroy",
            fillcolor=_rgba("#E74C3C", 0.45), line=dict(color="#E74C3C", width=0.5),
            name="Spread < 0 (inverted)", showlegend=True,
            hovertemplate="<b>%{x|%d %b %Y}</b><br>10Y−3M: %{y:+.2f}pp<extra></extra>"),
            row=2, col=1)
        fig_yc.add_trace(go.Scatter(x=us_yc_spr.index, y=us_yc_spr.values,
            mode="lines", name="US 10Y−3M",
            line=dict(color="#e6edf3", width=1.2), showlegend=False),
            row=2, col=1)
        fig_yc.add_hline(y=0, line_color="#484f58", line_width=1.2, line_dash="dot", row=2, col=1)
        _cv_spr = us_yc_spr.dropna()
        if not _cv_spr.empty:
            _cv = float(_cv_spr.iloc[-1])
            fig_yc.add_annotation(
                x=_cv_spr.index[-1], y=_cv, text=f"{_cv:+.2f}pp",
                showarrow=False, xanchor="left",
                font=dict(color="#3fb950" if _cv >= 0 else "#E74C3C", size=11, family="monospace"),
                row=2, col=1)
    elif _us10y_ok:
        fig_yc.add_trace(go.Scatter(x=us10y_full.index, y=us10y_full.values, mode="lines",
            name="US 10Y (3M unavailable)", line=dict(color="#58a6ff", width=1.5)), row=2, col=1)
    else:
        st.warning("⚠️ US Treasury data unavailable. Check yfinance (^TNX, ^IRX).")

    for s_, e_ in _inversion_spans:
        fig_yc.add_vrect(x0=s_, x1=e_,
            fillcolor=_rgba("#E74C3C", 0.06),
            line_width=0.3, line_color=_rgba("#E74C3C", 0.15),
            layer="below", row=2, col=1)

    # ── Panel 3: Fed Funds Rate (step) ────────────────────────────────────
    fig_yc.add_trace(go.Scatter(
        x=fedfunds_full.index, y=fedfunds_full.values,
        mode="lines", name="Fed Funds Rate",
        line=dict(color="#e6edf3", width=2.0, shape="hv"),
        fill="tozeroy", fillcolor=_rgba("#e6edf3", 0.08),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>FFR: %{y:.2f}%<extra></extra>"),
        row=3, col=1)
    _ff_now = float(fedfunds_full.dropna().iloc[-1]) if not fedfunds_full.dropna().empty else np.nan
    if not np.isnan(_ff_now):
        fig_yc.add_annotation(
            x=fedfunds_full.dropna().index[-1], y=_ff_now, text=f"  {_ff_now:.2f}%",
            showarrow=False, xanchor="left",
            font=dict(color="#e6edf3", size=11, family="monospace"), row=3, col=1)

    for dt_str, (rate, label) in [
        ("2019-07-31", (2.25, "Cut")), ("2019-09-19", (2.00, "Cut")), ("2019-10-31", (1.75, "Cut")),
        ("2020-03-04", (1.25, "Cut")), ("2020-03-16", (0.25, "Cut")),
        ("2022-03-17", (0.50, "Hike")),
        ("2023-07-27", (5.50, "Peak")),
        ("2024-09-19", (5.00, "Cut")), ("2024-11-08", (4.75, "Cut")), ("2024-12-19", (4.50, "Cut")),
    ]:
        _dt = pd.Timestamp(dt_str)
        if start_ts <= _dt <= end_ts:
            _fc  = "#2ECC71" if label == "Cut" else "#E74C3C" if label in ("Hike","Peak") else "#d29922"
            _sym = "▼" if label == "Cut" else "▲" if label in ("Hike","Peak") else "●"
            fig_yc.add_annotation(x=_dt, y=rate, text=_sym,
                showarrow=False, font=dict(color=_fc, size=12), row=3, col=1)

    for s_, e_ in _inversion_spans:
        fig_yc.add_vrect(x0=s_, x1=e_,
            fillcolor=_rgba("#E74C3C", 0.06),
            line_width=0.3, line_color=_rgba("#E74C3C", 0.15),
            layer="below", row=3, col=1)

    lo_yc = _layout(height=720)
    lo_yc["margin"] = dict(l=65, r=20, t=55, b=40)
    lo_yc["legend"] = dict(bgcolor="rgba(22,27,34,0.9)", bordercolor=_LINE, borderwidth=1,
                           orientation="h", y=1.02, font=dict(size=10))
    lo_yc["hovermode"] = "x unified"
    fig_yc.update_layout(**lo_yc)
    fig_yc.update_annotations(font_color=_FONT, font_size=11)
    fig_yc.update_yaxes(title_text="S&P 500", tickformat=",.0f",
                        showgrid=True, gridcolor=_GRID, tickfont=dict(size=10), row=1, col=1)
    fig_yc.update_yaxes(title_text="Spread (pp)", showgrid=True, gridcolor=_GRID,
                        zeroline=True, zerolinecolor="#484f58", tickfont=dict(size=10), row=2, col=1)
    fig_yc.update_yaxes(title_text="FFR (%)", ticksuffix="%", showgrid=True,
                        gridcolor=_GRID, tickfont=dict(size=10), row=3, col=1)
    st.plotly_chart(fig_yc, use_container_width=True)
    st.caption(
        "Blue shading in SPX panel = yield curve inversion period (10Y < 3M). "
        "Red fill = inverted spread — leading recession signal. "
        "▲▼ on FFR panel = Fed policy moves. "
        "Data: ^GSPC, ^TNX, ^IRX via yfinance; FFR from policy anchors."
    )

    # ── Two supplemental FRED panels: Foreign Treasury Holdings | Nominal GDP ─
    _SUPP_IDS = ("TRESEGINM052N", "NGDPNSAXDCINQ")

    @st.cache_data(ttl=86_400, show_spinner=False)
    def _load_supp_fred(force):
        import urllib.request
        out = {}
        for sid in _SUPP_IDS:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
            try:
                with urllib.request.urlopen(url, timeout=14) as resp:
                    lines = resp.read().decode().strip().splitlines()
                vals = {}
                for row in lines[1:]:
                    parts = row.split(",")
                    if len(parts) == 2 and parts[1].strip() not in (".", ""):
                        try:
                            vals[parts[0].strip()] = float(parts[1].strip())
                        except ValueError:
                            pass
                out[sid] = vals
            except Exception as exc:
                logger.warning("FRED supp fetch failed for %s: %s", sid, exc)
                out[sid] = {}
        return out

    _supp_raw = _load_supp_fred(refresh)

    def _fred_series(sid):
        raw = _supp_raw.get(sid, {})
        if not raw:
            return pd.Series(dtype=float)
        s = pd.Series(raw, dtype=float)
        s.index = pd.to_datetime(s.index)
        return s.sort_index().loc[start_ts:end_ts]

    _tres_s = _fred_series("TRESEGINM052N")
    _ngdp_s = _fred_series("NGDPNSAXDCINQ")

    col_tres, col_ngdp = st.columns(2)

    # ── Left: Foreign Treasury Holdings (TRESEGINM052N) ──────────────────────
    with col_tres:
        if not _tres_s.empty:
            _tres_chg = _tres_s.diff()
            fig_tres = go.Figure()
            fig_tres.add_trace(go.Bar(
                x=_tres_s.index, y=_tres_chg.values,
                name="MoM Δ",
                marker_color=[_rgba("#E74C3C", 0.75) if v < 0 else _rgba("#3fb950", 0.75)
                               for v in _tres_chg.fillna(0)],
                hovertemplate="<b>%{x|%b %Y}</b><br>MoM Δ: %{y:+,.1f}B<extra></extra>",
            ))
            fig_tres.add_trace(go.Scatter(
                x=_tres_s.index, y=_tres_s.values,
                name="Level ($B)", yaxis="y2", mode="lines",
                line=dict(color="#58a6ff", width=2),
                hovertemplate="<b>%{x|%b %Y}</b><br>Holdings: $%{y:,.0f}B<extra></extra>",
            ))
            _t_last = float(_tres_s.iloc[-1]); _t_dt = _tres_s.index[-1]
            lo_tres = _layout(height=320)
            lo_tres.update(dict(
                margin=dict(l=60, r=60, t=40, b=35),
                title=dict(text="Foreign Holdings of US Treasuries", font=dict(size=12, color=_FONT), x=0.0),
                yaxis=dict(title="MoM Δ ($B)", showgrid=True, gridcolor=_GRID, tickfont=dict(size=10)),
                yaxis2=dict(title="Level ($B)", overlaying="y", side="right",
                            showgrid=False, tickfont=dict(size=10)),
                legend=dict(orientation="h", y=1.04, font=dict(size=10)),
                hovermode="x unified",
                barmode="relative",
            ))
            fig_tres.update_layout(**lo_tres)
            fig_tres.add_annotation(
                x=_t_dt, y=_t_last, text=f"  ${_t_last:,.0f}B",
                yref="y2", showarrow=False, xanchor="left",
                font=dict(color="#58a6ff", size=11, family="monospace"),
            )
            st.plotly_chart(fig_tres, use_container_width=True)
            st.caption("TRESEGINM052N · FRED · Monthly · Foreign & intl accounts holding US Treasuries ($B)")
        else:
            st.warning("⚠️ TRESEGINM052N unavailable — FRED endpoint unreachable.")

    # ── Right: Nominal GDP (NGDPNSAXDCINQ) ───────────────────────────────────
    with col_ngdp:
        if not _ngdp_s.empty:
            _ngdp_yoy = _ngdp_s.pct_change(4) * 100          # QoQ 4-period = YoY
            _ngdp_bar_colors = [
                _rgba("#E74C3C", 0.75) if v < 0 else _rgba("#d29922", 0.75)
                for v in _ngdp_yoy.fillna(0)
            ]
            fig_ngdp = go.Figure()
            fig_ngdp.add_trace(go.Bar(
                x=_ngdp_yoy.index, y=_ngdp_yoy.values,
                name="YoY %", marker_color=_ngdp_bar_colors,
                hovertemplate="<b>%{x|Q%q %Y}</b><br>YoY: %{y:+.1f}%<extra></extra>",
            ))
            fig_ngdp.add_trace(go.Scatter(
                x=_ngdp_s.index, y=_ngdp_s.values,
                name="Level ($B SAAR)", yaxis="y2", mode="lines",
                line=dict(color="#d29922", width=2),
                hovertemplate="<b>%{x|Q%q %Y}</b><br>GDP: $%{y:,.0f}B<extra></extra>",
            ))
            _g_last = float(_ngdp_s.iloc[-1]); _g_dt = _ngdp_s.index[-1]
            lo_ngdp = _layout(height=320)
            lo_ngdp.update(dict(
                margin=dict(l=60, r=60, t=40, b=35),
                title=dict(text="US Nominal GDP (SAAR)", font=dict(size=12, color=_FONT), x=0.0),
                yaxis=dict(title="YoY Growth (%)", showgrid=True, gridcolor=_GRID,
                           ticksuffix="%", tickfont=dict(size=10)),
                yaxis2=dict(title="Level ($B)", overlaying="y", side="right",
                            showgrid=False, tickfont=dict(size=10)),
                legend=dict(orientation="h", y=1.04, font=dict(size=10)),
                hovermode="x unified",
                barmode="relative",
            ))
            fig_ngdp.update_layout(**lo_ngdp)
            fig_ngdp.add_hline(y=0, line_color="#484f58", line_width=1, line_dash="dot")
            fig_ngdp.add_annotation(
                x=_g_dt, y=_g_last, text=f"  ${_g_last/1000:,.1f}T",
                yref="y2", showarrow=False, xanchor="left",
                font=dict(color="#d29922", size=11, family="monospace"),
            )
            st.plotly_chart(fig_ngdp, use_container_width=True)
            st.caption("NGDPNSAXDCINQ · FRED · Quarterly · US Nominal GDP, not seasonally adj., current $B SAAR")
        else:
            st.warning("⚠️ NGDPNSAXDCINQ unavailable — FRED endpoint unreachable.")

# ── US–India Trade & Global EPU ───────────────────────────────────────────────
st.markdown(
    '<div class="section-title">'
    'US–India Trade Flow &nbsp;·&nbsp; Global Economic Policy Uncertainty'
    '</div>',
    unsafe_allow_html=True,
)

@st.cache_data(ttl=86_400, show_spinner=False)
def _load_fred_series(series_ids: tuple, force: bool) -> dict:
    """Fetch multiple FRED series via public CSV endpoint. Returns {id: {date: val}}."""
    import urllib.request
    out = {}
    for sid in series_ids:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
        try:
            with urllib.request.urlopen(url, timeout=12) as resp:
                lines = resp.read().decode().strip().splitlines()
            vals = {}
            for row in lines[1:]:
                parts = row.split(",")
                if len(parts) == 2 and parts[1].strip() not in (".", ""):
                    try:
                        vals[parts[0].strip()] = float(parts[1].strip())
                    except ValueError:
                        pass
            out[sid] = vals
        except Exception as exc:
            logger.warning("FRED fetch failed for %s: %s", sid, exc)
            out[sid] = {}
    return out

_TRADE_IDS  = ("EXP5330", "EXP0015", "IMP5330", "IMP0015")
_EPU_IDS    = ("USEPUINDXM", "CHIEPUINDXM", "EUEPUINDXM", "INDEPUINDXM")

_trade_raw  = _load_fred_series(_TRADE_IDS,  refresh)
_epu_raw    = _load_fred_series(_EPU_IDS,    refresh)

def _to_series(raw: dict, sid: str, s_ts, e_ts) -> "pd.Series":
    d = raw.get(sid, {})
    if not d:
        return pd.Series(dtype=float)
    s = pd.Series(d, dtype=float)
    s.index = pd.to_datetime(s.index)
    return s.sort_index().loc[s_ts:e_ts]

_col_trade, _col_epu = st.columns(2)

# ── Left: US–India Trade ─────────────────────────────────────────────────────
with _col_trade:
    _exp_ind  = _to_series(_trade_raw, "EXP5330", start_ts, end_ts)
    _exp_wld  = _to_series(_trade_raw, "EXP0015", start_ts, end_ts)
    _imp_ind  = _to_series(_trade_raw, "IMP5330", start_ts, end_ts)
    _imp_wld  = _to_series(_trade_raw, "IMP0015", start_ts, end_ts)

    _trade_ok = not (_exp_ind.empty and _imp_ind.empty)
    if _trade_ok:
        # India share of world exports/imports (%)
        _exp_share = (_exp_ind / _exp_wld * 100).dropna() if not (_exp_ind.empty or _exp_wld.empty) else pd.Series(dtype=float)
        _imp_share = (_imp_ind / _imp_wld * 100).dropna() if not (_imp_ind.empty or _imp_wld.empty) else pd.Series(dtype=float)

        fig_tr = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            subplot_titles=[
                "US Exports & Imports to/from India (USD mn)",
                "India Share of Total US Trade (%)",
            ],
        )
        # Absolute flows
        if not _exp_ind.empty:
            fig_tr.add_trace(go.Scatter(
                x=_exp_ind.index, y=_exp_ind.values, mode="lines",
                name="US→India Exports", line=dict(color="#3fb950", width=1.6),
                hovertemplate="<b>%{x|%b %Y}</b><br>Exports: $%{y:,.0f}M<extra></extra>",
            ), row=1, col=1)
        if not _imp_ind.empty:
            fig_tr.add_trace(go.Scatter(
                x=_imp_ind.index, y=_imp_ind.values, mode="lines",
                name="US←India Imports", line=dict(color="#F0A500", width=1.6),
                hovertemplate="<b>%{x|%b %Y}</b><br>Imports: $%{y:,.0f}M<extra></extra>",
            ), row=1, col=1)
        # Trade balance (exports − imports)
        if not _exp_ind.empty and not _imp_ind.empty:
            _bal = (_exp_ind - _imp_ind).reindex(_exp_ind.index.union(_imp_ind.index)).interpolate()
            _bal_clr = ["#3fb950" if v >= 0 else "#E74C3C" for v in _bal.fillna(0)]
            fig_tr.add_trace(go.Bar(
                x=_bal.index, y=_bal.values, name="Trade Balance",
                marker_color=_bal_clr, opacity=0.55,
                hovertemplate="<b>%{x|%b %Y}</b><br>Balance: $%{y:+,.0f}M<extra></extra>",
            ), row=1, col=1)
        # Share lines
        if not _exp_share.empty:
            fig_tr.add_trace(go.Scatter(
                x=_exp_share.index, y=_exp_share.values, mode="lines",
                name="Export share", line=dict(color="#3fb950", width=1.4, dash="dot"),
                hovertemplate="<b>%{x|%b %Y}</b><br>Export share: %{y:.2f}%<extra></extra>",
            ), row=2, col=1)
        if not _imp_share.empty:
            fig_tr.add_trace(go.Scatter(
                x=_imp_share.index, y=_imp_share.values, mode="lines",
                name="Import share", line=dict(color="#F0A500", width=1.4, dash="dot"),
                hovertemplate="<b>%{x|%b %Y}</b><br>Import share: %{y:.2f}%<extra></extra>",
            ), row=2, col=1)

        lo_tr = _layout(height=480)
        lo_tr["margin"] = dict(l=60, r=20, t=50, b=30)
        lo_tr["hovermode"] = "x unified"
        lo_tr["legend"] = dict(bgcolor="rgba(22,27,34,0.85)", bordercolor=_LINE,
                               borderwidth=1, font=dict(size=9), x=0.01, y=0.99)
        fig_tr.update_yaxes(title_text="USD mn", showgrid=True, gridcolor=_GRID,
                            tickformat=",.0f", tickfont=dict(size=9), row=1, col=1)
        fig_tr.update_yaxes(title_text="Share (%)", showgrid=True, gridcolor=_GRID,
                            ticksuffix="%", tickfont=dict(size=9), row=2, col=1)
        fig_tr.update_layout(**lo_tr)
        st.plotly_chart(fig_tr, use_container_width=True)
        st.caption("US Census Bureau / BEA via FRED · EXP5330, IMP5330, EXP0015, IMP0015 · NSA monthly")
    else:
        st.warning("⚠️ US–India trade data unavailable (fred.stlouisfed.org)")

# ── Right: Global EPU ─────────────────────────────────────────────────────────
with _col_epu:
    _EPU_META = {
        "USEPUINDXM":  ("#58a6ff", "US"),
        "CHIEPUINDXM": ("#E74C3C", "China"),
        "EUEPUINDXM":  ("#d29922", "Europe"),
        "INDEPUINDXM": ("#F0A500", "India"),
    }
    _epu_series = {sid: _to_series(_epu_raw, sid, start_ts, end_ts) for sid in _EPU_META}
    _epu_ok = any(not s.empty for s in _epu_series.values())

    if _epu_ok:
        fig_epu = go.Figure()
        for sid, (color, label) in _EPU_META.items():
            s = _epu_series[sid]
            if s.empty:
                continue
            # 3M EWM smoothing for readability
            s_sm = s.ewm(span=3, adjust=False).mean()
            fig_epu.add_trace(go.Scatter(
                x=s_sm.index, y=s_sm.values, mode="lines",
                name=label, line=dict(color=color, width=1.6),
                hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{label} EPU: %{{y:.1f}}<extra></extra>",
            ))
            # Latest annotation
            _lv = float(s_sm.iloc[-1])
            fig_epu.add_annotation(
                x=s_sm.index[-1], y=_lv,
                text=f"  {label} {_lv:.0f}",
                showarrow=False, xanchor="left",
                font=dict(color=color, size=8, family="JetBrains Mono, monospace"),
            )
        # High-uncertainty threshold band (index > 200 = historically elevated)
        fig_epu.add_hrect(
            y0=200, y1=fig_epu.data[0].y.max() * 1.05 if fig_epu.data else 400,
            fillcolor=_rgba("#E74C3C", 0.05),
            line_width=0.5, line_color=_rgba("#E74C3C", 0.25),
            annotation_text="Elevated (>200)", annotation_position="top left",
            annotation_font=dict(color=_rgba("#E74C3C", 0.7), size=8),
        )
        lo_epu = _layout(height=480)
        lo_epu["margin"] = dict(l=55, r=55, t=50, b=30)
        lo_epu["hovermode"] = "x unified"
        lo_epu["legend"] = dict(bgcolor="rgba(22,27,34,0.85)", bordercolor=_LINE,
                                borderwidth=1, font=dict(size=9), x=0.01, y=0.99)
        lo_epu["title"] = dict(
            text="Economic Policy Uncertainty Index (3M EWM)",
            font=dict(color=_FONT, size=11), x=0.03, y=0.97,
        )
        fig_epu.update_yaxes(title_text="EPU Index", showgrid=True,
                             gridcolor=_GRID, tickfont=dict(size=9))
        fig_epu.update_layout(**lo_epu)
        st.plotly_chart(fig_epu, use_container_width=True)
        st.caption("Baker, Bloom & Davis · USEPUINDXM, CHIEPUINDXM, EUEPUINDXM, INDEPUINDXM via FRED · NSA monthly")
    else:
        st.warning("⚠️ EPU data unavailable (fred.stlouisfed.org)")

# ── Volatility Panel ──────────────────────────────────────────────────────────
if show_vol:
    st.markdown('<div class="section-title">Realised Volatility Regime</div>', unsafe_allow_html=True)
    p75 = float(vol_21d.quantile(0.75)); p90 = float(vol_21d.quantile(0.90))
    cvr = float(vol_ratio.dropna().iloc[-1]) if not vol_ratio.dropna().empty else np.nan
    vol_lbl = ("\U0001f534 High Vol" if cur_vol>p90 else "\U0001f7e0 Elevated" if cur_vol>p75 else "\U0001f7e2 Normal") if not np.isnan(cur_vol) else "\u2014"
    vc1,vc2,vc3,vc4 = st.columns(4)
    vc1.metric("21D Realised Vol",  f"{cur_vol:.1f}%"  if not np.isnan(cur_vol) else "\u2014",
               delta=f"{cur_vol-cv63:+.1f}pp vs 63D" if not np.isnan(cv63) else None)
    vc2.metric("63D Realised Vol",  f"{cv63:.1f}%"     if not np.isnan(cv63) else "\u2014")
    vc3.metric("Vol Ratio (21/63)", f"{cvr:.2f}\u00d7" if not np.isnan(cvr) else "\u2014",
               help=">1.0 = vol expanding")
    vc4.metric("Vol Regime",        vol_lbl)
    fig_v = go.Figure()
    fig_v.add_hrect(y0=0,   y1=p75, fillcolor=_rgba("#2ECC71",0.04), line_width=0)
    fig_v.add_hrect(y0=p75, y1=p90, fillcolor=_rgba("#E67E22",0.06), line_width=0)
    fig_v.add_hrect(y0=p90, y1=100, fillcolor=_rgba("#E74C3C",0.09), line_width=0)
    fig_v.add_trace(go.Scatter(x=vol_21d.index, y=vol_21d.values, mode="lines",
        name="21D Vol", line=dict(color="#58a6ff", width=1.8)))
    fig_v.add_trace(go.Scatter(x=vol_63d.index, y=vol_63d.values, mode="lines",
        name="63D Vol", line=dict(color="#d29922", width=1.2, dash="dot")))
    for yv, lbl, col in [(p75,f"p75 {p75:.1f}%","#E67E22"),(p90,f"p90 {p90:.1f}%","#E74C3C")]:
        fig_v.add_hline(y=yv, line_dash="dash", line_color=col, line_width=1,
                        annotation_text=lbl, annotation_font_color=col, annotation_font_size=9)
    lo_v = _layout(height=260)
    lo_v["yaxis"].update(ticksuffix="%", title="Annualised Vol")
    lo_v["margin"] = dict(l=65, r=20, t=10, b=40)
    lo_v["legend"]["orientation"] = "h"; lo_v["legend"]["y"] = 1.08
    fig_v.update_layout(**lo_v)
    st.plotly_chart(fig_v, use_container_width=True)

# ── Drawdown Panel ────────────────────────────────────────────────────────────
if show_drawdown:
    st.markdown('<div class="section-title">Drawdown from Peak</div>', unsafe_allow_html=True)
    eq = price/price.iloc[0]; dd = (eq-eq.cummax())/eq.cummax()*100
    dd1, dd2 = st.columns(2)
    dd1.metric("Current Drawdown",       f"{float(dd.iloc[-1]):.1f}%")
    dd2.metric("Max Drawdown (period)",  f"{float(dd.min()):.1f}%")
    fig_d = go.Figure()
    fig_d.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines",
        fill="tozeroy", fillcolor=_rgba("#f85149",0.15),
        line=dict(color="#f85149", width=1), name="Drawdown",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>DD: %{y:.1f}%<extra></extra>"))
    lo_d = _layout(height=200)
    lo_d["yaxis"].update(ticksuffix="%", title="Drawdown")
    lo_d["showlegend"] = False; lo_d["margin"] = dict(l=65, r=20, t=10, b=40)
    fig_d.update_layout(**lo_d); st.plotly_chart(fig_d, use_container_width=True)

# ── Backtest ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f'<div class="section-title">Backtest \u2014 MA({ma_window}) vs Regime-Filtered (v5 Hard Gate)</div>', unsafe_allow_html=True)

# Base signal: price > MA(N), shifted +1D inside compute_price_above_ma
signal_naive = compute_price_above_ma(price, window=ma_window)

# Regime gate: HARD gate based on dominant HMM state, shifted +1D
# States 0,1 (Bull) = full position, 2 (Neutral) = 0.75×, 3 (Stress) = 0, 4 (Panic) = 0
# This makes a visible difference vs naive signal (soft multiplier was ≈1 in bull markets)
if hmm_ok:
    # _hard_gate_shifted may already be computed in the Kelly panel above — reuse it
    if "_hard_gate_shifted" not in dir():
        _state_aln         = hmm_state_s.reindex(price.index).ffill()
        _hard_gate         = _state_aln.map({0:1.0, 1:1.0, 2:0.75, 3:0.0, 4:0.0}).fillna(1.0)
        _hard_gate_shifted = _hard_gate.shift(1).fillna(1.0)
    signal_full = (signal_naive * _hard_gate_shifted).clip(0.0, 1.0)
else:
    signal_full = signal_naive.copy()

bt_naive = run_backtest(price, signal_naive, cost_bps=tx_cost*10_000, name=f"MA({ma_window}) Only")
bt_full  = run_backtest(price, signal_full,  cost_bps=tx_cost*10_000, name="+ Regime Gate (v5)")
bt_bh    = run_backtest(price, pd.Series(1.0, index=price.index).shift(1).fillna(0.0),
                        cost_bps=0.0, name="Buy & Hold")

_mk = ["Total Return","CAGR","Sharpe","Sortino","Max DD","Calmar","Win Rate"]
perf_df = pd.DataFrame({"Metric":_mk,
    bt_naive.name:[bt_naive.metrics.get(k,"\u2014") for k in _mk],
    bt_full.name: [bt_full.metrics.get(k,"\u2014")  for k in _mk],
    bt_bh.name:   [bt_bh.metrics.get(k,"\u2014")    for k in _mk]})

col_bt1, col_bt2 = st.columns([2,3])
with col_bt1:
    st.dataframe(perf_df, hide_index=True, use_container_width=True)
    st.caption(
        f"**Hard gate (v5):** Bull 1.0\u00d7, Neutral 0.75\u00d7, Stress 0\u00d7, Panic 0\u00d7.  \n"
        f"Tx cost {tx_cost*10_000:.0f}bps/side. All signals pre-shifted +1D (no look-ahead)."
    )
with col_bt2:
    # ── Equity curves + divergence fill + drawdown subplot ────────────────
    fig_bt = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.04, row_heights=[0.68, 0.32],
        subplot_titles=["Equity (normalised to 1.0×)", "Drawdown — Regime Gate vs MA Only"],
    )
    # Buy & Hold (grey, thin)
    fig_bt.add_trace(go.Scatter(x=bt_bh.equity_curve.index, y=bt_bh.equity_curve.values,
        mode="lines", name=bt_bh.name, line=dict(color="#6e7681", width=1.0, dash="dot"),
        hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{bt_bh.name}: %{{y:.3f}}×<extra></extra>"),
        row=1, col=1)
    # MA Only (blue)
    fig_bt.add_trace(go.Scatter(x=bt_naive.equity_curve.index, y=bt_naive.equity_curve.values,
        mode="lines", name=bt_naive.name, line=dict(color="#58a6ff", width=1.8),
        hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{bt_naive.name}: %{{y:.3f}}×<extra></extra>"),
        row=1, col=1)
    # Regime Filtered — bright green, thicker, dashed for distinction
    fig_bt.add_trace(go.Scatter(x=bt_full.equity_curve.index, y=bt_full.equity_curve.values,
        mode="lines", name=bt_full.name,
        line=dict(color="#2ECC71", width=2.5, dash="solid"),
        hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{bt_full.name}: %{{y:.3f}}×<extra></extra>"),
        row=1, col=1)
    # Divergence fill: shade area between regime-filtered and MA-only
    _eq_full_aln  = bt_full.equity_curve.reindex(bt_naive.equity_curve.index).ffill()
    _eq_naive_aln = bt_naive.equity_curve
    _div_up   = _eq_full_aln.clip(lower=_eq_naive_aln)
    _div_down = _eq_full_aln.clip(upper=_eq_naive_aln)
    # Regime > MA = green fill (regime is protecting / outperforming)
    fig_bt.add_trace(go.Scatter(x=_div_up.index.tolist() + _eq_naive_aln.index[::-1].tolist(),
        y=_div_up.values.tolist() + _eq_naive_aln.values[::-1].tolist(),
        fill="toself", fillcolor=_rgba("#2ECC71", 0.12),
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"), row=1, col=1)
    # Regime < MA = red fill (regime is lagging / missing upside)
    fig_bt.add_trace(go.Scatter(x=_div_down.index.tolist() + _eq_naive_aln.index[::-1].tolist(),
        y=_div_down.values.tolist() + _eq_naive_aln.values[::-1].tolist(),
        fill="toself", fillcolor=_rgba("#E74C3C", 0.10),
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"), row=1, col=1)
    # Drawdown panel
    def _dd(ec):
        return ((ec - ec.cummax()) / ec.cummax() * 100)
    fig_bt.add_trace(go.Scatter(x=bt_naive.equity_curve.index,
        y=_dd(bt_naive.equity_curve).values,
        mode="lines", name=f"DD {bt_naive.name}", line=dict(color="#58a6ff", width=1.2),
        fill="tozeroy", fillcolor=_rgba("#58a6ff", 0.08), showlegend=False,
        hovertemplate="<b>%{x|%d %b %Y}</b><br>DD: %{y:.1f}%<extra></extra>"), row=2, col=1)
    fig_bt.add_trace(go.Scatter(x=bt_full.equity_curve.index,
        y=_dd(bt_full.equity_curve).values,
        mode="lines", name=f"DD {bt_full.name}", line=dict(color="#2ECC71", width=1.5),
        fill="tozeroy", fillcolor=_rgba("#2ECC71", 0.08), showlegend=False,
        hovertemplate="<b>%{x|%d %b %Y}</b><br>DD: %{y:.1f}%<extra></extra>"), row=2, col=1)

    lo_bt = _layout(height=440)
    lo_bt["margin"] = dict(l=65, r=20, t=35, b=40)
    lo_bt["legend"] = dict(bgcolor="rgba(22,27,34,0.9)", bordercolor=_LINE, borderwidth=1,
                           orientation="h", y=1.06, font=dict(size=11))
    lo_bt["hovermode"] = "x unified"
    fig_bt.update_layout(**lo_bt)
    fig_bt.update_yaxes(title_text="Equity (×)", tickformat=".2f",
                        showgrid=True, gridcolor=_GRID, tickfont=dict(size=10), row=1, col=1)
    fig_bt.update_yaxes(title_text="Drawdown (%)", ticksuffix="%",
                        showgrid=True, gridcolor=_GRID, tickfont=dict(size=10), row=2, col=1)
    st.plotly_chart(fig_bt, use_container_width=True)

# ── Regime Attribution ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">Regime Attribution & Statistics</div>', unsafe_allow_html=True)

if hmm_ok:
    dr = price.pct_change().dropna(); rows = []
    for rid, rinfo in REGIMES.items():
        mask  = hmm_state_s == rid; days = int(mask.sum())
        pct   = days/max(len(hmm_state_s),1)*100
        r_sub = dr.reindex(hmm_state_s[mask].index).dropna()
        ann   = ((1+r_sub).prod()**(252/max(len(r_sub),1))-1) if len(r_sub)>1 else np.nan
        sr    = r_sub.mean()/r_sub.std()*np.sqrt(252) if r_sub.std()>0 else np.nan
        avg_d = r_sub.mean()*100 if len(r_sub)>0 else np.nan
        mdd   = float(((1+r_sub).cumprod()/(1+r_sub).cumprod().cummax()-1).min()*100) if len(r_sub)>1 else np.nan
        rows.append({"Regime":f"{rinfo['emoji']} {rinfo['name']}","Days":days,
                     "% Time":f"{pct:.1f}%","Ann. Return":_fmt(ann,".1%"),
                     "Sharpe":_fmt(sr,".2f"),"Avg Day":f"{_fmt(avg_d,'.3f')}%",
                     "Max DD":f"{_fmt(mdd,'.1f')}%","Gate":f"{rinfo['mult']:.2f}\u00d7"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with st.expander("\U0001f4ca Regime Transition Matrix"):
        _A = np.array(hmm_raw["trans_matrix"])
        fig_tm = go.Figure(go.Heatmap(
            z=_A, x=REGIME_NAMES, y=REGIME_NAMES,
            colorscale=[[0,"#0d1117"],[0.3,"#1f3d6e"],[1.0,"#58a6ff"]],
            zmin=0, zmax=1,
            text=[[f"{_A[r,c]:.2f}" for c in range(N_REGIMES)] for r in range(N_REGIMES)],
            texttemplate="%{text}", textfont=dict(size=11, color="white"),
            colorbar=dict(tickfont=dict(color=_FONT,size=9),
                          title=dict(text="P(i\u2192j)", font=dict(color=_FONT,size=10))),
            hovertemplate="<b>%{y} \u2192 %{x}</b><br>P = %{z:.2f}<extra></extra>"))
        lo_tm = _layout(height=320)
        lo_tm["margin"] = dict(l=120, r=10, t=10, b=90)
        lo_tm["xaxis"]["tickangle"] = -30
        fig_tm.update_layout(**lo_tm); st.plotly_chart(fig_tm, use_container_width=True)
        st.caption("Row i = current state \u2192 column j = next state. Each row sums to 1.0.")
else:
    st.info("Regime attribution requires hmmlearn. Showing DMA-based split.")
    p_on = float(p_regime.mean()*100); dr = price.pct_change().dropna()
    on_r = dr.reindex(p_regime[p_regime==1].index).dropna()
    off_r= dr.reindex(p_regime[p_regime==0].index).dropna()
    def _a2(r): return f"{((1+r).prod()**(252/max(len(r),1))-1):.1%}" if len(r)>1 else "\u2014"
    def _s2(r): return f"{r.mean()/r.std()*np.sqrt(252):.2f}" if r.std()>0 else "\u2014"
    st.dataframe(pd.DataFrame({"Regime":["\U0001f7e2 Risk-ON (above DMA)","\U0001f534 Risk-OFF (below DMA)"],
        "% Time":[f"{p_on:.1f}%",f"{100-p_on:.1f}%"],
        "Ann. Return":[_a2(on_r),_a2(off_r)],"Sharpe":[_s2(on_r),_s2(off_r)]}),
        hide_index=True, use_container_width=True)

# ── Signal Preview ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">Live Signal Preview</div>', unsafe_allow_html=True)
_sk_now = cur_sk if hmm_ok else 1.0
_gate = _hard_gate_shifted.iloc[-1] if (hmm_ok and '_hard_gate_shifted' in dir()) else 1.0
c1,c2,c3,c4 = st.columns(4)
with c1: rs = st.slider("Raw Signal", -1.0, 1.0, 0.7, 0.05)
adj = float(np.clip(rs * _gate, -cur_cap, cur_cap))
c2.metric("Soft Kelly",     f"{_sk_now:.3f}\u00d7", help="Continuous weighted multiplier")
c3.metric("Hard Gate",      f"{_gate:.2f}\u00d7",   help="Binary gate: Bull=1.0, Neutral=0.75, Stress/Panic=0.0")
c4.metric("Gated Signal",   f"{adj:.3f}", delta=f"\u0394{adj-rs:+.3f}")
st.markdown(f"**Hard gate applied:** `{_gate:.2f}` \u2192 signal `{rs:.2f}` \u2192 `{adj:.3f}`")

# ── Downloads ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">\U0001f4e5 Data Exports</div>', unsafe_allow_html=True)
dl1,dl2,dl3 = st.columns(3)
with dl1:
    sig_tbl = pd.DataFrame({"nifty_close":price.values,
        "ma_signal":signal_naive.reindex(price.index).values,
        "regime_signal":signal_full.reindex(price.index).values},
        index=price.index)
    if hmm_ok:
        sig_tbl["hmm_state"]  = hmm_state_s.reindex(price.index).ffill()
        sig_tbl["hmm_regime"] = sig_tbl["hmm_state"].map({k:v["name"] for k,v in REGIMES.items()})
        sig_tbl["hard_gate"]  = _hard_gate_shifted.reindex(price.index).values
        sig_tbl["soft_kelly"] = soft_kelly_s.reindex(price.index).ffill().shift(1).values
    if not bond_s.empty:   sig_tbl["bond_yield_10y"] = bond_s.reindex(price.index).ffill()
    if not repo_s.empty:   sig_tbl["rbi_repo_rate"]  = repo_s.reindex(price.index).ffill()
    if not us10y_s.empty:  sig_tbl["us_10y_yield"]   = us10y_s.reindex(price.index).ffill()
    if not us3m_s.empty:   sig_tbl["us_3m_yield"]    = us3m_s.reindex(price.index).ffill()
    st.download_button("\u2b07 Signal Table (CSV)", sig_tbl.to_csv(),
                       file_name="nautilus_signals.csv", mime="text/csv", use_container_width=True)
with dl2:
    st.download_button("\u2b07 Backtest Summary (CSV)", perf_df.to_csv(index=False),
                       file_name="nautilus_backtest.csv", mime="text/csv", use_container_width=True)
with dl3:
    eq_df = pd.DataFrame({"Buy & Hold":bt_bh.equity_curve,
        f"MA({ma_window})":bt_naive.equity_curve,"+ Regime Gate":bt_full.equity_curve})
    st.download_button("\u2b07 Equity Curves (CSV)", eq_df.to_csv(),
                       file_name="nautilus_equity.csv", mime="text/csv", use_container_width=True)

# ── Methodology ───────────────────────────────────────────────────────────────
feat_str = ", ".join(hmm_raw.get("feature_names",["\u2014"])) if hmm_ok else "\u2014"
with st.expander("\u2139\ufe0f Methodology \u2014 Model Design, Data Sources & No Look-Ahead"):
    st.markdown(f"""
### v5 Architecture \u2014 Hard Regime Gates

```
position(t) = base_signal(t) \u00d7 hard_gate(t-1)

hard_gate(t) = argmax P(state | data_{{1..t}}) \u2192 mult
  \U0001f7e2 Bull Quiet    = 1.00   \U0001f535 Bull Volatile = 1.00
  \U0001f7e1 Neutral       = 0.75
  \U0001f7e0 Stress        = 0.00   \U0001f534 Panic         = 0.00
```

The v5 hard gate produces a **measurable difference** vs the base MA signal by
zeroing positions entirely in Stress/Panic regimes. Soft Kelly (continuous weighted
multiplier) is shown for reference but the backtest uses the hard gate.

### HMM Features (`{feat_str}`)
| Group | Variables |
|---|---|
| Price/Vol | 21D vol, 21D ret, vol ratio 21/63, vol-of-vol, 5D ret, 252D drawdown |
| Macro | 10Y yield \u039421D, yield spread, RBI easing flag, 200-DMA ratio |

### Data Sources
| Series | Source | Coverage |
|---|---|---|
| Nifty 50 | yfinance `^NSEI` | Live |
| RBI Repo Rate | `data/rbi_repo_rate.csv` (RBI) | 2012\u2192present |
| India 10Y G-Sec | `data/india_10y_yield.csv` (Investing.com) | 2018\u2192present |
| US 10Y Treasury | yfinance `^TNX` | Live |
| US 3M T-Bill    | yfinance `^IRX` | Live |

### No Look-Ahead
All signals and gates shifted +1D. Backtest engine takes pre-shifted signals.
""")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:#484f58;font-size:0.78rem'>"
    f"Nautilus Research v{__version__} &nbsp;\u00b7&nbsp; "
    "Visual Research Tool \u2014 Not Investment Advice &nbsp;\u00b7&nbsp; "
    "Data: yfinance \u00b7 RBI \u00b7 Investing.com</div>",
    unsafe_allow_html=True,
)
