# Nautilus

**India Macro-Momentum Regime System** — 5-state Gaussian HMM · regime-gated backtests · full macro overlay

Nautilus is a state-space modeling framework that applies a 5-state Gaussian Hidden Markov Model to infer latent market regimes from price, volatility, and macro data. These regimes drive a rule-based execution layer with gated exposure and adaptive sizing, enabling robust, forward-consistent strategy behavior.

![PyPI version](https://img.shields.io/pypi/v/nautilus-spark)
![Python](https://img.shields.io/pypi/pyversions/nautilus-spark)
![License](https://img.shields.io/pypi/l/nautilus-spark)

---

## Use Cases

### Regime-Aware Strategy Development
Fit a 5-state HMM to Nifty 50 and apply the resulting Hard Gate to any base signal.
The gate zeros positions in Stress and Panic regimes while preserving full exposure in Bull states.

```python
from nautilus.etl.loader import load_index
from nautilus.etl.macro import build_macro_features
from nautilus.strategies.regime import fit_hmm, REGIMES
from nautilus.strategies.momentum import compute_price_above_ma
from nautilus.backtests.engine import run_backtest

nifty      = load_index(start="2018-01-01")
price      = nifty["Close"]
macro_raw  = build_macro_features(price, raw=True)   # unshifted — for HMM input

result     = fit_hmm(price, macro_df=macro_raw, n_states=5, n_iter=200)
gate       = result.soft_kelly                        # continuous [0, 1] multiplier

ma_signal  = compute_price_above_ma(price, window=45) # pre-shifted +1D
gated_sig  = (ma_signal * gate.reindex(price.index).ffill()).clip(0, 1)

bt = run_backtest(price, gated_sig, cost_bps=10, name="MA(45) + HMM Gate")
print(bt.metrics["Sharpe"], bt.metrics["CAGR"], bt.metrics["Max DD"])
```

---

### Current Regime Snapshot
Read the live regime, confidence, and Kelly multiplier in three lines.

```python
result   = fit_hmm(price, macro_df=macro_raw)
state    = int(result.states[-1])
conf     = float(result.posteriors[-1].max())
kelly    = float(result.soft_kelly[-1])

print(f"{REGIMES[state]['name']}  {conf:.0%} conf  {kelly:.2f}x Kelly")
# Bull Quiet  94% conf  1.00x Kelly
```

---

### Forward Regime Probability Forecast
Project the Markov chain N days forward from the current posterior to get a
probability distribution over regimes at any future horizon.

```python
from nautilus.strategies.regime import markov_forecast, REGIME_NAMES

paths = markov_forecast(result.trans_matrix, result.posteriors[-1], horizon=20)
print("Regime probabilities at t+20:")
for name, prob in zip(REGIME_NAMES, paths[-1]):
    print(f"  {name:<16} {prob:.1%}")
```

---

### Portfolio Risk Overlay
Scale a multi-asset portfolio's positions by the regime gate in one vectorised step.

```python
import pandas as pd

# portfolio_returns: DataFrame with one column per asset
gate_series = (
    pd.Series(result.soft_kelly, index=result.dates)
    .reindex(portfolio_returns.index).ffill().fillna(0)
)
portfolio_gated = portfolio_returns.mul(gate_series, axis=0)
```

---

### Cross-Sectional Regime Factory
Fit independent HMMs across an entire universe and build a regime dashboard
for every stock in one loop.

```python
from nautilus.etl.loader import load_index
from nautilus.strategies.regime import fit_hmm, REGIMES

universe = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
snapshot = {}

for ticker in universe:
    df  = load_index(ticker=ticker, start="2021-01-01")
    res = fit_hmm(df["Close"], n_states=5, n_iter=100)
    if res is None:
        continue
    cur = int(res.states[-1])
    snapshot[ticker] = {
        "regime":  REGIMES[cur]["name"],
        "kelly":   round(float(res.soft_kelly[-1]), 3),
        "conf":    round(float(res.posteriors[-1].max()), 3),
    }

import pandas as pd
print(pd.DataFrame(snapshot).T.sort_values("kelly", ascending=False))
```

---

### Walk-Forward Out-of-Sample Validation

```python
from sklearn.preprocessing import RobustScaler

TRAIN_YEARS, OOS_DAYS = 3, 126
results = []

for oos_start in pd.date_range(price.index[int(TRAIN_YEARS*252)], price.index[-OOS_DAYS-1], freq="126D"):
    train = price.loc[:oos_start]
    oos   = price.loc[oos_start: oos_start + pd.Timedelta(days=200)]
    if len(train) < 300 or len(oos) < 30:
        continue

    from nautilus.strategies.regime import build_hmm_features
    res     = fit_hmm(train, n_states=5, n_iter=100)
    feat    = build_hmm_features(oos)
    scaler  = RobustScaler().fit(build_hmm_features(train).values)
    _, post = res.model.score_samples(scaler.transform(feat.values))
    from nautilus.strategies.regime import MULT_VEC
    results.append({"oos_start": oos_start.date(), "mean_kelly": float((post @ MULT_VEC).mean())})

print(pd.DataFrame(results))
```

---

### Regime Attribution + Event Study
Decompose Nifty returns by HMM regime and study return windows around RBI policy dates.

```python
import numpy as np

log_ret   = np.log(price / price.shift(1))
state_aln = pd.Series(result.states, index=result.dates).reindex(price.index).ffill()

attribution = (
    pd.DataFrame({"return": log_ret, "regime": state_aln})
    .dropna()
    .groupby("regime")["return"]
    .agg(["mean", "std", "count"])
)
attribution["ann_sharpe"] = attribution["mean"] / attribution["std"] * np.sqrt(252)
attribution.index = [REGIMES[i]["name"] for i in attribution.index]
print(attribution.round(4))
```

---

### FastAPI / Lambda Deployment
Wrap the regime engine in a REST endpoint or serverless handler.

```python
# FastAPI
from fastapi import FastAPI
app = FastAPI()

@app.get("/regime")
def get_regime():
    nifty  = load_index(start="2018-01-01")
    result = fit_hmm(nifty["Close"], n_states=5, n_iter=200)
    state  = int(result.states[-1])
    return {
        "regime":     REGIMES[state]["name"],
        "confidence": round(float(result.posteriors[-1].max()), 4),
        "soft_kelly": round(float(result.soft_kelly[-1]), 4),
        "gate_open":  state < 4,
    }

# AWS Lambda
def lambda_handler(event, context):
    import json
    return {"statusCode": 200, "body": json.dumps(get_regime())}
```

---

### Automated Daily Report

```python
forecast = markov_forecast(result.trans_matrix, result.posteriors[-1], horizon=20)
top_t20  = int(forecast[-1].argmax())

report = (
    f"=== NAUTILUS DAILY REPORT — {pd.Timestamp.today().date()} ===\n"
    f"Regime   : {REGIMES[state]['name']}  ({conf:.0%} conf)\n"
    f"Soft Kelly: {kelly:.2f}x  |  Gate: {'OPEN' if state < 4 else 'CLOSED'}\n"
    f"Nifty 50 : {float(price.iloc[-1]):,.0f}  |  1D: {float(price.pct_change().iloc[-1]*100):+.2f}%\n"
    f"T+20 view: {REGIMES[top_t20]['name']}  ({forecast[-1].max():.0%})"
)
print(report)
# pipe to telegram.send(report) / email / Slack
```

---

## Dashboard

```bash
nautilus
# or
streamlit run src/nautilus/dashboard/regime_dashboard.py
```

Opens at **http://localhost:8501**. The dashboard includes:

- Nifty 50 price chart with regime shading and Soft Kelly overlay
- 5-state probability stack + 20-day Markov forecast
- Regime-filtered backtest vs plain MA with 18-metric comparison table
- RBI repo rate, India 10Y G-Sec yield, and yield spread panel
- US macro monitor — S&P 500, 10Y−3M Treasury spread, Fed Funds Rate
- BRICS FX normalised to 100, US–India trade flows, Global EPU index
- Regime attribution table and transition matrix heatmap
- Monthly returns heatmap and full CSV export

---

## Model architecture

```
position(t) = base_signal(t) × hard_gate(t−1)

hard_gate(t) = f( argmax P(state | data₁..ₜ) )
  Bull Quiet    → 1.00×   (full exposure)
  Bull Volatile → 0.85×   (mild haircut)
  Neutral       → 0.75×   (moderate reduction)
  Stress        → 0.00×   (flat — no longs)
  Panic         → 0.00×   (cash)
```

HMM input features: `vol_21d`, `ret_21d`, `vol_ratio` (21/63), `vol_of_vol`, `ret_5d`,
`drawdown`, `bond_yield_chg_21d`, `yield_spread`, `repo_easing`, `dma_200_ratio`.

All signals are pre-shifted +1D before entering the backtest engine. No look-ahead.

---

## Data sources

| Series | Source | Cadence |
|---|---|---|
| Nifty 50 | `^NSEI` via yfinance + Parquet cache | Daily |
| India 10Y G-Sec | Bundled `india_10y_yield.csv` | Bundled (2018–present) |
| RBI Repo Rate | Bundled `rbi_repo_rate.csv` + optional live scrape | Bundled |
| US 10Y Treasury | `^TNX` via yfinance | Daily |
| US 3M T-Bill | `^IRX` via yfinance | Daily |
| S&P 500 | `^GSPC` via yfinance | Daily |
| Fed Funds Rate | Hardcoded policy anchors | Per FOMC |

---

## Backtest metrics

18 metrics across four groups: **Returns** (Total Return, CAGR, Ann. Vol),
**Risk-Adjusted** (Sharpe, Sortino, Calmar, Profit Factor), **Drawdown** (Max DD,
Max DD Days, Longest DD Days, Avg DD, DD Episodes, Recovery Days, Current DD),
**Trade** (Win Rate, Avg Win, Avg Loss).

---

## Project layout

```
src/nautilus/
├── config.py                # paths, tickers, model defaults
├── cli.py                   # nautilus command entry point
├── data/                    # bundled CSVs (shipped with wheel)
│   ├── india_10y_yield.csv
│   └── rbi_repo_rate.csv
├── etl/
│   ├── loader.py            # yfinance download + Parquet cache
│   └── macro.py             # macro pipeline (raw= flag for HMM)
├── strategies/
│   ├── momentum.py          # MA signal, price-above-MA regime
│   └── regime.py            # HMM fitting, Markov forecast
├── backtests/
│   └── engine.py            # vectorised engine + 18-metric suite
└── dashboard/
    └── regime_dashboard.py  # Streamlit app
```

---

## Honest limitations

- **In-sample only.** The HMM is fit and evaluated on the same history. Reported Sharpe
  and CAGR are descriptive, not predictive. Use walk-forward validation before drawing
  conclusions.
- **State identity can shift.** Regime labels are assigned post-hoc by sorting on
  `vol − return`. State identities can change between re-fits; anchoring logic is not
  yet implemented.
- **Sparse macro features.** The repo rate changes ~4–6 times per year. Forward-filling
  creates ~250 identical rows between decisions.
- **Hard gate multipliers are hand-tuned.** They are not derived from a risk model.
- **No statistical significance.** With ~2,700 days the standard error on annualised
  Sharpe is ≈ 0.35. A 0.3 improvement is within noise.
- **Research tool, not a trading system.** No order routing, position reconciliation,
  or risk limit enforcement.

---

## License

MIT — see [LICENSE](LICENSE).
