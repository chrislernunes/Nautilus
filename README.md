# NAUTILUS

**India Macro-Regime Platform** — 5-state Gaussian HMM · US macro overlay 

---

## What it does

Nautilus ingests Nifty 50 prices alongside Indian and US macro data, fits a 5-state Hidden Markov Model on price volatility and macro features, and renders a full-stack research dashboard for regime-aware strategy research.

**Core capabilities:**

- Fits a **5-state Gaussian HMM** on 21D vol, returns, vol-of-vol, drawdown, 10Y yield Δ, yield spread, RBI easing flag, and 200-DMA ratio
- Applies a **v5 Hard Regime Gate** — binary position scaling based on dominant HMM state, shifted +1D (no look-ahead)
- **Sharpened Soft Kelly** — EWM(5) exponential weighting for ~3-day signal response vs the old 21-day rolling lag
- **US Macro Monitor** — S&P 500 · 10Y−3M Treasury spread · Fed Funds Rate with aligned inversion-period shading across all three panels
- **Backtest engine** — MA(N) vs regime-filtered equity curves with divergence fill and drawdown comparison subplot
- **Regime attribution** — days, % time, ann. return, Sharpe, max DD per HMM state
- **Markov forward forecast** — N-day regime probability paths from current posterior
- **Monthly returns heatmap** and full CSV export

---

## Quickstart

```bash
# 1. Clone / extract
cd nautilus_v6

# 2. Create virtual environment (recommended)
python -m venv .venv

# Windows MSYS2 / PowerShell:
source .venv/Scripts/activate
# Linux / macOS:
source .venv/bin/activate

# 3. Install package (editable)
pip install -e ".[dev]"
pip install hmmlearn        # required for 5-state HMM

# 4. Run dashboard
streamlit run src/nautilus/dashboard/regime_dashboard.py
```

Dashboard opens at **http://localhost:8501**

---

## Dashboard panels

| Panel | Description |
|---|---|
| **KPI row** | Nifty 50 · DMA spread · HMM regime · Soft Kelly · 10Y G-Sec · RBI Repo · US 10Y−3M spread |
| **Regime distribution bar** | Colour-coded time allocation across 5 HMM states |
| **Nifty 50 · Regime Shading** | Price chart with HMM state bands + sharpened EWM(5) Kelly overlay |
| **Regime Probability Stack** | Stacked area chart of posterior probabilities per state |
| **Regime Forward Forecast** | Markov N-day probability paths from current state |
| **Soft Kelly — Sharpened** | EWM(5) vs MA(21) Kelly with hard gate overlay on secondary axis |
| **Monthly Returns Heatmap** | Year × month return grid |
| **Macro Panel** | India 10Y G-Sec yield + RBI repo rate + term spread |
| **US Macro Monitor** | S&P 500 · US 10Y−3M spread · Fed Funds Rate — inversion shading aligned across all three panels |
| **Volatility Panel** | 21D and 63D realised vol with p75/p90 regime bands |
| **Drawdown Panel** | Rolling drawdown from peak (optional toggle) |
| **Backtest** | MA(N) vs Regime-Filtered equity curves with divergence fill + drawdown subplot |
| **Regime Attribution** | Per-state statistics table + Markov transition matrix heatmap |
| **Signal Preview** | Live hard gate · soft Kelly · gated signal interactive calculator |
| **Data Exports** | Signal table · backtest summary · equity curves (CSV) |

---

## v5 Hard Gate logic

```
position(t) = base_signal(t) × hard_gate(t−1)

hard_gate(t) = f(argmax P(state | data₁..ₜ))
  🟢 Bull Quiet    → 1.00×   (full exposure)
  🔵 Bull Volatile → 1.00×   (still directionally long)
  🟡 Neutral       → 0.75×   (mild reduction)
  🟠 Stress        → 0.00×   (flat)
  🔴 Panic         → 0.00×   (cash)
```

All signals are pre-shifted +1D before entering the backtest engine. No look-ahead bias.

The hard gate produces a **measurable difference** vs a plain MA signal by zeroing positions entirely in Stress/Panic regimes — typically the 15–25% of time when drawdowns are largest.

---

## Sharpened Kelly

The dashboard exposes two Kelly variants:

| Variant | Lag | Use |
|---|---|---|
| **EWM(5)** | ~3 days | Primary signal — fast regime response |
| **MA(21)** | ~10 days | Reference — shown dotted in history panel |
| **Hard Gate** | 1 day (shift) | Binary 0/0.75/1.0 from dominant HMM state |

EWM(5) is computed as `soft_kelly.ewm(span=5, min_periods=3).mean()` and is also overlaid on the main Nifty price chart.

---

## US Macro Monitor

Three shared-x panels with **inversion-period shading aligned across all panels**:

1. **S&P 500** (`^GSPC`) — red vrect shading during yield curve inversion, drawdown annotations
2. **US 10Y−3M spread** (`^TNX − ^IRX`) — teal fill (normal) / red fill (inverted) with zero line
3. **Fed Funds Rate** — step-function from hardcoded policy anchors, ▲▼ markers at each FOMC decision

Inversion is defined as `US 10Y < 3M T-Bill`. Historically a leading recession indicator with 12–18 month lead time.

Fed Funds Rate uses hardcoded policy decision anchors forward-filled on business days — reliable even when `^FED` is unavailable via yfinance.

---

## Data sources

| Series | Ticker / Source | Cadence |
|---|---|---|
| Nifty 50 | `^NSEI` via yfinance | Daily |
| India 10Y G-Sec | `data/india_10y_yield.csv` | Bundled |
| RBI Repo Rate | `data/rbi_repo_rate.csv` | Bundled |
| US 10Y Treasury | `^TNX` via yfinance | Daily |
| US 3M T-Bill | `^IRX` via yfinance | Daily |
| S&P 500 | `^GSPC` via yfinance | Daily |
| Fed Funds Rate | Hardcoded policy anchors | Per FOMC meeting |


---

## Preview

![Nautilus Dashboard](nautilus-preview.png)

---

## Project structure

```
nautilus_v6/
├── data/
│   ├── rbi_repo_rate.csv        # RBI repo rate history (public domain)
│   └── india_10y_yield.csv      # India 10Y G-Sec yield history
├── src/nautilus/
│   ├── __init__.py              # Version: 0.5.0
│   ├── config.py                # Paths, tickers, model defaults
│   ├── etl/
│   │   ├── loader.py            # yfinance download + Parquet cache
│   │   └── macro.py             # Macro feature pipeline (RBI, bond, HMM features)
│   ├── strategies/
│   │   ├── momentum.py          # MA signal, price-above-MA regime
│   │   └── regime.py            # HMM fitting, Markov forecast, regime containers
│   ├── backtests/
│   │   └── engine.py            # Vectorised backtest, metrics, equity curves
│   └── dashboard/
│       └── regime_dashboard.py  # Streamlit app (main entry point)
├── tests/
│   └── test_engine.py           # Backtest engine unit tests
└── pyproject.toml
```

---

## Updating data

**RBI repo rate** — after each MPC meeting, append a new row to `data/rbi_repo_rate.csv`:

```
2025-06-06,5.50
```

Then click **🔄 Refresh Data** in the sidebar to bust the Streamlit cache.

**India 10Y yield** — replace `data/india_10y_yield.csv` with an updated export from Investing.com (same column format: `Date,Price`).

---

## Sidebar controls

| Control | Default | Effect |
|---|---|---|
| Date Range | 2018-01-01 → today | Slices all data and charts |
| DMA Window | 200 days | Long-term trend filter line on price chart |
| HMM Iterations | 200 | EM convergence steps (higher = slower, more stable) |
| Forecast Horizon | 20 days | Markov forward path length |
| Macro features in HMM | On | Adds yield/repo/DMA-ratio features to HMM input |
| MA Window | 45 days | Base signal window for backtest |
| Transaction Cost | 10 bps | Applied per side in backtest engine |
| Regime Shading | On | Colour bands on price chart |
| Live Mode | Off | Auto-refresh every 60 seconds |

---

## Library Usage

The `nautilus` package is designed as a **reusable Python library**. You can import individual modules and build far more than just the Streamlit dashboard.

## Advanced Library Usage

After `pip install nautilus-spark`, the package becomes a full-featured **regime engine** you can import and extend in any Python script, notebook, or production system.

Here are 25 advanced ways serious users leverage `nautilus` in their own code:

| # | Use Case | Description | Perfect for | Quick Code Sketch |
|---|----------|-------------|-------------|-------------------|
| 1 | **HMM States as ML Features** | Extract regime labels + posterior probabilities and append them as features to any ML pipeline (XGBoost, LightGBM, CatBoost, LSTM, etc.). | ML quants & feature engineers | `from nautilus.strategies.regime import RegimeHMM`<br>`hmm = RegimeHMM(n_components=5)`<br>`probs = hmm.predict_proba(features_df)`<br>`ml_df = pd.concat([features_df, probs], axis=1)` |
| 2 | **Sector / Stock-Level Regime Analysis** | Run the identical 5-state HMM on any Nifty 50 stock, sector ETF, or Bank Nifty for relative strength or pair-trading signals. | Sector rotation & stock pickers | `stock = load_index("RELIANCE.NS")`<br>`features = build_hmm_features(stock)`<br>`hmm.fit(features)` |
| 3 | **Options / Volatility Trading Overlay** | Scale option delta/gamma/vega exposure using the Hard Gate or current regime probability (sell premium only in Bull Quiet/Volatile). | Options & vol traders | `gate = hard_gate_from(hmm_result)`<br>`option_position = base_signal * gate * vega_scaler` |
| 4 | **Hyperparameter Grid Search** | Systematically test number of states, EM iterations, EWM span, macro features, etc. and rank by backtest Sharpe/Calmar. | Strategy researchers | `from itertools import product`<br>`for n, span in product([3,5,7], [3,5,8]):`<br>`    result = run_experiment(n_states=n, ewm=span)` |
| 5 | **Walk-Forward Out-of-Sample Validation** | Rolling-window re-fit the HMM to simulate true out-of-sample regime detection and avoid look-ahead bias. | Academic & robust backtesters | `for train_end in pd.date_range(...):`<br>`    hmm.fit(data[:train_end])`<br>`    oos = hmm.predict_proba(data[train_end:])` |
| 6 | **Portfolio-Wide Risk Overlay** | Apply one Nifty-derived Hard Gate across an entire equity, mutual-fund, or multi-asset portfolio to derisk everything in Stress/Panic regimes. | Portfolio managers & robo-advisors | `portfolio = pd.concat([asset1, asset2, ...], axis=1)`<br>`portfolio_gated = portfolio.mul(hard_gate, axis=0)` |
| 7 | **Multi-Model Ensemble Regimes** | Average or vote across 2–4 different HMMs (different features or state counts) for a more robust gate. | Advanced systematic traders | `gate = (gate1 + gate2 + gate3) / 3` |
| 8 | **Automated Daily Reporting Bot** | Script that runs every morning, prints current regime + 20-day forecast + signal, and pushes to Telegram/Discord/email. | Solo traders & small desks | `result = hmm.forward_simulate(20)`<br>`telegram.send(f"Regime: {current_regime} \| Gate: {gate.iloc[-1]}")` |
| 9 | **FastAPI / REST Microservice** | Wrap the regime engine behind a tiny API so trading bots or dashboards can query today's gate & probabilities. | Prop desks & algo platforms | `@app.get("/regime")`<br>`def get_regime():`<br>`    return {"gate": hard_gate.iloc[-1], "probs": posterior.tolist()}` |
| 10 | **Integration with VectorBT / Backtrader** | Feed Nautilus regimes + signals into popular backtesting frameworks for ultra-fast vectorized testing or live execution. | Power users | `import vectorbt as vbt`<br>`pf = vbt.Portfolio.from_signals(price, entries=signal * gate)` |
| 11 | **Custom Macro Feature Injection** | Add your own features (FII/DII flows, VIX, USDINR, CPI, etc.) to the HMM input pipeline. | Macro overlay researchers | `custom_features = build_hmm_features(price, extra_df=my_macro)`<br>`hmm.fit(custom_features)` |
| 12 | **Regime-Based Dynamic Position Sizing** | Combine Hard Gate with Kelly, volatility targeting, or risk-parity sizing per regime. | Risk-aware traders | `size = sharpened_kelly * hard_gate * (target_vol / current_vol)` |
| 13 | **Drawdown-Regime Filtering** | Create hybrid filters that only allow trades when both HMM regime and rolling drawdown are in acceptable states. | Drawdown-conscious quants | `filter = (hard_gate == 1) & (drawdown < 0.08)` |
| 14 | **Markov Transition Matrix Analysis** | Study regime persistence and transition probabilities to forecast regime duration and turning points. | Regime-cycle researchers | `trans = hmm_result.trans_matrix`<br>`expected_duration = 1 / (1 - np.diag(trans))` |
| 15 | **Posterior Probability Thresholding** | Generate probabilistic signals (e.g. only go long when P(Bull Quiet) > 0.65). | Probabilistic traders | `signal = (hmm.probs["Bull Quiet"] > 0.65).astype(int)` |
| 16 | **Batch Multi-Asset Regime Factory** | Run the full HMM + backtest pipeline on 50+ stocks or indices in one go and compare regime statistics. | Quant researchers | `results = {ticker: run_full_pipeline(ticker) for ticker in universe}` |
| 17 | **Monte-Carlo Regime Path Simulation** | Simulate thousands of future regime paths using the transition matrix for stress-testing. | Risk & scenario analysts | `paths = hmm.monte_carlo_simulate(n_paths=10000, steps=252)` |
| 18 | **Regime-Conditioned Strategy Library** | Build a collection of strategies that only activate in specific regimes (e.g. momentum only in Bull states). | Strategy developers | `if current_regime in ["Bull Quiet", "Bull Volatile"]:`<br>`    signal = momentum_strategy(...)` |
| 19 | **Anomaly Detection in Regime Shifts** | Flag unusual transitions or low-probability regime jumps for early warning. | Surveillance & risk teams | `if trans_prob < 0.01:`<br>`    alert("Rare regime jump detected")` |
| 20 | **Integration with pandas-ta / TA-Lib** | Combine Nautilus regimes with 100+ technical indicators for hybrid signals. | Technical + regime traders | `import pandas_ta as ta`<br>`df["rsi"] = ta.rsi(price)`<br>`final_signal = regime_gate * (df["rsi"] < 30)` |
| 21 | **Backtest with Realistic Slippage & Impact** | Extend the backtest engine with custom slippage models that vary by regime. | Institutional-grade backtesters | `result = run_backtest(..., slippage_model=regime_slippage)` |
| 22 | **Live Data Feed + Intraday Extension** | Hook Nautilus into websocket feeds (e.g. Zerodha, Alice Blue) for near-real-time regime updates. | Algo traders | `while True:`<br>`    tick = ws.get_tick()`<br>`    live_hmm.update(tick)` |
| 23 | **Cloud / Serverless Deployment** | Deploy the regime engine as AWS Lambda, Google Cloud Function, or Vercel cron job for daily signal generation. | Production teams | `# lambda_handler`<br>`return compute_current_gate()` |
| 24 | **Regime Attribution + Event Studies** | Correlate regime changes with macro events (RBI MPC, FOMC, budget day, etc.) for causal research. | Academic & macro researchers | `event_study = attribution.groupby(event_dates).agg({"return": "mean"})` |
| 25 | **Hybrid Model Benchmarking** | Compare Nautilus 5-state HMM against simpler models (RSI regime, volatility regime, 200-DMA only) in the same backtest engine. | Model validation & research | `bench = run_backtest(plain_ma_signal)`<br>`nautilus_perf = run_backtest(regime_signal)` |
