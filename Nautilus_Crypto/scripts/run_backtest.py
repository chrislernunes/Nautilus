"""
scripts/run_backtest.py
─────────────────────────────────────────────────────────────────────────────
Standalone backtest runner.  Fetches fresh historical data, fits the HMM,
runs the backtest, prints a full performance report, and saves an equity
curve CSV to logs/.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --optimise   # run WF grid search first
    python scripts/run_backtest.py --days 365   # use only last N days
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# ── Project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import pandas as pd

from python.utils.logging_config import configure_logging
configure_logging(level=logging.INFO)
log = logging.getLogger("scripts.backtest")


def main() -> None:
    parser = argparse.ArgumentParser(description="Nautilus BTC — Standalone Backtest")
    parser.add_argument("--optimise", action="store_true",
                        help="Run walk-forward hyperparameter optimisation first")
    parser.add_argument("--days", type=int, default=730,
                        help="Number of historical daily bars to use (default: 730)")
    parser.add_argument("--states", type=int, default=4,
                        help="Number of HMM hidden states (default: 4)")
    parser.add_argument("--cov", default="diag",
                        choices=["diag", "full", "tied", "spherical"],
                        help="HMM covariance type (default: diag)")
    args = parser.parse_args()

    # ── Fetch historical data ─────────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  NAUTILUS BTC  —  Standalone Backtest Runner          ║")
    print("╚══════════════════════════════════════════════════════╝\n")
    print(f"Fetching {args.days} days of BTCUSDT daily data from Binance...")

    from python.core.historical import fetch_klines
    daily_df = asyncio.run(fetch_klines("1d", args.days))

    if daily_df.empty:
        print("ERROR: Failed to fetch historical data. Check your internet connection.")
        sys.exit(1)

    print(f"✓ Fetched {len(daily_df)} daily bars  "
          f"({daily_df.index[0].date()} → {daily_df.index[-1].date()})\n")

    # ── Feature engineering ───────────────────────────────────────────────────
    from python.core.features import build_features
    print("Building feature matrix...")
    features = build_features(daily_df, z_score=True)
    print(f"✓ Features: {features.shape[0]} bars × {features.shape[1]} cols")
    print(f"  Columns: {', '.join(features.columns.tolist())}\n")

    # ── Hyperparameter optimisation (optional) ────────────────────────────────
    from python.backtest.engine import HyperParams, optimise_hyperparams

    if args.optimise:
        print("Running walk-forward hyperparameter optimisation (4 folds)...")
        best_hp, opt_results = optimise_hyperparams(daily_df, n_splits=4)
        print("\nOptimisation results (sorted by OOS Sharpe):")
        print(opt_results.to_string(index=False))
        print(f"\n✓ Best config: {best_hp.label()}")
        n_states = best_hp.n_states
        cov_type = best_hp.covariance_type
    else:
        n_states = args.states
        cov_type = args.cov
        print(f"Using config: n_states={n_states}, covariance_type={cov_type}")

    # ── Fit HMM ───────────────────────────────────────────────────────────────
    from python.core.hmm_model import RegimeHMM
    print(f"\nFitting GaussianHMM ({n_states} states, {cov_type} covariance)...")
    model = RegimeHMM(n_states=n_states, covariance_type=cov_type, n_iter=100)
    model.fit(features)

    if not model.is_fitted:
        print("ERROR: Model fit failed. Check logs for details.")
        sys.exit(1)

    print(f"✓ Model fitted successfully")

    # Transition matrix diagnostic
    import numpy as np
    trans_diag = np.diag(model.transition_matrix)
    print(f"  Transition matrix diagonal: {np.round(trans_diag, 3)}")
    if trans_diag.mean() < 0.70:
        print("  ⚠  Low persistence — consider increasing min_fit_obs or reducing n_states")

    # Regime distribution
    states, posteriors = model.predict(features)
    from python.core.hmm_model import Regime, REGIME_LABELS
    print("\n  Regime distribution across sample:")
    for r in Regime:
        if r.value < n_states:
            count = (states == r.value).sum()
            pct   = count / len(states) * 100
            print(f"    {REGIME_LABELS[r]:<18} {count:>4} days  ({pct:.1f}%)")

    # ── Backtest ──────────────────────────────────────────────────────────────
    from python.backtest.engine import run_backtest
    print("\nRunning vectorised backtest...")
    bnh, regime = run_backtest(daily_df, model, features)
    print("✓ Backtest complete\n")

    # ── Print results table ───────────────────────────────────────────────────
    col_w = 24
    metrics = list(bnh.to_dict().keys())
    metrics = [m for m in metrics if m != "Label"]

    print("═" * 60)
    print(f"  {'METRIC':<22}  {'BUY & HOLD':>15}  {'REGIME STRAT':>15}")
    print("═" * 60)
    for m in metrics:
        bv = bnh.to_dict()[m]
        rv = regime.to_dict()[m]

        # Highlight if regime beats B&H
        marker = " "
        try:
            bv_num = float(str(bv).replace("%", ""))
            rv_num = float(str(rv).replace("%", ""))
            if m in ("Total Ret %", "Ann Ret %", "Sharpe", "Sortino", "Calmar", "Win Rate %"):
                marker = "▲" if rv_num > bv_num else " "
            elif m in ("Max DD %", "Volatility %"):
                marker = "▲" if rv_num > bv_num else " "   # lower DD is better
        except (ValueError, TypeError):
            pass

        print(f"  {m:<22}  {str(bv):>15}  {str(rv):>14}{marker}")
    print("═" * 60)

    # ── Save equity curves ────────────────────────────────────────────────────
    Path("logs").mkdir(exist_ok=True)
    eq_df = pd.DataFrame({
        "buy_and_hold":     bnh.equity,
        "regime_strategy":  regime.equity,
    })
    out_path = Path("logs/equity_curves.csv")
    eq_df.to_csv(out_path)
    print(f"\n✓ Equity curves saved to {out_path}")

    # Summary
    print(f"\n  Regime Sharpe: {regime.sharpe:.2f}  vs  B&H Sharpe: {bnh.sharpe:.2f}")
    print(f"  Regime Max DD: {regime.max_dd*100:.1f}%  vs  B&H Max DD: {bnh.max_dd*100:.1f}%")
    print(f"  Total trades: {regime.n_trades}")
    print()


if __name__ == "__main__":
    main()
