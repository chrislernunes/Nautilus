"""
scripts/run_optimise.py
─────────────────────────────────────────────────────────────────────────────
Standalone walk-forward hyperparameter optimisation.
Outputs a ranked config table and writes the best model to models/.

Usage:
    python scripts/run_optimise.py
    python scripts/run_optimise.py --folds 5 --days 1000
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from python.utils.logging_config import configure_logging
configure_logging(level=logging.INFO)
log = logging.getLogger("scripts.optimise")


def main() -> None:
    parser = argparse.ArgumentParser(description="Nautilus BTC — WF Optimisation")
    parser.add_argument("--folds", type=int, default=4, help="Walk-forward folds")
    parser.add_argument("--days",  type=int, default=730, help="Historical days")
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  NAUTILUS BTC  —  Walk-Forward Optimisation           ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    from python.core.historical import fetch_klines
    print(f"Fetching {args.days} daily bars...")
    daily_df = asyncio.run(fetch_klines("1d", args.days))
    print(f"✓ {len(daily_df)} bars ({daily_df.index[0].date()} → {daily_df.index[-1].date()})\n")

    from python.backtest.engine import optimise_hyperparams, HYPERPARAMETER_GRID
    print(f"Grid size: {len(HYPERPARAMETER_GRID)} configurations × {args.folds} folds\n")

    best_hp, results = optimise_hyperparams(daily_df, n_splits=args.folds)

    print("\n╔══ Optimisation Results (sorted by OOS Sharpe) ══════════╗")
    for _, row in results.iterrows():
        marker = "◄ BEST" if row["config"] == best_hp.label() else ""
        print(f"  {row['config']:<35}  Sharpe={row['sharpe']:+.3f}  "
              f"Ret={row['total_return']*100:+.1f}%  "
              f"DD={row['max_dd']*100:.1f}%  {marker}")
    print("╚═══════════════════════════════════════════════════════════╝\n")

    # Fit and save best model
    from python.core.features import build_features
    from python.core.hmm_model import RegimeHMM
    from python.backtest.engine import run_backtest, FEATURE_SETS

    feats = build_features(daily_df, z_score=True, z_window=best_hp.z_window)
    feat_cols = FEATURE_SETS.get(best_hp.feature_set)
    if feat_cols:
        feats = feats[[c for c in feat_cols if c in feats.columns]]

    model = RegimeHMM(
        n_states        = best_hp.n_states,
        covariance_type = best_hp.covariance_type,
        n_iter          = 150,
        model_path      = Path("models/hmm_best.pkl"),
    )
    model.fit(feats)
    print(f"✓ Best model saved to models/hmm_best.pkl")

    bnh, regime = run_backtest(daily_df, model, feats)
    print(f"\nFull-sample IS performance:")
    print(f"  Regime  Sharpe={regime.sharpe:.2f}  MaxDD={regime.max_dd*100:.1f}%"
          f"  Trades={regime.n_trades}")
    print(f"  B&H     Sharpe={bnh.sharpe:.2f}  MaxDD={bnh.max_dd*100:.1f}%\n")


if __name__ == "__main__":
    main()
