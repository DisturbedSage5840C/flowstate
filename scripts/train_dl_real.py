"""Train the CNN-BiLSTM-attention model on the REAL table with honest out-of-fold spatial validation.

    python -m scripts.train_dl_real --epochs 80 --window 3

Same folds, targets and metrics as scripts/train_real_models.py (XGBoost), so the two are directly comparable.
Writes reports/real/dl_metrics_table.csv, dl_oof_predictions.parquet, dl_summary.json and dl_artifact.pt.
"""
import argparse
import json

import pandas as pd

from src.data import nwdp
from src.models.dl_training import run_dl_experiment
from src.models.schema import REAL_FEATURE_COLS, REAL_TARGET_COLS

TABLE = nwdp.ROOT / "data" / "processed" / "train_real.parquet"
OUT = nwdp.ROOT / "reports" / "real"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--max-gap-days", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(TABLE)
    summary = run_dl_experiment(df, REAL_FEATURE_COLS, REAL_TARGET_COLS, OUT, n_folds=args.folds, seed=args.seed,
                                epochs=args.epochs, patience=args.patience, window=args.window,
                                max_gap_days=args.max_gap_days)
    m = pd.DataFrame(summary["metrics"])
    print(m[m["water_body_type"] == "overall"][["model", "target", "n", "R2", "RMSE", "MAE"]].round(3).to_string(index=False))
    print(f"share of samples with visit history: {summary['share_of_samples_with_history']:.0%}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
