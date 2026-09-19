"""Predict WQI tier (Excellent..Very Poor) from satellite-only features, honest spatial-CV OOF.

    python -m scripts.train_tier_classifier [--target wqi_tier|cpcb_class] [--folds 5]

wqi_tier/cpcb_class are derived from measured chemistry, never from satellite bands (see
src/wqi/wqi_engine.py), so this is a leakage-free secondary deliverable alongside the DO/BOD/
turbidity regressors: predicting the tier is a coarser, more decision-useful question than an
exact concentration, though it shares the same weak-optical-signal ceiling.
"""
import argparse
import json

import pandas as pd

from src.data import nwdp
from src.models.schema import REAL_FEATURE_COLS
from src.models.tier_classifier import WQITierClassifier

TABLE_LARGE = nwdp.ROOT / "data" / "processed" / "train_real_large.parquet"
TABLE_SMALL = nwdp.ROOT / "data" / "processed" / "train_real.parquet"
TABLE = TABLE_LARGE if TABLE_LARGE.exists() else TABLE_SMALL
OUT = nwdp.ROOT / "reports" / "real" / "tier_classification_metrics.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="wqi_tier", choices=["wqi_tier", "cpcb_class"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(TABLE)
    df = df.dropna(subset=REAL_FEATURE_COLS + [args.target]).reset_index(drop=True)
    print(f"{len(df)} rows | {df['site'].nunique()} stations | "
          f"{args.target} distribution: {df[args.target].value_counts().to_dict()}")

    clf = WQITierClassifier(feature_cols=REAL_FEATURE_COLS, target=args.target, n_folds=args.folds,
                            random_state=args.seed)
    result = clf.evaluate(df)
    result["table"] = TABLE.name
    result["feature_cols"] = REAL_FEATURE_COLS

    print(f"\nOOF accuracy: {result['model']['accuracy']:.3f}  (majority-class baseline: "
          f"{result['majority_class_baseline']['accuracy']:.3f})")
    print(f"OOF macro F1: {result['model']['macro_f1']:.3f}  (majority-class baseline: "
          f"{result['majority_class_baseline']['macro_f1']:.3f})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, default=str))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
