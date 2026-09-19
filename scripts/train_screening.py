"""Pollution screening: rank station-visits by the probability of breaching a regulatory limit.

    python -m scripts.train_screening [--folds 5]

This is the project's primary model output. Concentration regression from reflectance is near-zero
out-of-fold (see scripts/train_real_models.py and reports/real/metrics_table.csv) because 85-100% of each
target's variance sits *between* stations, which site-blocked validation deliberately withholds. Asking
"which stations are likely to breach a limit, worst first" is answerable from the same data: BOD > 3 mg/L
reaches AUC ~0.75 at roughly twice the base-rate precision.

Labels come from measured CPCB chemistry only (src/models/tier_classifier.SCREENING_TARGETS), never from
satellite bands, so the task is leakage-free. Every number is out-of-fold on site-blocked spatial folds.
Writes reports/real/screening_metrics.json and a ranked shortlist CSV.
"""
import argparse
import json

import numpy as np
import pandas as pd

from src.data import nwdp
from src.models.schema import REAL_FEATURE_COLS
from src.models.tier_classifier import SCREENING_TARGETS, ScreeningClassifier

TABLE_LARGE = nwdp.ROOT / "data" / "processed" / "train_real_large.parquet"
TABLE_SMALL = nwdp.ROOT / "data" / "processed" / "train_real.parquet"
TABLE = TABLE_LARGE if TABLE_LARGE.exists() else TABLE_SMALL
OUT = nwdp.ROOT / "reports" / "real"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shortlist-target", default="cpcb_polluted", choices=sorted(SCREENING_TARGETS))
    args = ap.parse_args()

    df = pd.read_parquet(TABLE)
    print(f"{TABLE.name}: {len(df)} rows, {df['site'].nunique()} stations\n")

    results = {}
    for name in SCREENING_TARGETS:
        clf = ScreeningClassifier(REAL_FEATURE_COLS, name, n_folds=args.folds, random_state=args.seed)
        res = clf.evaluate(df)
        results[name] = res
        if "roc_auc" in res:
            print(f"{name:16s} n={res['n']:5d} base={res['base_rate']:.1%}  AUC={res['roc_auc']:.3f}  "
                  f"AP={res['average_precision']:.3f} ({res['lift_over_base_rate']:.1f}x base)  "
                  f"precision@top10%={res['precision_at_k']['top_10pct']:.2f}")
        else:
            print(f"{name:16s} {res.get('note')}")

    # Ranked inspection shortlist for the headline target: worst-first, out-of-fold so it is honest.
    clf = ScreeningClassifier(REAL_FEATURE_COLS, args.shortlist_target, n_folds=args.folds, random_state=args.seed)
    frame = clf.frame(df)
    frame["breach_probability"] = clf.predict_oof(frame)
    cols = [c for c in ("site", "state", "water_body_type", "date", "do", "bod", "turbidity", "cpcb_class",
                        "breach_probability", "_y") if c in frame.columns]
    shortlist = (frame[cols].rename(columns={"_y": "actually_breached"})
                 .dropna(subset=["breach_probability"])
                 .sort_values("breach_probability", ascending=False))
    station_rank = (shortlist.groupby("site")
                    .agg(breach_probability=("breach_probability", "mean"),
                         visits=("breach_probability", "size"),
                         actually_breached=("actually_breached", "mean"),
                         state=("state", "first") if "state" in shortlist.columns else ("site", "first"))
                    .sort_values("breach_probability", ascending=False).reset_index())

    OUT.mkdir(parents=True, exist_ok=True)
    station_rank.to_csv(OUT / "screening_shortlist.csv", index=False)
    summary = {
        "table": TABLE.name,
        "validation": f"{args.folds}-fold site-blocked spatial CV, out-of-fold probabilities",
        "features": REAL_FEATURE_COLS,
        "note": ("Accuracy is deliberately not reported: at a 7-29% base rate, always predicting 'clean' scores "
                 "well and is useless. AUC, average precision vs base rate and precision@k are what a screening "
                 "shortlist is judged on."),
        "targets": results,
        "shortlist_target": args.shortlist_target,
        "shortlist_rows": int(len(station_rank)),
    }
    (OUT / "screening_metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\ntop stations to inspect ({args.shortlist_target}):")
    print(station_rank.head(8).round(3).to_string(index=False))
    print(f"\nwrote {OUT / 'screening_metrics.json'} and {OUT / 'screening_shortlist.csv'}")


if __name__ == "__main__":
    main()
