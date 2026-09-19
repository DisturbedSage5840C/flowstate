"""Train pollution-screening classifiers with honest out-of-fold spatial validation.

    python -m scripts.train_screening [--folds 5]

This is the project's headline deliverable (see AQUA_SENSE_PROJECT_PLAN.md): clean ablations found
real skill in a binary "is this station polluted" screen (BOD>3 AUC ~0.75) where the row-level DO/BOD/
turbidity regressions have none. Targets are derived only from measured chemistry (leakage-free, see
src.models.tier_classifier.add_screening_targets): bod_gt_3 (CPCB Class C limit), bod_gt_6, do_lt_4,
cpcb_polluted (assigned class D/E/Below E). The 5-class wqi_tier model is also retrained here and
reported as a secondary result (macro-F1 vs the majority-class baseline).

Writes reports/real/screening_metrics.json (AUC/AP/precision@k/calibration per target) and
reports/real/screening_oof.parquet (out-of-fold predicted probabilities per visit, for the dashboard's
ranked "inspect first" shortlist).
"""
import argparse
import json

import pandas as pd

from src.data import nwdp
from src.models.schema import REAL_FEATURE_COLS
from src.models.tier_classifier import SCREENING_TARGETS, ScreeningClassifier, WQITierClassifier, add_screening_targets

TABLE_LARGE = nwdp.ROOT / "data" / "processed" / "train_real_large.parquet"
TABLE_SMALL = nwdp.ROOT / "data" / "processed" / "train_real.parquet"
TABLE = TABLE_LARGE if TABLE_LARGE.exists() else TABLE_SMALL
OUT = nwdp.ROOT / "reports" / "real" / "screening_metrics.json"
OOF_OUT = nwdp.ROOT / "reports" / "real" / "screening_oof.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(TABLE)
    df = add_screening_targets(df)

    results = {"table": TABLE.name, "feature_cols": REAL_FEATURE_COLS, "targets": {}}
    oof_out = None
    for target in SCREENING_TARGETS:
        sub = df.dropna(subset=[*REAL_FEATURE_COLS, target]).reset_index(drop=True)
        if sub[target].nunique() < 2 or sub["site"].nunique() < args.folds:
            print(f"skip {target}: not enough labelled rows/sites ({len(sub)} rows)")
            continue
        clf = ScreeningClassifier(feature_cols=REAL_FEATURE_COLS, target=target,
                                  n_folds=args.folds, random_state=args.seed)
        res = clf.evaluate(sub)
        print(f"{target}: n={res['n']} base_rate={res['base_rate']:.3f} AUC={res['auc']:.3f} "
              f"AP={res['average_precision']:.3f} (lift {res['lift_vs_base_rate']:.2f}x)")
        results["targets"][target] = res

        proba = clf.predict_proba_oof(sub)
        col = pd.DataFrame({"site": sub["site"], "date": sub["date"], f"{target}_proba": proba})
        oof_out = col if oof_out is None else oof_out.merge(col, on=["site", "date"], how="outer")

    if oof_out is not None:
        oof_out.to_parquet(OOF_OUT, index=False)
        print("wrote", OOF_OUT)

    tier = WQITierClassifier(feature_cols=REAL_FEATURE_COLS, target="wqi_tier",
                             n_folds=args.folds, random_state=args.seed)
    sub = df.dropna(subset=[*REAL_FEATURE_COLS, "wqi_tier"]).reset_index(drop=True)
    if sub["site"].nunique() >= args.folds:
        tier_result = tier.evaluate(sub)
        print(f"wqi_tier (secondary, 5-class): macro-F1 {tier_result['model']['macro_f1']:.3f} vs "
              f"majority-class baseline {tier_result['majority_class_baseline']['macro_f1']:.3f}")
        results["wqi_tier"] = tier_result

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, default=str))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
