"""Train XGBoost on REAL CPCB labels + Sentinel-2 features with honest out-of-fold spatial validation.

    python -m scripts.train_real_models --trials 30 --folds 5

Targets: DO, BOD, turbidity (real labels). Chl-a has no in-situ label and is not modelled.
Reported metrics are out-of-fold: a model never sees the stations it is scored on, and nearby stations
share a fold. "Predict the training mean / median" baselines are scored on the same folds.
"""
import argparse
import json

import pandas as pd

from src.data import nwdp
from src.data.training_table import FEATURE_COLS_REAL, TARGETS
from src.models.baselines import mean_baseline_oof, median_baseline_oof
from src.models.metrics import MetricsReporter
from src.models.xgboost_pipeline import WaterQualityXGB

TABLE_LARGE = nwdp.ROOT / "data" / "processed" / "train_real_large.parquet"
TABLE_SMALL = nwdp.ROOT / "data" / "processed" / "train_real.parquet"
TABLE = TABLE_LARGE if TABLE_LARGE.exists() else TABLE_SMALL
OUT = nwdp.ROOT / "reports" / "real"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-shap", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(TABLE)
    df = df.dropna(subset=FEATURE_COLS_REAL).reset_index(drop=True)
    print(f"{len(df)} rows | {df['site'].nunique()} stations | labels:",
          {t: int(df[t].notna().sum()) for t in TARGETS})

    pipe = WaterQualityXGB(n_folds=args.folds, random_state=args.seed,
                           feature_cols=FEATURE_COLS_REAL, targets=TARGETS)
    cv = pipe.train(df, n_trials=args.trials, verbose=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "models").mkdir(exist_ok=True)
    pipe.save(OUT / "models")

    reporter = MetricsReporter(targets=TARGETS)
    tables = []
    xgb_oof = pipe.predict_oof(df)
    oof_out = pd.concat([df[["site", "date"]], xgb_oof.add_suffix("_pred")], axis=1)
    oof_out.to_parquet(OUT / "oof_predictions.parquet", index=False)
    for name, preds in (
        ("xgboost_oof", xgb_oof),
        ("baseline_mean", mean_baseline_oof(df, TARGETS, args.folds, args.seed)),
        ("baseline_median", median_baseline_oof(df, TARGETS, args.folds, args.seed)),
    ):
        tab = reporter.report(df, preds)
        tab.insert(0, "model", name)
        tables.append(tab)
    metrics = pd.concat(tables, ignore_index=True)
    metrics.to_csv(OUT / "metrics_table.csv", index=False)
    reporter.print_table(metrics[metrics.model == "xgboost_oof"].drop(columns="model"))

    summary = {
        "data": f"real CPCB in-situ + Sentinel-2 L2A ({TABLE.name})",
        "validation": f"{args.folds}-fold site-blocked spatial CV, out-of-fold predictions",
        "rows": int(len(df)),
        "stations": int(df["site"].nunique()),
        "best_trial_cv_rmse": cv,
        "metrics": metrics.to_dict(orient="records"),
    }
    (OUT / "metrics_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    if not args.no_shap:
        from src.models import SHAPExplainer
        SHAPExplainer(output_dir=OUT / "shap_plots", dpi=150).explain_all(
            pipe, df[FEATURE_COLS_REAL], targets=[t for t in TARGETS if t in pipe._models])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
