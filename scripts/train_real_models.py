"""Train XGBoost on REAL CPCB labels + Sentinel-2 features with honest out-of-fold spatial validation.

    python -m scripts.train_real_models --trials 30 --folds 5

Targets: DO, BOD, turbidity (real labels). Chl-a has no in-situ label and is not modelled. Each target
uses its own feature set (src.models.schema.FEATURE_SETS), chosen by clean ablations, not one pooled
list -- DO gets worse with context features added, BOD gets much better. This is now a *secondary*
deliverable: see scripts/train_screening.py for the pollution-screening classifiers, where the same
data actually shows skill. Reported metrics are out-of-fold: a model never sees the stations it is
scored on, and nearby stations share a fold. "Predict the training mean / median" baselines are scored
on the same folds.
"""
import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from src.data import nwdp
from src.data.training_table import FEATURE_COLS_REAL, TARGETS
from src.models.baselines import mean_baseline_oof, median_baseline_oof
from src.models.metrics import MetricsReporter
from src.models.schema import FEATURE_SETS
from src.models.spatial_cv import SpatialKFold
from src.models.xgboost_pipeline import WaterQualityXGB

TABLE_LARGE = nwdp.ROOT / "data" / "processed" / "train_real_large.parquet"
TABLE_SMALL = nwdp.ROOT / "data" / "processed" / "train_real.parquet"
TABLE = TABLE_LARGE if TABLE_LARGE.exists() else TABLE_SMALL
OUT = nwdp.ROOT / "reports" / "real"


def station_level_bod_ranking(df: pd.DataFrame, folds: int, seed: int, n_trials: int = 15) -> dict:
    """Median per-station spectra -> median per-station BOD: a station *ranking*, not a per-visit value.

    The strongest regression this project supports (clean-ablation R2 ~0.18 vs ~0 row-level): station
    identity, not any single visit's reflectance, carries most of BOD's variance (see schema.py).
    """
    feats = FEATURE_SETS["bod"]
    sub = df.dropna(subset=[*feats, "bod", "lat", "lon"])
    if sub.empty:
        return {"skill": "no labelled rows"}
    agg = sub.groupby("site").agg({**{c: "median" for c in feats}, "bod": "median",
                                   "lat": "median", "lon": "median"}).reset_index()
    n_stations = int(len(agg))
    if n_stations < folds:
        return {"n_stations": n_stations, "skill": f"too few stations for {folds}-fold CV"}
    pipe = WaterQualityXGB(n_folds=folds, random_state=seed, feature_cols={"bod": feats},
                           targets=["bod"], log_targets=("bod",))
    pipe.train(agg, n_trials=n_trials, verbose=False)
    oof = pipe.predict_oof(agg)["bod"].to_numpy()
    y = agg["bod"].to_numpy()
    return {
        "n_stations": n_stations,
        "R2": float(r2_score(y, oof)),
        "R2_log": float(r2_score(np.log1p(y), np.log1p(np.maximum(oof, 0.0)))),
        "spearman": float(spearmanr(y, oof).correlation),
    }


def turbidity_anomaly_oof(df: pd.DataFrame, folds: int, seed: int) -> dict:
    """Features and target as deviations from each station's own mean: can the satellite see *change*
    at a known station, rather than which station is generally dirtier (clean-ablation R2 ~0.085).

    Deliberately bypasses WaterQualityXGB, whose log1p transform and clipping assume a non-negative
    physical target -- an anomaly is signed and centred on zero.
    """
    feats = FEATURE_SETS["turbidity"]
    sub = df.dropna(subset=[*feats, "turbidity", "lat", "lon"]).copy()
    sub = sub[sub.groupby("site")["turbidity"].transform("count") >= 2]      # need >=2 visits for a within-station mean
    sub = sub.reset_index(drop=True)
    if sub.empty or sub["site"].nunique() < folds:
        return {"skill": f"too few multi-visit stations for {folds}-fold CV"}
    for c in feats:
        sub[c] = sub[c] - sub.groupby("site")[c].transform("mean")
    sub["turbidity_anom"] = sub["turbidity"] - sub.groupby("site")["turbidity"].transform("mean")

    X, y = sub[feats], sub["turbidity_anom"].to_numpy()
    oof = np.full(len(sub), np.nan)
    for train_idx, val_idx in SpatialKFold(n_folds=folds, random_state=seed).split(sub):
        model = xgb.XGBRegressor(tree_method="hist", random_state=seed, n_jobs=2, verbosity=0,
                                 n_estimators=150, max_depth=4, learning_rate=0.05)
        model.fit(X.iloc[train_idx], y[train_idx])
        oof[val_idx] = model.predict(X.iloc[val_idx])
    mask = np.isfinite(oof)
    return {
        "n_rows": int(mask.sum()), "n_stations": int(sub["site"].nunique()),
        "R2": float(r2_score(y[mask], oof[mask])),
        "spearman": float(spearmanr(y[mask], oof[mask]).correlation),
    }


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
                           feature_cols=FEATURE_SETS, targets=TARGETS)
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

    station_bod = station_level_bod_ranking(df, args.folds, args.seed)
    print(f"station-level BOD ranking: {station_bod}")
    turb_anomaly = turbidity_anomaly_oof(df, args.folds, args.seed)
    print(f"turbidity anomaly (within-station): {turb_anomaly}")

    summary = {
        "data": f"real CPCB in-situ + Sentinel-2 L2A ({TABLE.name})",
        "validation": f"{args.folds}-fold site-blocked spatial CV, out-of-fold predictions",
        "rows": int(len(df)),
        "stations": int(df["site"].nunique()),
        "feature_sets": FEATURE_SETS,
        "best_trial_cv_rmse": cv,
        "metrics": metrics.to_dict(orient="records"),
        "station_level_bod_ranking": station_bod,
        "turbidity_anomaly": turb_anomaly,
        "skill_notes": {"do": "none — DO is not optically active; kept for the low-DO screening flag "
                              "(scripts/train_screening.py), not for its own accuracy"},
    }
    (OUT / "metrics_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    if not args.no_shap:
        from src.models import SHAPExplainer
        SHAPExplainer(output_dir=OUT / "shap_plots", dpi=150).explain_all(
            pipe, df[FEATURE_COLS_REAL], targets=[t for t in TARGETS if t in pipe._models])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
