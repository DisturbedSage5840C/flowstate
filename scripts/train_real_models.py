"""Train XGBoost on REAL CPCB labels + Sentinel-2 features with honest out-of-fold spatial validation.

    python -m scripts.train_real_models --trials 30 --folds 5

Targets: DO, BOD, turbidity (real labels). Chl-a has no in-situ label and is not modelled.
Each target uses its own feature set (src.models.schema.FEATURE_SETS) -- context features help BOD and
measurably hurt DO and turbidity, so one shared list was wrong.

Reported metrics are out-of-fold: a model never sees the stations it is scored on, and nearby stations
share a fold. "Predict the training mean / median" baselines are scored on the same folds.

These regressions are the project's SECONDARY output. Concentration retrieval from reflectance is close to
unskilled for these targets (see the per-target verdicts written into metrics_summary.json); the primary
deliverable is scripts/train_screening.py, which ranks stations by breach probability instead.
Two variants the evidence supports are also reported: a station-level BOD ranking, and a turbidity anomaly
model for stations with repeat visits.
"""
import argparse
import json

import numpy as np
import pandas as pd

from src.data import nwdp
from src.models.baselines import mean_baseline_oof, median_baseline_oof
from src.models.metrics import MetricsReporter
from src.models.schema import FEATURE_SETS, REAL_TARGET_COLS as TARGETS, SPECTRAL_COLS, TYPE_COLS
from src.models.spatial_cv import SpatialKFold
from src.models.xgboost_pipeline import WaterQualityXGB

TABLE_LARGE = nwdp.ROOT / "data" / "processed" / "train_real_large.parquet"
TABLE_SMALL = nwdp.ROOT / "data" / "processed" / "train_real.parquet"
TABLE = TABLE_LARGE if TABLE_LARGE.exists() else TABLE_SMALL
OUT = nwdp.ROOT / "reports" / "real"

# Written next to the numbers so a reader cannot take a near-zero R2 for a working retrieval.
SKILL_NOTES = {
    "do": ("none - dissolved oxygen is not optically active, so reflectance cannot see it. Out-of-fold R2 is "
           "~0 and context features make it worse. Use the do_lt_4 screening flag instead."),
    "bod": ("weak - BOD is not optically active either; what skill exists comes mostly from urban-proximity "
            "context, not the bands. Useful for ranking, not for a concentration."),
    "turbidity": ("weak - turbidity IS optically active, but 85% of its variance is between stations and "
                  "site-blocked validation withholds exactly that. Rank correlation is modest; the anomaly "
                  "variant below is the usable form."),
}


def fold_labels(df: pd.DataFrame, folds: int, seed: int) -> np.ndarray:
    """Which spatial fold each row was scored in (so constant baselines don't get a fake Spearman)."""
    labels = np.full(len(df), -1)
    for i, (_, val_idx) in enumerate(SpatialKFold(n_folds=folds, random_state=seed).split(df)):
        labels[val_idx] = i
    return labels


def oof_r2_spearman(df: pd.DataFrame, feats: list[str], target: str, folds: int, seed: int) -> dict:
    """Plain out-of-fold R2/Spearman for the two variant models (log1p scale for heavy-tailed targets)."""
    import xgboost as xgb
    from scipy import stats

    d = df.reset_index(drop=True)
    y = d[target].to_numpy()
    p = np.full(len(d), np.nan)
    for train_idx, val_idx in SpatialKFold(n_folds=folds, random_state=seed).split(d):
        model = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, reg_lambda=2.0, n_jobs=4, verbosity=0, random_state=seed)
        model.fit(d.loc[train_idx, feats], y[train_idx])
        p[val_idx] = model.predict(d.loc[val_idx, feats])
    return {"n": int(len(d)), "sites": int(d["site"].nunique()),
            "r2": round(float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()), 4),
            "spearman": round(float(stats.spearmanr(y, p).correlation), 4)}


def station_level_bod(df: pd.DataFrame, folds: int, seed: int) -> dict:
    """One row per station: median spectra -> median BOD. Averaging out matchup noise roughly doubles R2."""
    d = df[df["bod"].notna()]
    feats = FEATURE_SETS["bod"]
    agg = d.groupby("site").agg({**{c: "median" for c in feats}, "lat": "first", "lon": "first",
                                 "water_body_type": "first", "bod": "median"}).reset_index()
    agg["y"] = np.log1p(agg["bod"])
    res = oof_r2_spearman(agg, feats, "y", folds, seed)
    res["description"] = "station-level BOD ranking (median spectra -> median BOD, log1p)"
    return res


def turbidity_anomaly(df: pd.DataFrame, folds: int, seed: int, min_visits: int = 4) -> dict:
    """Deviation from each station's own mean: can the satellite see turbidity *change* where we already
    know the station? This is the monitoring question, and unlike the absolute level it has positive skill."""
    d = df[df["turbidity"].notna()].copy()
    d["y0"] = np.log1p(d["turbidity"])
    d = d[d.groupby("site")["y0"].transform("size") >= min_visits].copy()
    if len(d) < 100:
        return {"note": f"not enough stations with >= {min_visits} visits"}
    d["y"] = d["y0"] - d.groupby("site")["y0"].transform("mean")
    feats = []
    for c in SPECTRAL_COLS:
        d[f"{c}_anom"] = d[c] - d.groupby("site")[c].transform("mean")
        feats.append(f"{c}_anom")
    res = oof_r2_spearman(d, feats, "y", folds, seed)
    res["description"] = (f"turbidity anomaly vs each station's own mean, stations with >= {min_visits} visits "
                          "(spectra also as anomalies)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-shap", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(TABLE)
    pipe = WaterQualityXGB(n_folds=args.folds, random_state=args.seed,
                           feature_cols=FEATURE_SETS, targets=TARGETS)
    df = df.dropna(subset=pipe.all_feature_cols).reset_index(drop=True)
    print(f"{len(df)} rows | {df['site'].nunique()} stations | labels:",
          {t: int(df[t].notna().sum()) for t in TARGETS})
    print("feature sets:", {t: len(FEATURE_SETS[t]) for t in TARGETS}, "columns per target\n")

    cv = pipe.train(df, n_trials=args.trials, verbose=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "models").mkdir(exist_ok=True)
    pipe.save(OUT / "models")

    reporter = MetricsReporter(targets=TARGETS)
    folds_of = fold_labels(df, args.folds, args.seed)
    tables = []
    xgb_oof = pipe.predict_oof(df)
    pd.concat([df[["site", "date"]], xgb_oof.add_suffix("_pred")], axis=1).to_parquet(
        OUT / "oof_predictions.parquet", index=False)
    for name, preds in (
        ("xgboost_oof", xgb_oof),
        ("baseline_mean", mean_baseline_oof(df, TARGETS, args.folds, args.seed)),
        ("baseline_median", median_baseline_oof(df, TARGETS, args.folds, args.seed)),
    ):
        tab = reporter.report(df, preds, fold_labels=folds_of)
        tab.insert(0, "model", name)
        tables.append(tab)
    metrics = pd.concat(tables, ignore_index=True)
    metrics.to_csv(OUT / "metrics_table.csv", index=False)
    reporter.print_table(metrics[metrics.model == "xgboost_oof"].drop(columns="model"))

    print("variants:")
    variants = {"station_level_bod": station_level_bod(df, args.folds, args.seed),
                "turbidity_anomaly": turbidity_anomaly(df, args.folds, args.seed)}
    for name, v in variants.items():
        if "r2" in v:
            print(f"  {name:20s} n={v['n']:5d} sites={v['sites']:4d}  R2={v['r2']:+.3f}  Spearman={v['spearman']:+.3f}")

    summary = {
        "data": f"real CPCB in-situ + Sentinel-2 L2A ({TABLE.name})",
        "validation": f"{args.folds}-fold site-blocked spatial CV, out-of-fold predictions",
        "role": "SECONDARY output - the primary deliverable is scripts/train_screening.py",
        "rows": int(len(df)),
        "stations": int(df["site"].nunique()),
        "feature_sets": {t: FEATURE_SETS[t] for t in TARGETS},
        "skill": SKILL_NOTES,
        "best_trial_cv_rmse": cv,
        "variants": variants,
        "metrics": metrics.to_dict(orient="records"),
    }
    (OUT / "metrics_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    if not args.no_shap:
        from src.models import SHAPExplainer
        explainer = SHAPExplainer(output_dir=OUT / "shap_plots", dpi=150)
        for target in [t for t in TARGETS if t in pipe._models]:
            explainer.explain_all(pipe, df[FEATURE_SETS[target]], targets=[target])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
