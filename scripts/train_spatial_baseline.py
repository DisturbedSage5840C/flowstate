"""Train + evaluate the spatial-KNN "densification" regressor: given the existing CPCB network stays in
place, how well can DO/BOD/turbidity be estimated at a nearby unmonitored point?

    python -m scripts.train_spatial_baseline [--folds 20] [--trials 5] [--k 5]

This is a DIFFERENT question from `scripts.train_real_models` (which asks "can this work with zero
nearby CPCB coverage", answered honestly near R^2 0 via site-blocked SpatialKFold). This script uses
`src.models.spatial_cv.StationKFold` (a station is held out entirely, but the rest of the network stays
available) -- see `src/models/spatial_baseline.py`'s module docstring for the full reasoning and the
validated numbers. Results here must never be compared directly against `reports/real/metrics_table.csv`
without both being clearly labelled -- they are written to separate files for exactly that reason.
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from src.data import nwdp
from src.data.city_proximity import haversine_km
from src.models.schema import FEATURE_SETS, LOG_TARGETS, REAL_TARGET_COLS
from src.models.spatial_baseline import SpatialKNNRegressor

TABLE_LARGE = nwdp.ROOT / "data" / "processed" / "train_real_large.parquet"
TABLE_SMALL = nwdp.ROOT / "data" / "processed" / "train_real.parquet"
TABLE = TABLE_LARGE if TABLE_LARGE.exists() else TABLE_SMALL
OUT = nwdp.ROOT / "reports" / "real"
DUP_KM_THRESHOLD = 0.1    # near-exact-coordinate stations (see spatial_baseline.py's module docstring)


def _scores(y_true: np.ndarray, y_pred: np.ndarray, log_scale: bool) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    out = {"n": int(mask.sum()), "R2": float(r2_score(y_true, y_pred)) if mask.sum() >= 2 else float("nan"),
          "spearman": float(spearmanr(y_true, y_pred).correlation) if mask.sum() >= 2 else float("nan")}
    if log_scale and mask.sum() >= 2:
        out["R2_log"] = float(r2_score(np.log1p(np.maximum(y_true, 0)), np.log1p(np.maximum(y_pred, 0))))
    else:
        out["R2_log"] = float("nan")
    return out


def _distance_decile_breakdown(y_true: np.ndarray, y_pred: np.ndarray, nearest_km: np.ndarray) -> list[dict]:
    """Spearman by bucket of distance to the nearest known station -- the honest "how far can this
    reach" picture, not a single blended number."""
    bins = [(0, 5), (5, 15), (15, 50), (50, 200), (200, float("inf"))]
    out = []
    for lo, hi in bins:
        m = (nearest_km >= lo) & (nearest_km < hi) & np.isfinite(y_true) & np.isfinite(y_pred)
        if m.sum() < 20:
            out.append({"km_range": f"{lo}-{hi}", "n": int(m.sum()), "spearman": None})
            continue
        out.append({"km_range": f"{lo}-{hi}", "n": int(m.sum()),
                    "spearman": float(spearmanr(y_true[m], y_pred[m]).correlation)})
    return out


def _duplicate_coordinate_sites(df: pd.DataFrame, threshold_km: float = DUP_KM_THRESHOLD) -> set:
    """Stations whose nearest OTHER station is within threshold_km -- a data-quality caveat (limited
    coordinate precision in places), not proof of leakage (confirmed these are genuinely different,
    differently-named monitoring points), but worth a sensitivity re-score excluding them."""
    sites = df.groupby("site")[["lat", "lon"]].median().reset_index()
    lat, lon = sites["lat"].to_numpy(), sites["lon"].to_numpy()
    dist = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    np.fill_diagonal(dist, np.inf)
    return set(sites.loc[dist.min(axis=1) < threshold_km, "site"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=20)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--production-trials", type=int, default=30)
    args = ap.parse_args()

    df = pd.read_parquet(TABLE)
    summary = {"table": TABLE.name, "folds": args.folds, "trials": args.trials, "k": args.k, "targets": {}}
    tables = []

    (OUT / "models_spatial_knn").mkdir(parents=True, exist_ok=True)
    dup_sites = _duplicate_coordinate_sites(df)
    print(f"{len(dup_sites)} stations have a near-exact (<{DUP_KM_THRESHOLD}km) coordinate twin")

    for target in REAL_TARGET_COLS:
        feats = FEATURE_SETS[target]
        sub = df.dropna(subset=[*feats, target, "lat", "lon"]).reset_index(drop=True)
        if sub["site"].nunique() < args.folds:
            print(f"skip {target}: only {sub['site'].nunique()} stations (< {args.folds} folds)")
            continue
        log_scale = target in LOG_TARGETS
        reg = SpatialKNNRegressor(target=target, k=args.k, feature_cols=feats, inner_n_folds=3)

        oof_knn = reg.predict_oof_knn_only(sub, n_folds=args.folds)
        oof_combo = reg.predict_oof(sub, n_folds=args.folds, n_trials=args.trials)
        y = sub[target].to_numpy()

        knn_scores = _scores(y, oof_knn[target].to_numpy(), log_scale)
        combo_scores = _scores(y, oof_combo[target].to_numpy(), log_scale)
        decile = _distance_decile_breakdown(y, oof_knn[target].to_numpy(), oof_knn["nearest_station_km"].to_numpy())

        not_dup = ~sub["site"].isin(dup_sites)
        sensitivity = _scores(y[not_dup.to_numpy()], oof_combo.loc[not_dup, target].to_numpy(), log_scale)

        print(f"{target}: KNN-alone R2={knn_scores['R2']:.4f} spearman={knn_scores['spearman']:.4f}  |  "
              f"KNN+XGB R2={combo_scores['R2']:.4f} R2log={combo_scores['R2_log']:.4f} "
              f"spearman={combo_scores['spearman']:.4f}")

        summary["targets"][target] = {
            "n": int(len(sub)), "n_stations": int(sub["site"].nunique()),
            "knn_alone": knn_scores, "knn_plus_xgboost": combo_scores,
            "distance_decile_spearman": decile,
            "sensitivity_excluding_duplicate_coordinate_stations": sensitivity,
        }

        for name, scores, validation in (
            ("knn_densification_oof", knn_scores, "station_shuffled_densification"),
            ("knn_xgb_densification_oof", combo_scores, "station_shuffled_densification"),
        ):
            tables.append({"model": name, "target": target, "validation": validation, **scores})

        # Production model: fit on the full table, save for serving (src/data/aoi.py).
        reg_prod = SpatialKNNRegressor(target=target, k=args.k, feature_cols=feats, inner_n_folds=5)
        reg_prod.fit(sub, n_trials=args.production_trials, verbose=False)
        reg_prod.save(OUT / "models_spatial_knn")

    pd.DataFrame(tables).to_csv(OUT / "metrics_table_spatial_knn.csv", index=False)
    (OUT / "spatial_knn_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("wrote", OUT / "metrics_table_spatial_knn.csv", "and", OUT / "spatial_knn_summary.json")


if __name__ == "__main__":
    main()
