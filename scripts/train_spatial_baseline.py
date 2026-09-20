"""Train + evaluate the spatial-KNN "densification" regressor: given the existing CPCB network stays in
place, how well can DO/BOD/turbidity be estimated at a nearby unmonitored point?

    python -m scripts.train_spatial_baseline [--folds 20] [--trials 5] [--k 5] [--eps-km 0.1]

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
from src.models.schema import FEATURE_SETS, LAND_COVER_COLS, LOG_TARGETS, REAL_TARGET_COLS
from src.models.spatial_baseline import DEFAULT_EPS_KM, SpatialKNNRegressor

TABLE_LARGE = nwdp.ROOT / "data" / "processed" / "train_real_large.parquet"
TABLE_SMALL = nwdp.ROOT / "data" / "processed" / "train_real.parquet"
TABLE = TABLE_LARGE if TABLE_LARGE.exists() else TABLE_SMALL
OUT = nwdp.ROOT / "reports" / "real"
DUP_KM_THRESHOLD = 0.1    # near-exact-coordinate stations (see spatial_baseline.py's module docstring)
SWEEP_FILE = OUT / "spatial_knn_sweep.json"
FALLBACK_K, FALLBACK_EPS_KM = 5, DEFAULT_EPS_KM


def _k_eps_for_target(target: str, k_override: int | None, eps_km_override: float | None,
                      use_sweep: bool = False) -> tuple[int, float]:
    """Per-target k/eps_km: explicit CLI overrides win. Otherwise defaults to k=5, eps_km=0.1 for every
    target -- NOT scripts.sweep_spatial_knn's picks, despite the sweep infrastructure below still being
    able to read them with --use-sweep. Confirmed empirically this session: the sweep picks k/eps_km by
    minimizing predict_oof_knn_only's error in isolation (cheap, no XGBoost retrain), but once the inner
    XGBoost has nearest_station_km/knn_neighbor_std as input features it can already learn
    distance-adaptive trust in knn_baseline itself -- so a tighter, sweep-optimal k (e.g. k=3 for bod)
    just feeds it a noisier knn_baseline and made the deployed combo model WORSE for every target
    (bod combo R2_log 0.475 -> 0.443, do combo R2 0.417 -> 0.416, turbidity combo R2_log 0.469 -> 0.465)
    versus plain k=5/eps_km=0.1. See reports/real/spatial_knn_sweep.json for the raw KNN-alone-only grid
    (kept for reference/future work on a combo-scored sweep) -- it is no longer consulted by default."""
    if k_override is not None and eps_km_override is not None:
        return k_override, eps_km_override
    sweep_best = None
    if use_sweep and SWEEP_FILE.exists():
        sweep = json.loads(SWEEP_FILE.read_text())
        sweep_best = sweep.get("targets", {}).get(target, {}).get("best")
        if sweep_best is None:
            print(f"warning: --use-sweep given but no sweep entry for target={target} in {SWEEP_FILE} "
                  f"-- falling back to k={FALLBACK_K}, eps_km={FALLBACK_EPS_KM}.")
    if sweep_best is None:
        sweep_best = {"k": FALLBACK_K, "eps_km": FALLBACK_EPS_KM}
    k = k_override if k_override is not None else sweep_best["k"]
    eps_km = eps_km_override if eps_km_override is not None else sweep_best["eps_km"]
    return k, eps_km


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


def spatial_n_water_px_ablation(sub: pd.DataFrame, target: str, feats: list[str], folds: int, n_trials: int,
                                k: int, eps_km: float) -> dict:
    """Does weighting the inner XGBoost's training rows by n_water_px help the spatial-KNN combo model
    specifically? Re-tested here (not just on the row-level regressors in scripts.train_real_models,
    where it came back flat/mixed) because this model trains on a smaller, labelled-only table with an
    extra dominant feature (knn_baseline) -- a different enough setting to be worth a cheap re-check.
    Weighted vs unweighted predict_oof, same folds, reported honestly -- adopt only if it helps."""
    if "n_water_px" not in sub.columns:
        return {"skill": "n_water_px column not present"}
    log_scale = target in LOG_TARGETS
    y = sub[target].to_numpy()
    out = {}
    for weighted in (False, True):
        reg = SpatialKNNRegressor(target=target, k=k, eps_km=eps_km, feature_cols=feats, inner_n_folds=3,
                                  sample_weight_col="n_water_px" if weighted else None)
        oof = reg.predict_oof(sub, n_folds=folds, n_trials=n_trials)
        out["weighted" if weighted else "unweighted"] = _scores(y, oof[target].to_numpy(), log_scale)
    return out


def spatial_land_cover_ablation(df: pd.DataFrame, target: str, feats: list[str], folds: int, n_trials: int,
                                k: int, eps_km: float) -> dict:
    """Does adding ESA WorldCover cropland/built/tree fractions help the spatial-KNN combo model for this
    target, on top of its current best feats? With vs without, using SpatialKNNRegressor's own
    StationKFold predict_oof loop (NOT SpatialKFold -- see spatial_cv.py's docstring: SpatialKFold pushes
    the nearest neighbour ~358km away for this model, defeating the densification premise it tests).
    Re-drops NaN rows against the land-cover columns so both arms compare on identical rows -- coverage
    is 100% (src/data/land_cover.py), so this should not shrink n in practice."""
    if not all(c in df.columns for c in LAND_COVER_COLS):
        return {"skill": "land cover columns not present"}
    log_scale = target in LOG_TARGETS
    out = {}
    for with_lc in (False, True):
        cols = feats + LAND_COVER_COLS if with_lc else feats
        sub = df.dropna(subset=[*cols, target, "lat", "lon"]).reset_index(drop=True)
        y = sub[target].to_numpy()
        reg = SpatialKNNRegressor(target=target, k=k, eps_km=eps_km, feature_cols=cols, inner_n_folds=3)
        oof = reg.predict_oof(sub, n_folds=folds, n_trials=n_trials)
        out["with_land_cover" if with_lc else "without_land_cover"] = {
            "n": int(len(sub)), **_scores(y, oof[target].to_numpy(), log_scale)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=20)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--k", type=int, default=None, help="override k for every target; default is 5")
    ap.add_argument("--eps-km", type=float, default=None, help="override eps_km for every target; default is 0.1")
    ap.add_argument("--use-sweep", action="store_true",
                    help="use scripts.sweep_spatial_knn's per-target best k/eps_km instead of the "
                         "k=5/eps_km=0.1 default -- NOT recommended, see _k_eps_for_target's docstring "
                         "for why the sweep's picks made the deployed combo model worse for every target")
    ap.add_argument("--production-trials", type=int, default=30)
    args = ap.parse_args()

    df = pd.read_parquet(TABLE)
    summary = {"table": TABLE.name, "folds": args.folds, "trials": args.trials, "targets": {},
              "n_water_px_ablation": {}}
    land_cover_ablation = {}
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
        k, eps_km = _k_eps_for_target(target, args.k, args.eps_km, use_sweep=args.use_sweep)
        reg = SpatialKNNRegressor(target=target, k=k, eps_km=eps_km, feature_cols=feats, inner_n_folds=3)

        oof_knn = reg.predict_oof_knn_only(sub, n_folds=args.folds)
        oof_combo = reg.predict_oof(sub, n_folds=args.folds, n_trials=args.trials)
        y = sub[target].to_numpy()

        knn_scores = _scores(y, oof_knn[target].to_numpy(), log_scale)
        combo_scores = _scores(y, oof_combo[target].to_numpy(), log_scale)
        decile = _distance_decile_breakdown(y, oof_knn[target].to_numpy(), oof_knn["nearest_station_km"].to_numpy())

        not_dup = ~sub["site"].isin(dup_sites)
        sensitivity = _scores(y[not_dup.to_numpy()], oof_combo.loc[not_dup, target].to_numpy(), log_scale)

        water_px_ablation = spatial_n_water_px_ablation(sub, target, feats, args.folds, min(args.trials, 15),
                                                        k, eps_km)
        summary["n_water_px_ablation"][target] = water_px_ablation
        land_cover_ablation[target] = spatial_land_cover_ablation(df, target, feats, args.folds,
                                                                   min(args.trials, 15), k, eps_km)

        print(f"{target}: KNN-alone R2={knn_scores['R2']:.4f} spearman={knn_scores['spearman']:.4f}  |  "
              f"KNN+XGB R2={combo_scores['R2']:.4f} R2log={combo_scores['R2_log']:.4f} "
              f"spearman={combo_scores['spearman']:.4f}")

        summary["targets"][target] = {
            "n": int(len(sub)), "n_stations": int(sub["site"].nunique()), "k": k, "eps_km": eps_km,
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
        reg_prod = SpatialKNNRegressor(target=target, k=k, eps_km=eps_km, feature_cols=feats, inner_n_folds=5)
        reg_prod.fit(sub, n_trials=args.production_trials, verbose=False)
        reg_prod.save(OUT / "models_spatial_knn")

    pd.DataFrame(tables).to_csv(OUT / "metrics_table_spatial_knn.csv", index=False)
    (OUT / "spatial_knn_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("wrote", OUT / "metrics_table_spatial_knn.csv", "and", OUT / "spatial_knn_summary.json")

    lc_file = OUT / "land_cover_ablation.json"
    combined = json.loads(lc_file.read_text()) if lc_file.exists() else {}
    combined["spatial_knn"] = land_cover_ablation
    lc_file.write_text(json.dumps(combined, indent=2, default=str))
    print("wrote", lc_file)


if __name__ == "__main__":
    main()
