"""Sweep k and eps_km for the spatial-KNN densification model (src/models/spatial_baseline.py), per
target, and record the best combination for scripts/train_spatial_baseline.py to use by default.

    python -m scripts.sweep_spatial_knn [--folds 20]

Uses ``predict_oof_knn_only`` (no inner XGBoost retrain needed per grid point, so this is cheap) scored
under ``StationKFold`` -- the correct CV for this model (see src.models.spatial_cv.StationKFold's
docstring for why SpatialKFold would defeat the densification premise being tested here). Writes
reports/real/spatial_knn_sweep.json: the full grid plus the argmax per target (by R2_log for
LOG_TARGETS, else plain R2).

**Known limitation (confirmed empirically)**: this sweep's picks do NOT reliably transfer to the full
combo pipeline once the inner XGBoost has ``nearest_station_km``/``knn_neighbor_std`` as input features
(see ``scripts.train_spatial_baseline._k_eps_for_target``'s docstring for the full story). The sweep
optimizes ``predict_oof_knn_only``'s isolated error, cheaply, without retraining the inner XGBoost -- but
once that model can see distance to the nearest station and the spread of the k neighbours it used, it
can already learn distance-adaptive trust in ``knn_baseline`` on its own. A sweep-optimal tighter k (e.g.
k=3 for bod) just feeds it a noisier ``knn_baseline`` signal and made the deployed combo model *worse*
for every target versus the untuned k=5/eps_km=0.1 defaults. ``train_spatial_baseline.py`` therefore no
longer consults this file by default -- pass ``--use-sweep`` to opt back in. This script and its output
are kept for reference and as a starting point for a future combo-scored sweep (i.e. one that scores the
full ``predict_oof`` combo, not just the KNN-alone proxy), which would be expensive but might recover a
real gain.
"""
import argparse
import json

import pandas as pd

from src.data import nwdp
from src.models.schema import LOG_TARGETS, REAL_TARGET_COLS
from src.models.spatial_baseline import SpatialKNNRegressor
from scripts.train_spatial_baseline import TABLE, _scores

OUT = nwdp.ROOT / "reports" / "real"
K_GRID = [3, 5, 8, 12]
EPS_KM_GRID = [0.05, 0.1, 0.5, 1.0]


def sweep_one_target(sub: pd.DataFrame, target: str, k_grid: list[int], eps_km_grid: list[float],
                     folds: int) -> dict:
    log_scale = target in LOG_TARGETS
    y = sub[target].to_numpy()
    grid = []
    for k in k_grid:
        for eps_km in eps_km_grid:
            reg = SpatialKNNRegressor(target=target, k=k, eps_km=eps_km)
            oof = reg.predict_oof_knn_only(sub, n_folds=folds)
            scores = _scores(y, oof[target].to_numpy(), log_scale)
            grid.append({"k": k, "eps_km": eps_km, **scores})

    metric = "R2_log" if log_scale else "R2"
    best = max(grid, key=lambda row: row[metric] if row[metric] == row[metric] else float("-inf"))
    return {"grid": grid, "best": {"k": best["k"], "eps_km": best["eps_km"], metric: best[metric]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=20)
    args = ap.parse_args()

    df = pd.read_parquet(TABLE)
    result = {"table": TABLE.name, "folds": args.folds, "k_grid": K_GRID, "eps_km_grid": EPS_KM_GRID,
             "targets": {}}

    for target in REAL_TARGET_COLS:
        sub = df.dropna(subset=[target, "lat", "lon"]).reset_index(drop=True)
        if sub["site"].nunique() < args.folds:
            print(f"skip {target}: only {sub['site'].nunique()} stations (< {args.folds} folds)")
            continue
        target_result = sweep_one_target(sub, target, K_GRID, EPS_KM_GRID, args.folds)
        result["targets"][target] = target_result
        best = target_result["best"]
        print(f"{target}: best k={best['k']} eps_km={best['eps_km']} "
              f"({'R2_log' if target in LOG_TARGETS else 'R2'}={best.get('R2_log', best.get('R2')):.4f})")

    (OUT / "spatial_knn_sweep.json").write_text(json.dumps(result, indent=2, default=str))
    print("wrote", OUT / "spatial_knn_sweep.json")


if __name__ == "__main__":
    main()
