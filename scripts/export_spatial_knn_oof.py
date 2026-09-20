"""Export the spatial-KNN model's out-of-fold predictions per (station, visit), for the UI/API.

    python -m scripts.export_spatial_knn_oof [--folds 20] [--trials 5]

Same rows, folds (StationKFold), k/eps_km and inner-model settings as scripts.train_spatial_baseline, so the
numbers reproduce that script's headline scores (checked and printed below). Every prediction is
HELD-OUT: the station's own values are absent from the neighbour pool and from the inner model's training
rows, so ``<target>_est`` is what the model would have said had the station not been monitored, next to the
measured value. ``<target>_nearest_km`` is the distance to the closest other station used, the confidence
signal this model's skill depends on.

Writes reports/real/spatial_knn_oof.parquet (columns: site, date, then per target ``<t>_est`` and
``<t>_nearest_km``).
"""
import argparse

import pandas as pd

from scripts.train_spatial_baseline import TABLE, _k_eps_for_target, _scores
from src.data import nwdp
from src.models.schema import FEATURE_SETS, LOG_TARGETS, REAL_TARGET_COLS
from src.models.spatial_baseline import SpatialKNNRegressor

OUT = nwdp.ROOT / "reports" / "real" / "spatial_knn_oof.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=20)
    ap.add_argument("--trials", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_parquet(TABLE)
    merged = None
    for target in REAL_TARGET_COLS:
        feats = FEATURE_SETS[target]
        sub = df.dropna(subset=[*feats, target, "lat", "lon"]).reset_index(drop=True)
        k, eps_km = _k_eps_for_target(target, None, None, use_sweep=False)
        reg = SpatialKNNRegressor(target=target, k=k, eps_km=eps_km, feature_cols=feats, inner_n_folds=3)
        oof = reg.predict_oof(sub, n_folds=args.folds, n_trials=args.trials)
        scores = _scores(sub[target].to_numpy(), oof[target].to_numpy(), target in LOG_TARGETS)
        print(f"{target}: R2={scores['R2']:.4f} R2_log={scores['R2_log']:.4f} spearman={scores['spearman']:.4f}")
        part = sub[["site", "date"]].copy()
        part[f"{target}_est"] = oof[target].to_numpy()
        part[f"{target}_nearest_km"] = oof["nearest_station_km"].to_numpy()
        merged = part if merged is None else merged.merge(part, on=["site", "date"], how="outer")
    merged.to_parquet(OUT, index=False)
    print("wrote", OUT, merged.shape)


if __name__ == "__main__":
    main()
