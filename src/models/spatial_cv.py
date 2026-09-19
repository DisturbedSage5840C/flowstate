"""
spatial_cv.py — Spatial k-fold cross-validation wrapper (Navya / P3)

This stub exposes the fold assignment interface Jashan's training script
and dl_model.py depend on. Navya's full Optuna-tuned XGBoost + KMeans-based
spatial fold assignment will replace the body of `assign_spatial_folds`.

Interface contract (do NOT change signatures):
    assign_spatial_folds(df, n_folds, lat_col, lon_col) -> pd.Series
    get_fold_splits(df, n_folds)                         -> Iterator of (train_df, val_df)
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import pandas as pd


def assign_spatial_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    lat_col: str = "lat",
    lon_col: str = "lon",
    site_col: str = "site",
) -> pd.Series:
    """
    Assign a spatial fold index (0..n_folds-1) to each row.

    Strategy: cluster sites by lat/lon using KMeans so that
    geographically adjacent sites land in the same fold —
    preventing spatial autocorrelation leakage.

    Falls back to site-level stratification if sklearn is unavailable.

    Returns
    -------
    pd.Series of int, same index as df, values in [0, n_folds-1].
    """
    try:
        from sklearn.cluster import KMeans

        # Cluster on unique site centroids
        site_coords = (
            df.groupby(site_col)[[lat_col, lon_col]]
            .mean()
            .reset_index()
        )
        coords = site_coords[[lat_col, lon_col]].values

        n_clusters = min(n_folds, len(site_coords))
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        site_coords["fold"] = km.fit_predict(coords) % n_folds

        fold_map = dict(zip(site_coords[site_col], site_coords["fold"]))
        return df[site_col].map(fold_map).fillna(0).astype(int)

    except ImportError:
        # Fallback: assign folds round-robin by unique site
        sites = df[site_col].unique()
        fold_map = {s: i % n_folds for i, s in enumerate(sorted(sites))}
        return df[site_col].map(fold_map).fillna(0).astype(int)


def get_fold_splits(
    df: pd.DataFrame,
    n_folds: int = 5,
    lat_col: str = "lat",
    lon_col: str = "lon",
    site_col: str = "site",
) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Yield (train_df, val_df) for each spatial fold.

    Usage
    -----
    for fold_idx, (train_df, val_df) in enumerate(get_fold_splits(df)):
        model.fit(train_df, val_df)
    """
    folds = assign_spatial_folds(df, n_folds, lat_col, lon_col, site_col)
    df = df.copy()
    df["_fold"] = folds

    for fold_idx in range(n_folds):
        val_mask   = df["_fold"] == fold_idx
        train_mask = ~val_mask
        yield df[train_mask].drop(columns=["_fold"]), df[val_mask].drop(columns=["_fold"])
