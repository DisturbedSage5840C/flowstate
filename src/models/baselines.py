"""Baselines that any real model must beat, evaluated on the same spatial folds."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.spatial_cv import SpatialKFold


def _constant_oof(df: pd.DataFrame, targets: list[str], reducer, n_folds: int, random_state: int) -> pd.DataFrame:
    splits = list(SpatialKFold(n_folds=n_folds, random_state=random_state).split(df))
    out = {t: np.full(len(df), np.nan) for t in targets}
    for t in targets:
        y = df[t].to_numpy(dtype=float)
        for train_idx, val_idx in splits:
            train_y = y[train_idx]
            train_y = train_y[np.isfinite(train_y)]
            if len(train_y):
                out[t][val_idx] = reducer(train_y)
    return pd.DataFrame(out, index=df.index)


def mean_baseline_oof(df: pd.DataFrame, targets: list[str], n_folds: int = 5, random_state: int = 42) -> pd.DataFrame:
    """Out-of-fold predictions of "predict the training-fold mean" (R^2 <= ~0 by construction)."""
    return _constant_oof(df, targets, np.mean, n_folds, random_state)


def median_baseline_oof(df: pd.DataFrame, targets: list[str], n_folds: int = 5, random_state: int = 42) -> pd.DataFrame:
    """Same with the median (a stronger baseline for skewed targets such as BOD and turbidity)."""
    return _constant_oof(df, targets, np.median, n_folds, random_state)
