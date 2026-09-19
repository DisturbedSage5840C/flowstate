"""
src/models/spatial_cv.py
========================
Spatial K-Fold cross-validation for water quality modelling.

Key design decisions
--------------------
* Group by **site** first: all pixels / observations from the same water body
  must land in the same fold.  Splitting at pixel level leaks spatial
  autocorrelation and inflates R².
* Cluster site centroids with KMeans so geographically close water bodies are
  in the same fold → held-out folds represent *regional* generalisation.
* Returns a list of (train_idx, val_idx) pairs compatible with sklearn /
  XGBoost / PyTorch data loaders.

Usage
-----
    from src.models.spatial_cv import SpatialKFold
    skf = SpatialKFold(n_folds=5, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df)):
        X_train, X_val = df.iloc[train_idx], df.iloc[val_idx]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from typing import Iterator


class SpatialKFold:
    """Assign samples to K geographically-blocked folds.

    Parameters
    ----------
    n_folds : int
        Number of cross-validation folds (default 5).
    random_state : int
        RNG seed for KMeans (reproducibility).
    site_col : str
        Column in df that identifies the water body (default "site").
    lat_col, lon_col : str
        Coordinate columns used for centroid computation.
    """

    def __init__(
        self,
        n_folds: int = 5,
        random_state: int = 42,
        site_col: str = "site",
        lat_col: str = "lat",
        lon_col: str = "lon",
    ):
        self.n_folds = n_folds
        self.random_state = random_state
        self.site_col = site_col
        self.lat_col = lat_col
        self.lon_col = lon_col
        self._fold_map: dict[str, int] = {}  # site → fold assignment

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(
        self, df: pd.DataFrame
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_row_indices, val_row_indices) for each fold.

        Parameters
        ----------
        df : pd.DataFrame
            The full dataset.  Must contain ``site_col``, ``lat_col``,
            ``lon_col`` columns.

        Yields
        ------
        train_idx, val_idx : np.ndarray of integer row positions
        """
        self._assign_folds(df)
        fold_col = df[self.site_col].map(self._fold_map)

        for fold in range(self.n_folds):
            val_mask = fold_col == fold
            train_idx = np.where(~val_mask)[0]
            val_idx = np.where(val_mask)[0]
            if len(val_idx) == 0:
                raise ValueError(
                    f"Fold {fold} has zero validation samples.  "
                    "Reduce n_folds or add more sites."
                )
            yield train_idx, val_idx

    def get_fold_labels(self, df: pd.DataFrame) -> pd.Series:
        """Return a Series of integer fold labels aligned with df's index."""
        self._assign_folds(df)
        return df[self.site_col].map(self._fold_map).rename("fold")

    def fold_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a summary DataFrame: site, water_body_type, fold, n_samples."""
        labels = self.get_fold_labels(df)
        summary = df[[self.site_col, "water_body_type"]].copy()
        summary["fold"] = labels.values
        return (
            summary.groupby([self.site_col, "water_body_type", "fold"])
            .size()
            .reset_index(name="n_samples")
            .sort_values(["fold", self.site_col])
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assign_folds(self, df: pd.DataFrame) -> None:
        """Compute site centroids, cluster with KMeans, assign fold IDs.

        Re-uses existing assignment if already computed for this df shape;
        re-computes if sites have changed.
        """
        sites = df.groupby(self.site_col)[[self.lat_col, self.lon_col]].mean()

        n_sites = len(sites)
        if n_sites < self.n_folds:
            raise ValueError(
                f"n_folds={self.n_folds} > n_sites={n_sites}.  "
                "Reduce n_folds or add more sites."
            )

        kmeans = KMeans(
            n_clusters=self.n_folds,
            random_state=self.random_state,
            n_init="auto",
        )
        cluster_labels = kmeans.fit_predict(sites[[self.lat_col, self.lon_col]].values)
        self._fold_map = dict(zip(sites.index, cluster_labels.tolist()))


class StationKFold:
    """Plain shuffled K-fold over station identity -- deliberately NOT geographically blocked.

    ``SpatialKFold`` clusters geographically close stations into the same fold on purpose, to test
    "regional generalization": can a model work somewhere with zero nearby monitored stations. That is
    the right question for the row-level regressors, and it is why their out-of-fold R^2 sits near zero
    (85-100% of DO/BOD/turbidity's variance is between-station; see src/models/schema.py). It is the
    WRONG question for src.models.spatial_baseline.SpatialKNNRegressor, which asks something different:
    "given the existing ~2,000-station CPCB network stays in place, how well can I estimate an
    unmonitored point near it" -- the real deployment scenario for densifying coverage, not extending
    into an unmonitored region. Under that question, nearby stations SHOULD be available as neighbours
    (median distance to the nearest other station is ~4.7 km); ``SpatialKFold``'s clustering would
    remove exactly the signal being tested, pushing the nearest available neighbour to ~360 km on
    median and making the technique look like it has no skill when it actually does (verified: median
    nearest-neighbour distance under SpatialKFold's 5 clusters is ~358 km vs ~5 km under this class).

    A station never straddles folds (all of its visits move together), same guarantee as SpatialKFold,
    just without the geographic clustering step.
    """

    def __init__(self, n_folds: int = 5, random_state: int = 42, site_col: str = "site"):
        self.n_folds = n_folds
        self.random_state = random_state
        self.site_col = site_col

    def split(self, df: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        sites = df[self.site_col].unique()
        if len(sites) < self.n_folds:
            raise ValueError(f"n_folds={self.n_folds} > n_sites={len(sites)}.")
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        site_col = df[self.site_col].to_numpy()
        for train_site_idx, val_site_idx in kf.split(sites):
            train_sites, val_sites = set(sites[train_site_idx]), set(sites[val_site_idx])
            train_idx = np.where(np.isin(site_col, list(train_sites)))[0]
            val_idx = np.where(np.isin(site_col, list(val_sites)))[0]
            yield train_idx, val_idx


# ---------------------------------------------------------------------------
# Convenience function (mirrors scikit-learn's cross_val_score interface)
# ---------------------------------------------------------------------------

def spatial_cross_val_score(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    df_meta: pd.DataFrame,
    scoring,
    n_folds: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """Run spatial K-fold CV and return an array of per-fold scores.

    Parameters
    ----------
    estimator : sklearn-compatible estimator (fit / predict)
    X : pd.DataFrame of features (same index as df_meta)
    y : pd.Series of targets
    df_meta : pd.DataFrame with at least [site, lat, lon, water_body_type]
    scoring : callable(y_true, y_pred) → float (e.g. sklearn metric)
    n_folds : int
    random_state : int

    Returns
    -------
    np.ndarray of shape (n_folds,)
    """
    skf = SpatialKFold(n_folds=n_folds, random_state=random_state)
    scores = []
    for train_idx, val_idx in skf.split(df_meta):
        est = estimator  # caller is responsible for cloning if needed
        est.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = est.predict(X.iloc[val_idx])
        scores.append(scoring(y.iloc[val_idx], preds))
    return np.array(scores)
