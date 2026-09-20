"""
src/models/spatial_baseline.py
===============================
Distance-weighted nearest-neighbour "densification" regression: given the existing ~2,000-station CPCB
network stays in place, how well can water quality be estimated at a nearby unmonitored point?

This is a genuinely different question from the row-level XGBoost regressors in xgboost_pipeline.py,
which are evaluated with SpatialKFold (geographically blocked at the regional level, see spatial_cv.py)
to test "can this work somewhere with zero nearby monitored stations" -- and honestly answer close to
R^2 0 (85-100% of DO/BOD/turbidity's variance is between-station; see schema.py's FEATURE_SETS docstring).

Under StationKFold (a station is held out entirely, but the rest of the network stays available -- the
real "densify existing coverage" deployment scenario, not "cover an unmonitored region"), a distance-
weighted median of nearby OTHER stations' known values is a strong baseline on its own, and combining it
with the existing satellite/context features (schema.py's FEATURE_SETS) as an extra feature improves
further. Measured on train_real_large.parquet (see reports/real/spatial_knn_summary.json for the exact,
regenerated numbers):

    target      | KNN alone R^2 | KNN alone Spearman | KNN + XGBoost R^2 | R^2(log)
    do          | 0.396         | 0.650               | 0.398              | 0.320
    bod         | -0.022        | 0.650               | 0.115              | 0.480
    turbidity   | 0.040         | 0.615               | 0.133              | 0.431

Skill degrades with distance to the nearest known station (BOD Spearman 0.70 at <5km -> 0.42 at
50-200km) -- real spatial autocorrelation (CPCB densely monitors many river reaches/urban lake systems
at several points; median distance between two stations is ~4.7km), not an artifact, but it means this
model's accuracy is CONDITIONAL on being near an already-monitored station. It is not a replacement for
the SpatialKFold "regional generalization" numbers, which stay the honest answer for "zero nearby CPCB
coverage" -- both are reported, never merged (see StationKFold's docstring in spatial_cv.py).

Known data-quality caveat: 83 of 2,121 stations (~4%) share a near-exact (<0.1km) coordinate with a
differently-named station -- confirmed to be genuinely distinct monitoring points (e.g. two different
river-name entries at a confluence), not a leakage artifact, but a reminder that station coordinates in
this dataset have limited precision in places.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.city_proximity import haversine_km
from src.models.spatial_cv import StationKFold
from src.models.xgboost_pipeline import WaterQualityXGB

KNN_FEATURE_COL = "knn_baseline"
NEAREST_KM_FEATURE_COL = "nearest_station_km"
NEIGHBOR_STD_FEATURE_COL = "knn_neighbor_std"
DEFAULT_EPS_KM = 0.1


def station_level_lookup(df: pd.DataFrame, target: str, site_col: str = "site",
                         lat_col: str = "lat", lon_col: str = "lon") -> pd.DataFrame:
    """One row per station: its median lat/lon and median ``target`` value.

    Rows without a labelled ``target`` are dropped before aggregating, so a station's lookup value is
    never NaN; a station with no labelled visits simply does not appear in the lookup.
    """
    labelled = df[df[target].notna()]
    return (labelled.groupby(site_col)
            .agg(**{lat_col: (lat_col, "median"), lon_col: (lon_col, "median"), target: (target, "median")})
            .reset_index())


def knn_predict(query_lat, query_lon, ref_lat, ref_lon, ref_value, k: int = 5,
                eps_km: float = DEFAULT_EPS_KM, exclude_idx: Optional[np.ndarray] = None,
                return_std: bool = False):
    """Inverse-distance-weighted average of the ``k`` nearest reference points' values, per query point.

    Returns ``(prediction, nearest_km)``; ``nearest_km`` is the distance to the single closest reference
    point used, a confidence signal (skill degrades with distance -- see module docstring). NaN
    predictions/distances where no reference point is available for a query.

    ``exclude_idx`` (optional, aligned with the query points): for each query point, the index into the
    reference arrays to exclude before searching for neighbours (e.g. the query's own station, so a
    training row never sees its own already-known value as a "neighbour" at distance 0 -- the standard
    leave-one-out construction for a meta-feature). Pass -1 for "nothing to exclude" at that position.

    ``return_std`` (optional): when True, also returns the inverse-distance-weighted standard deviation
    of the ``k`` neighbour values actually used, as a third array -- a second, distance-independent
    confidence signal (a locally heterogeneous neighbourhood is a reason to trust the prediction less,
    even at a fixed distance). Default False keeps the original 2-tuple return unchanged.
    """
    query_lat = np.atleast_1d(np.asarray(query_lat, dtype=np.float64))
    query_lon = np.atleast_1d(np.asarray(query_lon, dtype=np.float64))
    ref_lat = np.asarray(ref_lat, dtype=np.float64)
    ref_lon = np.asarray(ref_lon, dtype=np.float64)
    ref_value = np.asarray(ref_value, dtype=np.float64)
    n_ref = len(ref_lat)
    if n_ref == 0:
        nan = np.full(len(query_lat), np.nan)
        return (nan, nan, nan) if return_std else (nan, nan)

    dist = haversine_km(query_lat[:, None], query_lon[:, None], ref_lat[None, :], ref_lon[None, :])
    n_excluded_per_row = 0
    if exclude_idx is not None:
        exclude_idx = np.asarray(exclude_idx)
        rows = np.where(exclude_idx >= 0)[0]
        dist[rows, exclude_idx[rows]] = np.inf
        n_excluded_per_row = 1

    k_eff = max(1, min(k, n_ref - n_excluded_per_row))
    order = np.argsort(dist, axis=1)[:, :k_eff]
    nearest_km = dist[np.arange(len(query_lat)), order[:, 0]]
    d_k = np.take_along_axis(dist, order, axis=1)
    v_k = ref_value[order]
    w = 1.0 / (eps_km + d_k)
    with np.errstate(invalid="ignore"):
        pred = (w * v_k).sum(axis=1) / w.sum(axis=1)
    nearest_km = np.where(np.isinf(nearest_km), np.nan, nearest_km)
    pred = np.where(np.isfinite(nearest_km), pred, np.nan)
    if not return_std:
        return pred, nearest_km
    with np.errstate(invalid="ignore"):
        var = (w * (v_k - pred[:, None]) ** 2).sum(axis=1) / w.sum(axis=1)
        std = np.sqrt(var)
    std = np.where(np.isfinite(nearest_km), std, np.nan)
    return pred, nearest_km, std


class SpatialKNNRegressor:
    """Distance-weighted nearest-known-station baseline, combined with satellite/context features via
    an inner WaterQualityXGB, for a single target. See the module docstring for what this answers and
    its distance-conditional caveat.

    Parameters
    ----------
    target : str
        The single target column this instance predicts (e.g. "bod").
    k : int
        Number of nearest stations averaged for the KNN baseline feature.
    feature_cols : list[str] | None
        Satellite/context feature columns combined with the KNN baseline (typically
        src.models.schema.FEATURE_SETS[target]); default [] means KNN-baseline-alone.
    log_target : bool | None
        Force log1p modelling on/off for the inner XGBoost; None defers to WaterQualityXGB's own
        LOG_TARGETS default.
    inner_n_folds : int
        n_folds passed to the inner WaterQualityXGB (its own Optuna-tuning CV) -- kept small by default
        since this class's own predict_oof already does an outer CV loop (avoids very slow nested CV).
    eps_km : float
        Inverse-distance-weighting smoothing term passed through to every ``knn_predict`` call (was
        previously hardcoded to ``DEFAULT_EPS_KM`` regardless of what a caller wanted).
    sample_weight_col : str | None
        Forwarded to the inner WaterQualityXGB's own ``sample_weight_col`` (e.g. "n_water_px").
    include_distance_features : bool
        When True (default), ``nearest_station_km`` and ``knn_neighbor_std`` are added as input features
        to the inner XGBoost, not just attached to prediction output -- lets the tree learn to trust
        ``knn_baseline`` less when the nearest station is far away or neighbours disagree (see module
        docstring's distance-decile skill decay). Set False to fall back to the original behaviour.
    """

    def __init__(self, target: str, k: int = 5, feature_cols: Optional[list[str]] = None,
                log_target: Optional[bool] = None, inner_n_folds: int = 3, random_state: int = 42,
                eps_km: float = DEFAULT_EPS_KM, sample_weight_col: Optional[str] = None,
                include_distance_features: bool = True):
        self.target = target
        self.k = k
        self.feature_cols = list(feature_cols or [])
        self.log_target = log_target
        self.inner_n_folds = inner_n_folds
        self.random_state = random_state
        self.eps_km = eps_km
        self.sample_weight_col = sample_weight_col
        self.include_distance_features = include_distance_features
        self._lookup: Optional[pd.DataFrame] = None
        self._inner: Optional[WaterQualityXGB] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_inner(self) -> WaterQualityXGB:
        extra = [NEAREST_KM_FEATURE_COL, NEIGHBOR_STD_FEATURE_COL] if self.include_distance_features else []
        kwargs = dict(targets=[self.target], n_folds=self.inner_n_folds, random_state=self.random_state,
                     feature_cols={self.target: [KNN_FEATURE_COL, *extra, *self.feature_cols]},
                     sample_weight_col=self.sample_weight_col)
        if self.log_target is not None:
            kwargs["log_targets"] = (self.target,) if self.log_target else ()
        return WaterQualityXGB(**kwargs)

    def _loo_meta_features(self, df: pd.DataFrame, lookup: pd.DataFrame
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Leave-one-station-out KNN prediction, nearest-distance and neighbour-std for every row of
        ``df`` against ``lookup``."""
        site_to_col = {s: i for i, s in enumerate(lookup["site"].to_numpy())}
        own_col = df["site"].map(site_to_col).fillna(-1).to_numpy().astype(int)
        return knn_predict(df["lat"].to_numpy(), df["lon"].to_numpy(), lookup["lat"].to_numpy(),
                           lookup["lon"].to_numpy(), lookup[self.target].to_numpy(),
                           k=self.k, eps_km=self.eps_km, exclude_idx=own_col, return_std=True)

    def _loo_feature(self, df: pd.DataFrame, lookup: pd.DataFrame) -> np.ndarray:
        """Leave-one-station-out KNN feature for every row of ``df`` against ``lookup``."""
        pred, _, _ = self._loo_meta_features(df, lookup)
        return pred

    def _attach_meta_features(self, frame: pd.DataFrame, pred: np.ndarray, nearest_km: np.ndarray,
                              std: np.ndarray) -> None:
        """Set knn_baseline (+ nearest_station_km/knn_neighbor_std if enabled) on ``frame`` in place."""
        frame[KNN_FEATURE_COL] = pred
        if self.include_distance_features:
            frame[NEAREST_KM_FEATURE_COL] = nearest_km
            frame[NEIGHBOR_STD_FEATURE_COL] = std

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, n_trials: int = 30, verbose: bool = False) -> "SpatialKNNRegressor":
        """Fit the production model: the full-table station lookup, plus an inner XGBoost trained with
        a leave-one-station-out KNN feature (so training never trivially sees "your own value")."""
        self._lookup = station_level_lookup(df, self.target)
        labelled = df[df[self.target].notna()].copy()
        pred, nearest_km, std = self._loo_meta_features(labelled, self._lookup)
        self._attach_meta_features(labelled, pred, nearest_km, std)
        self._inner = self._make_inner()
        self._inner.train(labelled, n_trials=n_trials, verbose=verbose)
        return self

    def nearest_known_station_km(self, lat: float, lon: float) -> float:
        """Distance (km) from an arbitrary point to the nearest station in the fitted lookup -- the
        confidence signal this model's accuracy is conditional on (see module docstring)."""
        if self._lookup is None:
            raise RuntimeError("No model fitted yet. Call .fit() first.")
        _, nearest_km = knn_predict([lat], [lon], self._lookup["lat"].to_numpy(), self._lookup["lon"].to_numpy(),
                                    self._lookup[self.target].to_numpy(), k=1, eps_km=self.eps_km)
        return float(nearest_km[0])

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Serving-time prediction against the FULL fitted lookup (a genuinely new query point is never
        itself in the lookup, so no leave-one-out is needed here). Adds ``nearest_station_km``."""
        if self._inner is None or self._lookup is None:
            raise RuntimeError("No model fitted yet. Call .fit() first.")
        feats = df.copy()
        pred, nearest_km, std = knn_predict(feats["lat"].to_numpy(), feats["lon"].to_numpy(),
                                            self._lookup["lat"].to_numpy(), self._lookup["lon"].to_numpy(),
                                            self._lookup[self.target].to_numpy(), k=self.k,
                                            eps_km=self.eps_km, return_std=True)
        self._attach_meta_features(feats, pred, nearest_km, std)
        out = self._inner.predict(feats)
        out["nearest_station_km"] = nearest_km
        return out

    def predict_oof(self, df: pd.DataFrame, n_folds: int = 20, n_trials: int = 5) -> pd.DataFrame:
        """Honest out-of-fold predictions via StationKFold: a held-out station is wholly absent from the
        reference pool used to predict it (not merely down-weighted by distance), and every training
        row's own KNN feature excludes its own station too. Returns a frame aligned with the labelled
        subset of ``df`` (NaN target rows dropped, index reset), columns [target, "nearest_station_km"].
        """
        labelled = df[df[self.target].notna()].reset_index(drop=True)
        skf = StationKFold(n_folds=n_folds, random_state=self.random_state)
        oof = np.full(len(labelled), np.nan)
        oof_nearest_km = np.full(len(labelled), np.nan)
        for train_idx, val_idx in skf.split(labelled):
            train_df = labelled.iloc[train_idx]
            lookup = station_level_lookup(train_df, self.target)

            train_feat = train_df.copy()
            pred, nearest_km_train, std = self._loo_meta_features(train_feat, lookup)
            self._attach_meta_features(train_feat, pred, nearest_km_train, std)
            inner = self._make_inner()
            inner.train(train_feat, n_trials=n_trials, verbose=False)

            val_df = labelled.iloc[val_idx].copy()
            pred_knn, nearest_km, std_val = knn_predict(val_df["lat"].to_numpy(), val_df["lon"].to_numpy(),
                                                        lookup["lat"].to_numpy(), lookup["lon"].to_numpy(),
                                                        lookup[self.target].to_numpy(), k=self.k,
                                                        eps_km=self.eps_km, return_std=True)
            self._attach_meta_features(val_df, pred_knn, nearest_km, std_val)
            preds = inner.predict(val_df)
            oof[val_idx] = preds[self.target].to_numpy()
            oof_nearest_km[val_idx] = nearest_km
        return pd.DataFrame({self.target: oof, "nearest_station_km": oof_nearest_km}, index=labelled.index)

    def predict_oof_knn_only(self, df: pd.DataFrame, n_folds: int = 20) -> pd.DataFrame:
        """Same StationKFold loop as predict_oof, but the KNN baseline alone (no XGBoost combination) --
        the honest number for "what does spatial interpolation get you with zero satellite input"."""
        labelled = df[df[self.target].notna()].reset_index(drop=True)
        skf = StationKFold(n_folds=n_folds, random_state=self.random_state)
        oof = np.full(len(labelled), np.nan)
        oof_nearest_km = np.full(len(labelled), np.nan)
        for train_idx, val_idx in skf.split(labelled):
            lookup = station_level_lookup(labelled.iloc[train_idx], self.target)
            val_df = labelled.iloc[val_idx]
            pred, nearest_km = knn_predict(val_df["lat"].to_numpy(), val_df["lon"].to_numpy(),
                                           lookup["lat"].to_numpy(), lookup["lon"].to_numpy(),
                                           lookup[self.target].to_numpy(), k=self.k, eps_km=self.eps_km)
            oof[val_idx] = pred
            oof_nearest_km[val_idx] = nearest_km
        return pd.DataFrame({self.target: oof, "nearest_station_km": oof_nearest_km}, index=labelled.index)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model_dir: str | Path) -> None:
        """Saves under ``<model_dir>/<target>/`` -- the inner WaterQualityXGB writes a single shared
        ``config.json`` per directory (it's designed for one multi-target instance, see
        xgboost_pipeline.py), so multiple single-target SpatialKNNRegressors must each get their own
        subdirectory or they would silently overwrite each other's config."""
        if self._inner is None or self._lookup is None:
            raise RuntimeError("Nothing fitted yet. Call .fit() first.")
        target_dir = Path(model_dir) / self.target
        target_dir.mkdir(parents=True, exist_ok=True)
        self._lookup.to_parquet(target_dir / "lookup.parquet")
        self._inner.save(target_dir)
        with open(target_dir / "spatial_config.json", "w") as f:
            json.dump({"target": self.target, "k": self.k, "feature_cols": self.feature_cols,
                      "eps_km": self.eps_km, "sample_weight_col": self.sample_weight_col,
                      "include_distance_features": self.include_distance_features}, f, indent=2)

    @classmethod
    def load(cls, model_dir: str | Path, target: str) -> "SpatialKNNRegressor":
        target_dir = Path(model_dir) / target
        cfg = json.loads((target_dir / "spatial_config.json").read_text())
        instance = cls(target=cfg["target"], k=cfg["k"], feature_cols=cfg["feature_cols"],
                       eps_km=cfg.get("eps_km", DEFAULT_EPS_KM),
                       sample_weight_col=cfg.get("sample_weight_col"),
                       include_distance_features=cfg.get("include_distance_features", True))
        instance._lookup = pd.read_parquet(target_dir / "lookup.parquet")
        instance._inner = WaterQualityXGB.load(target_dir, targets=[target])
        return instance
