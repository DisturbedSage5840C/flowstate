"""tests/test_spatial_baseline.py — unit tests for src/models/spatial_baseline.py (the spatial-KNN
"densification" regressor: distance-weighted nearest-known-station baseline + satellite features)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.spatial_baseline import (
    KNN_FEATURE_COL,
    SpatialKNNRegressor,
    knn_predict,
    station_level_lookup,
)


# ---------------------------------------------------------------------------
# knn_predict: weighted-average arithmetic
# ---------------------------------------------------------------------------

def test_knn_predict_known_weighted_average():
    # Two reference points on the same meridian, 0km and ~111km away (1 degree of latitude); k=2.
    pred, nearest = knn_predict(
        query_lat=[0.0], query_lon=[0.0],
        ref_lat=[0.0, 1.0], ref_lon=[0.0, 0.0],
        ref_value=[10.0, 20.0], k=2, eps_km=0.1,
    )
    # The 0km reference dominates almost entirely (weight 1/0.1 vs 1/~111.2).
    assert nearest[0] == pytest.approx(0.0, abs=1e-6)
    assert pred[0] == pytest.approx(10.0, abs=0.1)


def test_knn_predict_k_larger_than_available_references():
    pred, nearest = knn_predict([0.0], [0.0], [0.0, 1.0], [0.0, 0.0], [5.0, 15.0], k=10)
    assert np.isfinite(pred[0]) and np.isfinite(nearest[0])


def test_knn_predict_empty_reference_returns_nan():
    pred, nearest = knn_predict([0.0], [0.0], [], [], [], k=5)
    assert np.isnan(pred[0]) and np.isnan(nearest[0])


def test_knn_predict_exclude_idx_removes_the_named_reference():
    # Without exclusion, the 0km-away point (index 0) dominates; with it excluded, must use index 1.
    ref_lat, ref_lon, ref_value = [0.0, 1.0], [0.0, 0.0], [10.0, 20.0]
    pred_no_excl, near_no_excl = knn_predict([0.0], [0.0], ref_lat, ref_lon, ref_value, k=1)
    pred_excl, near_excl = knn_predict([0.0], [0.0], ref_lat, ref_lon, ref_value, k=1, exclude_idx=[0])
    assert pred_no_excl[0] == pytest.approx(10.0, abs=0.1)
    assert pred_excl[0] == pytest.approx(20.0, abs=0.1)
    assert near_excl[0] > near_no_excl[0]


def test_knn_predict_exclude_idx_minus_one_means_nothing_excluded():
    pred_a, _ = knn_predict([0.0], [0.0], [0.0, 1.0], [0.0, 0.0], [10.0, 20.0], k=1)
    pred_b, _ = knn_predict([0.0], [0.0], [0.0, 1.0], [0.0, 0.0], [10.0, 20.0], k=1, exclude_idx=[-1])
    assert pred_a[0] == pytest.approx(pred_b[0])


# ---------------------------------------------------------------------------
# station_level_lookup
# ---------------------------------------------------------------------------

def test_station_level_lookup_aggregates_median_and_drops_unlabelled_rows():
    df = pd.DataFrame({
        "site": ["a", "a", "a", "b", "b"],
        "lat": [10.0, 10.001, 10.002, 20.0, 20.001],
        "lon": [70.0, 70.001, 70.002, 80.0, 80.001],
        "bod": [1.0, 3.0, np.nan, 5.0, 5.0],
    })
    lookup = station_level_lookup(df, "bod")
    assert set(lookup["site"]) == {"a", "b"}
    assert lookup.set_index("site").loc["a", "bod"] == pytest.approx(2.0)   # median of [1.0, 3.0], NaN dropped
    assert lookup.set_index("site").loc["b", "bod"] == pytest.approx(5.0)


def test_station_level_lookup_excludes_a_station_with_no_labelled_visits():
    df = pd.DataFrame({"site": ["a", "b"], "lat": [1.0, 2.0], "lon": [1.0, 2.0], "bod": [np.nan, 5.0]})
    lookup = station_level_lookup(df, "bod")
    assert set(lookup["site"]) == {"b"}


# ---------------------------------------------------------------------------
# Fixtures for SpatialKNNRegressor
# ---------------------------------------------------------------------------

def _clustered_df(n_sites=30, per_site=4, seed=0, spread_km=0.3):
    """Stations scattered across a wide area, each with its own "true" pollution level; a few visits
    per station with small measurement noise. spread_km controls how tightly stations cluster
    geographically (smaller = a denser, more realistic CPCB-like network)."""
    rng = np.random.default_rng(seed)
    rows = []
    deg_per_km = 1.0 / 111.0
    for i in range(n_sites):
        lat0 = rng.uniform(10.0, 30.0)
        lon0 = rng.uniform(70.0, 90.0)
        signal = rng.uniform(-3, 3)
        for _ in range(per_site):
            lat = lat0 + rng.normal(0, spread_km * deg_per_km)
            lon = lon0 + rng.normal(0, spread_km * deg_per_km)
            rows.append({"site": f"s{i}", "lat": lat, "lon": lon,
                        "f1": rng.normal(), "bod": max(0.1, 3.0 + 2 * signal + rng.normal(0, 0.2))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Leakage safety
# ---------------------------------------------------------------------------

def test_predict_oof_never_leaks_a_held_out_stations_own_value():
    """Plant one station with a wildly different value from everyone else. Its OOF prediction must NOT
    be close to its own true value -- if it were, its own visits leaked into its own prediction."""
    df = _clustered_df(n_sites=25, per_site=3, seed=1)
    outlier_true = 500.0
    df.loc[df["site"] == "s0", "bod"] = outlier_true
    reg = SpatialKNNRegressor(target="bod", k=3, feature_cols=["f1"], inner_n_folds=2)
    oof = reg.predict_oof(df, n_folds=5, n_trials=2)
    outlier_pred = oof.loc[df["site"] == "s0", "bod"]
    assert (outlier_pred < 50.0).all(), f"outlier's own OOF prediction {outlier_pred.tolist()} leaked toward its true value {outlier_true}"


def test_predict_oof_knn_only_never_leaks_a_held_out_stations_own_value():
    df = _clustered_df(n_sites=25, per_site=3, seed=2)
    outlier_true = 500.0
    df.loc[df["site"] == "s0", "bod"] = outlier_true
    reg = SpatialKNNRegressor(target="bod", k=3)
    oof = reg.predict_oof_knn_only(df, n_folds=5)
    outlier_pred = oof.loc[df["site"] == "s0", "bod"]
    assert (outlier_pred < 50.0).all()


def test_loo_feature_excludes_a_repeatedly_visited_stations_own_visits():
    """A station visited many times must not dominate its own neighbour search via sheer duplication --
    station_level_lookup collapses it to one row before any KNN search runs."""
    df = _clustered_df(n_sites=10, per_site=1, seed=3)
    heavy = pd.DataFrame({"site": "heavy", "lat": [15.0] * 50, "lon": [80.0] * 50,
                          "f1": np.random.default_rng(4).normal(size=50), "bod": [999.0] * 50})
    df = pd.concat([df, heavy], ignore_index=True)
    reg = SpatialKNNRegressor(target="bod", k=3, inner_n_folds=2)
    lookup = reg.fit(df, n_trials=1)._lookup
    assert (lookup["site"] == "heavy").sum() == 1        # collapsed to a single station-level row
    feat = reg._loo_feature(df[df["site"] == "heavy"], lookup)
    assert not np.any(np.isclose(feat, 999.0))            # never sees its own 999 value as a "neighbour"


# ---------------------------------------------------------------------------
# Distance degradation (synthetic, provable)
# ---------------------------------------------------------------------------

def test_skill_degrades_with_distance_to_nearest_known_station():
    """With a target that's a smooth function of location (genuine spatial autocorrelation, the real
    reason this technique works on the CPCB network), OOF error must be lower for stations that happen
    to be near a neighbour than for ones far from any -- measured within a single run, paired by row,
    not by comparing two independently-random synthetic worlds (too noisy to be a reliable assertion)."""
    rng = np.random.default_rng(11)
    n_sites = 80
    lat0 = rng.uniform(10, 30, n_sites)
    lon0 = rng.uniform(70, 90, n_sites)
    # A smooth, large-scale spatial field: nearby stations share similar values by construction.
    true_field = 5.0 * np.sin(lat0 / 3.0) + 5.0 * np.cos(lon0 / 4.0)
    bod_site = np.maximum(0.1, 3.0 + true_field + rng.normal(0, 0.3, n_sites))
    rows = []
    for i in range(n_sites):
        for _ in range(3):
            rows.append({"site": f"s{i}", "lat": lat0[i] + rng.normal(0, 1e-4), "lon": lon0[i] + rng.normal(0, 1e-4),
                        "bod": bod_site[i] + rng.normal(0, 0.1)})
    df = pd.DataFrame(rows)

    reg = SpatialKNNRegressor(target="bod", k=5)
    oof = reg.predict_oof_knn_only(df, n_folds=10)
    abs_err = (oof["bod"] - df["bod"]).abs()
    near = oof["nearest_station_km"] < oof["nearest_station_km"].median()
    assert abs_err[near].mean() < abs_err[~near].mean(), (
        f"near-neighbour error ({abs_err[near].mean():.3f}) should be lower than far-neighbour error "
        f"({abs_err[~near].mean():.3f})")


# ---------------------------------------------------------------------------
# Shape / NaN handling / not-fitted errors
# ---------------------------------------------------------------------------

def test_predict_raises_before_fit():
    with pytest.raises(RuntimeError, match="fitted"):
        SpatialKNNRegressor(target="bod").predict(pd.DataFrame({"lat": [1.0], "lon": [1.0]}))


def test_predict_oof_output_shape_matches_labelled_rows_only():
    df = _clustered_df(n_sites=15, per_site=3, seed=5)
    df.loc[0, "bod"] = np.nan
    reg = SpatialKNNRegressor(target="bod", k=3, inner_n_folds=2)
    oof = reg.predict_oof(df, n_folds=5, n_trials=1)
    assert len(oof) == df["bod"].notna().sum()
    assert list(oof.columns) == ["bod", "nearest_station_km"]
    assert oof["bod"].notna().all() and oof["nearest_station_km"].notna().all()


# ---------------------------------------------------------------------------
# Save / load round trip
# ---------------------------------------------------------------------------

def test_save_load_round_trip(tmp_path):
    df = _clustered_df(n_sites=20, per_site=3, seed=6)
    reg = SpatialKNNRegressor(target="bod", k=4, feature_cols=["f1"], inner_n_folds=2)
    reg.fit(df, n_trials=2)
    preds = reg.predict(df)

    reg.save(tmp_path)
    loaded = SpatialKNNRegressor.load(tmp_path, "bod")
    loaded_preds = loaded.predict(df)

    pd.testing.assert_frame_equal(preds, loaded_preds)
    assert loaded.k == 4 and loaded.feature_cols == ["f1"]


def test_fit_adds_knn_feature_column_that_predict_does_not_require_the_caller_to_supply():
    df = _clustered_df(n_sites=15, per_site=3, seed=8)
    reg = SpatialKNNRegressor(target="bod", k=3, feature_cols=["f1"], inner_n_folds=2)
    reg.fit(df, n_trials=1)
    new_points = df[["lat", "lon", "f1"]].head(5).copy()
    assert KNN_FEATURE_COL not in new_points.columns
    out = reg.predict(new_points)      # must compute knn_baseline internally, not require it as input
    assert len(out) == 5 and out["bod"].notna().all()
