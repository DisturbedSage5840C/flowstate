"""Regression tests for the label-leakage bug: a target must never be an exact function of its own inputs."""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import compute_chl_a_from_ndci, compute_turbidity_dogliotti
from src.models import schema


def test_targets_are_never_model_inputs():
    assert not set(schema.SYNTH_TARGET_COLS) & set(schema.SYNTH_FEATURE_COLS)
    assert not set(schema.REAL_TARGET_COLS) & set(schema.REAL_FEATURE_COLS)


def test_features_module_does_not_emit_target_named_columns():
    from src.features.feature_engineering import compute_all_features
    bands = pd.DataFrame({b: [0.05, 0.06] for b in ("B2", "B3", "B4", "B5", "B6", "B8", "B11")})
    out = compute_all_features(bands)
    assert not (set(schema.SYNTH_TARGET_COLS) | set(schema.REAL_TARGET_COLS)) & set(out.columns)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    from scripts.generate_synthetic_train import generate_synthetic_train
    return generate_synthetic_train(tmp_path_factory.mktemp("syn") / "train.parquet")


def test_synthetic_labels_are_not_formulas_of_the_features(synthetic):
    chl_formula = compute_chl_a_from_ndci(synthetic["ndci"].to_numpy())
    assert np.abs(chl_formula - synthetic["chl_a"].to_numpy()).max() > 1.0
    turb_formula = compute_turbidity_dogliotti(synthetic["B4"].to_numpy(), synthetic["B8"].to_numpy())
    rel = np.abs(turb_formula - synthetic["turbidity"].to_numpy()) / synthetic["turbidity"].to_numpy()
    assert np.nanmedian(rel) > 0.2


def test_synthetic_do_is_independent_of_the_features(synthetic):
    # DO is drawn from the site's own range: within a site it must not track any single band
    df = synthetic
    for band in ("B4", "B5", "B8", "ndci", "temp_surface"):
        within = df.groupby("site").apply(lambda g: abs(np.corrcoef(g[band], g["do"])[0, 1]), include_groups=False)
        assert within.median() < 0.3, band


def test_real_table_labels_are_measurements_not_computed(tmp_path):
    from src.data.training_table import build_training_table
    ins = pd.DataFrame([{"station": "a", "date": pd.Timestamp("2020-01-01"), "state": "X", "lat": 1.0, "lon": 2.0,
                         "water_body_type": "lake", "do": 3.3, "bod": 7.7, "turbidity": 12.1, "ph": 7.0,
                         "conductivity": 500.0, "total_coliform": np.nan, "ammonia_n": np.nan, "sar": np.nan,
                         "boron": np.nan, "source": "CPCB", "is_proxy": False, "retrieved_on": "2026-09-19"}])
    refl = pd.DataFrame([{"station": "a", "date": pd.Timestamp("2020-01-01"), "status": "ok", "n_water_px": 100,
                          "cloud_frac": 0.0, "scene_cloud": 1.0, "scene_id": "s", "scene_date": pd.Timestamp("2020-01-01"),
                          "day_diff": 0, "B2": 0.04, "B3": 0.06, "B4": 0.04, "B5": 0.07, "B6": 0.06, "B8": 0.02,
                          "B11": 0.01}])
    t = build_training_table(ins, refl)
    assert (t["do"].iloc[0], t["bod"].iloc[0], t["turbidity"].iloc[0]) == (3.3, 7.7, 12.1)      # untouched by any formula
