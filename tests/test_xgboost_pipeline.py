"""
tests/test_xgboost_pipeline.py
==============================
Unit tests for WaterQualityXGB pipeline (Navya / P3).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.xgboost_pipeline import WaterQualityXGB, FEATURE_COLS, TARGET_COLS


@pytest.fixture(scope="module")
def dummy_train_df():
    """Create a minimal synthetic dataframe matching train.parquet schema."""
    np.random.seed(42)
    n = 60
    sites = ["site_A", "site_B", "site_C", "site_D", "site_E", "site_F"]
    site_col = np.random.choice(sites, size=n)
    
    site_coords = {
        "site_A": (12.9, 77.6),
        "site_B": (12.95, 77.65),
        "site_C": (30.8, 75.8),
        "site_D": (31.5, 75.2),
        "site_E": (28.6, 77.2),
        "site_F": (25.3, 83.0),
    }
    lats = [site_coords[s][0] for s in site_col]
    lons = [site_coords[s][1] for s in site_col]
    
    data = {
        "site": site_col,
        "water_body_type": np.random.choice(["lake", "river"], size=n),
        "lat": lats,
        "lon": lons,
        "date": ["2024-01-01"] * n,
        "sensor": ["Sentinel-2"] * n,
    }
    
    for col in FEATURE_COLS:
        data[col] = np.random.uniform(0.01, 0.5, size=n)
        
    data["chl_a"] = np.random.uniform(5.0, 100.0, size=n)
    data["turbidity"] = np.random.uniform(2.0, 80.0, size=n)
    data["do"] = np.random.uniform(3.0, 10.0, size=n)
    
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def trained_pipeline(dummy_train_df):
    """Train once and share across tests."""
    xgb_pipe = WaterQualityXGB(n_folds=3, random_state=42)
    xgb_pipe.train(dummy_train_df, n_trials=1, verbose=False)
    return xgb_pipe


def test_xgboost_train_and_predict(dummy_train_df, trained_pipeline):
    """Verify predictions structure and validity."""
    preds = trained_pipeline.predict(dummy_train_df)
    assert isinstance(preds, pd.DataFrame)
    assert list(preds.columns) == TARGET_COLS
    assert len(preds) == len(dummy_train_df)
    assert not preds.isna().any().any()


def test_xgboost_clipping():
    """Verify physical clipping logic."""
    assert WaterQualityXGB._clip(np.array([-5.0, 100.0, 1500.0]), "chl_a").tolist() == [0.0, 100.0, 1000.0]
    assert WaterQualityXGB._clip(np.array([-1.0, 15.0, 30.0]), "do").tolist() == [0.0, 15.0, 20.0]
    # bounds cover observed extremes (real CPCB turbidity reaches ~3,600 NTU in monsoon rivers)
    assert WaterQualityXGB._clip(np.array([-10.0, 50.0, 3600.0, 9000.0]), "turbidity").tolist() == [0.0, 50.0, 3600.0, 5000.0]
    assert WaterQualityXGB._clip(np.array([-1.0, 40.0, 5000.0]), "bod").tolist() == [0.0, 40.0, 1000.0]


def test_xgboost_save_and_load(dummy_train_df, trained_pipeline, tmp_path):
    """Test saving models and reloading for inference."""
    save_dir = tmp_path / "models"
    trained_pipeline.save(save_dir)
    
    for target in TARGET_COLS:
        assert (save_dir / f"{target}_xgb.json").exists()
        assert (save_dir / f"{target}_params.json").exists()
        
    loaded_pipe = WaterQualityXGB.load(save_dir)
    preds = loaded_pipe.predict(dummy_train_df)
    assert list(preds.columns) == TARGET_COLS
    assert len(preds) == len(dummy_train_df)


def test_xgboost_evaluate(dummy_train_df, trained_pipeline):
    """Test evaluate outputs a report table."""
    eval_df = trained_pipeline.evaluate(dummy_train_df, verbose=False)
    assert isinstance(eval_df, pd.DataFrame)
    assert "target" in eval_df.columns
    assert "water_body_type" in eval_df.columns
    assert "R2" in eval_df.columns
    assert "RMSE" in eval_df.columns


# ---------------------------------------------------------------------------
# Real-data behaviour: each target has its own missing labels
# ---------------------------------------------------------------------------

def test_training_and_oof_handle_missing_labels_per_target(dummy_train_df):
    df = dummy_train_df.copy()
    rng = np.random.default_rng(0)
    df.loc[rng.random(len(df)) < 0.5, "turbidity"] = np.nan          # half of the rows unlabelled
    df["bod"] = np.where(rng.random(len(df)) < 0.7, rng.uniform(1, 30, len(df)), np.nan)
    pipe = WaterQualityXGB(n_folds=3, random_state=1, targets=["turbidity", "bod"])
    scores = pipe.train(df, n_trials=1, verbose=False)
    assert set(scores) == {"turbidity", "bod"} and all(np.isfinite(v) for v in scores.values())
    oof = pipe.predict_oof(df)
    assert oof["turbidity"].notna().all() and oof["bod"].notna().all()       # predicted for every row
    preds = pipe.predict(df)
    assert list(preds.columns) == ["turbidity", "bod"] and preds.notna().all().all()


def test_target_without_any_label_is_skipped(dummy_train_df):
    df = dummy_train_df.copy()
    df["chl_a"] = np.nan
    pipe = WaterQualityXGB(n_folds=3, random_state=1, targets=["chl_a", "do"])
    with pytest.warns(UserWarning, match="no labelled rows"):
        scores = pipe.train(df, n_trials=1, verbose=False)
    assert set(scores) == {"do"}


# ---------------------------------------------------------------------------
# Log-scale targets for heavy-tailed variables
# ---------------------------------------------------------------------------

def _skewed_df(n=120, seed=3):
    rng = np.random.default_rng(seed)
    site = np.repeat([f"s{i}" for i in range(12)], n // 12)
    lat = np.repeat(np.linspace(10, 30, 12), n // 12)
    lon = np.repeat(np.linspace(70, 90, 12), n // 12)
    f = rng.normal(size=(n, 3))
    df = pd.DataFrame({"site": site, "lat": lat, "lon": lon, "water_body_type": "lake",
                       "f1": f[:, 0], "f2": f[:, 1], "f3": f[:, 2]})
    df["turbidity"] = np.exp(2 + 1.2 * f[:, 0] + rng.normal(0, 0.1, n))            # log-normal, heavy right tail
    df["do"] = 6 + f[:, 1] + rng.normal(0, 0.1, n)
    return df


def test_log_targets_are_back_transformed_and_persisted(tmp_path):
    df = _skewed_df()
    pipe = WaterQualityXGB(n_folds=3, random_state=0, feature_cols=["f1", "f2", "f3"], targets=["turbidity", "do"])
    assert pipe.log_targets >= {"turbidity"} and "do" not in pipe.log_targets
    pipe.train(df, n_trials=3, verbose=False)
    pred = pipe.predict(df)
    assert (pred["turbidity"] > 0).all()                                            # back on the original scale, never negative
    assert np.median(pred["turbidity"]) == pytest.approx(np.median(df["turbidity"]), rel=0.6)
    oof = pipe.predict_oof(df)
    assert oof["turbidity"].notna().all() and (oof["turbidity"] > 0).all()

    pipe.save(tmp_path)
    loaded = WaterQualityXGB.load(tmp_path)
    assert loaded.log_targets == pipe.log_targets and loaded.feature_cols == ["f1", "f2", "f3"]
    pd.testing.assert_frame_equal(pipe.predict(df), loaded.predict(df))


def test_log_scale_fit_beats_raw_scale_on_a_lognormal_target():
    df = _skewed_df(seed=5)
    cols = ["f1", "f2", "f3"]
    log_pipe = WaterQualityXGB(n_folds=3, random_state=0, feature_cols=cols, targets=["turbidity"])
    raw_pipe = WaterQualityXGB(n_folds=3, random_state=0, feature_cols=cols, targets=["turbidity"], log_targets=())
    log_pipe.train(df, n_trials=4, verbose=False)
    raw_pipe.train(df, n_trials=4, verbose=False)
    y = np.log1p(df["turbidity"].to_numpy())
    err_log = np.mean((y - np.log1p(log_pipe.predict_oof(df)["turbidity"].to_numpy())) ** 2)
    err_raw = np.mean((y - np.log1p(raw_pipe.predict_oof(df)["turbidity"].to_numpy())) ** 2)
    assert err_log < err_raw


def test_models_saved_without_a_log_setting_load_as_raw_scale(tmp_path):
    import json
    df = _skewed_df()
    pipe = WaterQualityXGB(n_folds=3, random_state=0, feature_cols=["f1", "f2", "f3"], targets=["do"])
    pipe.train(df, n_trials=2, verbose=False)
    pipe.save(tmp_path)
    cfg = json.loads((tmp_path / "config.json").read_text())
    cfg.pop("log_targets")
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    assert WaterQualityXGB.load(tmp_path).log_targets == set()


# ---------------------------------------------------------------------------
# Per-target feature sets (feature_cols as a dict: src.models.schema.FEATURE_SETS)
# ---------------------------------------------------------------------------

def test_features_for_returns_the_right_list_for_dict_or_flat_feature_cols():
    dict_pipe = WaterQualityXGB(feature_cols={"do": ["f1"], "bod": ["f1", "f2"]})
    assert dict_pipe._features_for("do") == ["f1"]
    assert dict_pipe._features_for("bod") == ["f1", "f2"]
    with pytest.raises(KeyError):
        dict_pipe._features_for("turbidity")

    flat_pipe = WaterQualityXGB(feature_cols=["f1", "f2", "f3"])
    assert flat_pipe._features_for("do") == ["f1", "f2", "f3"]
    assert flat_pipe._features_for("anything") == ["f1", "f2", "f3"]


def test_train_predict_and_oof_use_each_targets_own_columns(tmp_path):
    df = _skewed_df(n=144)
    df["f4"] = df["f1"] * 2 + df["f3"]     # extra column only "do" is allowed to see
    feature_cols = {"turbidity": ["f1", "f2", "f3"], "do": ["f1", "f2", "f3", "f4"]}
    pipe = WaterQualityXGB(n_folds=3, random_state=0, feature_cols=feature_cols, targets=["turbidity", "do"])
    pipe.train(df, n_trials=2, verbose=False)

    # predict() must select the target's own columns even from a df carrying every column
    preds = pipe.predict(df)
    assert set(preds.columns) == {"turbidity", "do"} and preds.notna().all().all()

    # a df missing "f4" still works for the "turbidity" model (which never needed it)
    partial = df.drop(columns=["f4"])
    turb_only = WaterQualityXGB(feature_cols=feature_cols, targets=["turbidity"])
    turb_only._models["turbidity"] = pipe._models["turbidity"]
    assert turb_only.predict(partial)["turbidity"].notna().all()

    oof = pipe.predict_oof(df)
    assert oof["turbidity"].notna().all() and oof["do"].notna().all()


def test_per_target_feature_cols_round_trip_through_save_and_load(tmp_path):
    import json
    df = _skewed_df(n=144)
    feature_cols = {"turbidity": ["f1", "f2"], "do": ["f2", "f3"]}
    pipe = WaterQualityXGB(n_folds=3, random_state=0, feature_cols=feature_cols, targets=["turbidity", "do"])
    pipe.train(df, n_trials=2, verbose=False)
    pipe.save(tmp_path)

    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["feature_cols"] == feature_cols

    loaded = WaterQualityXGB.load(tmp_path)
    assert loaded.feature_cols == feature_cols
    assert loaded._features_for("turbidity") == ["f1", "f2"]
    assert loaded._features_for("do") == ["f2", "f3"]
    pd.testing.assert_frame_equal(pipe.predict(df), loaded.predict(df))
