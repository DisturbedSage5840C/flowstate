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
    assert WaterQualityXGB._clip(np.array([-5.0, 100.0, 800.0]), "chl_a").tolist() == [0.0, 100.0, 500.0]
    assert WaterQualityXGB._clip(np.array([-1.0, 15.0, 30.0]), "do").tolist() == [0.0, 15.0, 20.0]
    assert WaterQualityXGB._clip(np.array([-10.0, 50.0, 2500.0]), "turbidity").tolist() == [0.0, 50.0, 2000.0]


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
