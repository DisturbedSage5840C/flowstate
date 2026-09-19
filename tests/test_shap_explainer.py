"""
tests/test_shap_explainer.py
============================
Unit tests for SHAPExplainer (Navya / P3).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.models.shap_explainer import SHAPExplainer
from src.models.xgboost_pipeline import WaterQualityXGB, FEATURE_COLS


@pytest.fixture
def trained_toy_pipeline():
    np.random.seed(42)
    n = 30
    X = pd.DataFrame(np.random.uniform(0.01, 0.5, size=(n, len(FEATURE_COLS))), columns=FEATURE_COLS)
    y = np.random.uniform(5.0, 50.0, size=n)
    
    model = xgb.XGBRegressor(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    
    pipe = WaterQualityXGB(targets=["chl_a"])
    pipe._models["chl_a"] = model
    return pipe, X


def test_shap_explainer_initialization(tmp_path):
    exp = SHAPExplainer(output_dir=tmp_path / "plots", dpi=100)
    assert exp.output_dir.exists()
    assert exp.dpi == 100


def test_shap_explainer_explain_all(trained_toy_pipeline, tmp_path):
    pipe, X = trained_toy_pipeline
    out_dir = tmp_path / "shap_plots"
    exp = SHAPExplainer(output_dir=out_dir, dpi=100)
    
    exp.explain_all(pipe, X, targets=["chl_a"])
    
    assert (out_dir / "summary_chl_a.png").exists()
    assert (out_dir / "bar_chl_a.png").exists()
    assert (out_dir / "dep_B5_chl_a.png").exists()
