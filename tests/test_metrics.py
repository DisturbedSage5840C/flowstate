"""tests/test_metrics.py — unit tests for metrics.py"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.models.metrics import MetricsReporter, spatial_cv_rmse


def _make_pred_df(n=100, seed=42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    wbt = rng.choice(["lake", "river", "reservoir"], size=n)
    y_true = pd.DataFrame({
        "water_body_type": wbt,
        "chl_a":     rng.uniform(5, 200, n),
        "turbidity": rng.uniform(2, 500, n),
        "do":        rng.uniform(2, 12,  n),
    })
    # Predictions = truth + small noise
    y_pred = pd.DataFrame({
        "chl_a":     y_true["chl_a"]     + rng.normal(0, 5, n),
        "turbidity": y_true["turbidity"] + rng.normal(0, 20, n),
        "do":        y_true["do"]        + rng.normal(0, 0.5, n),
    })
    return y_true, y_pred


class TestMetricsReporter:
    def test_report_returns_dataframe(self):
        y_true, y_pred = _make_pred_df()
        reporter = MetricsReporter()
        table = reporter.report(y_true, y_pred)
        assert isinstance(table, pd.DataFrame)

    def test_report_has_required_columns(self):
        y_true, y_pred = _make_pred_df()
        reporter = MetricsReporter()
        table = reporter.report(y_true, y_pred)
        for col in ["target", "water_body_type", "n", "R2", "RMSE", "MAE"]:
            assert col in table.columns, f"Missing column: {col}"

    def test_overall_row_present(self):
        y_true, y_pred = _make_pred_df()
        reporter = MetricsReporter()
        table = reporter.report(y_true, y_pred)
        assert "overall" in table["water_body_type"].values

    def test_r2_close_to_1_for_near_perfect_preds(self):
        rng = np.random.default_rng(0)
        n = 200
        wbt = rng.choice(["lake", "river"], size=n)
        y_true = pd.DataFrame({
            "water_body_type": wbt,
            "chl_a": rng.uniform(5, 200, n),
        })
        y_pred = pd.DataFrame({"chl_a": y_true["chl_a"] + rng.normal(0, 0.01, n)})
        reporter = MetricsReporter(targets=["chl_a"])
        table = reporter.report(y_true, y_pred)
        overall_r2 = table.loc[table["water_body_type"] == "overall", "R2"].values[0]
        assert overall_r2 > 0.99, f"Expected R²≈1 for near-perfect predictions, got {overall_r2:.4f}"

    def test_rmse_known_value(self):
        """RMSE of constant error = that constant."""
        n = 10
        wbt = ["lake"] * n
        y_true = pd.DataFrame({"water_body_type": wbt, "chl_a": np.ones(n) * 50.0})
        y_pred = pd.DataFrame({"chl_a": np.ones(n) * 53.0})  # error = 3 everywhere
        reporter = MetricsReporter(targets=["chl_a"])
        table = reporter.report(y_true, y_pred)
        rmse = table.loc[table["water_body_type"] == "overall", "RMSE"].values[0]
        assert abs(rmse - 3.0) < 1e-6, f"Expected RMSE=3.0, got {rmse}"

    def test_print_table_runs_without_error(self, capsys):
        y_true, y_pred = _make_pred_df()
        reporter = MetricsReporter()
        table = reporter.report(y_true, y_pred)
        reporter.print_table(table)
        captured = capsys.readouterr()
        assert "RMSE" in captured.out


class TestSpatialCvRmse:
    def test_zero_error(self):
        y = np.array([1.0, 2.0, 3.0])
        assert spatial_cv_rmse(y, y) == pytest.approx(0.0)

    def test_known_rmse(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])
        # RMSE = sqrt((9+16)/2) = sqrt(12.5)
        expected = np.sqrt(12.5)
        assert spatial_cv_rmse(y_true, y_pred) == pytest.approx(expected)


def test_r2_log_reported_for_heavy_tailed_targets_only():
    import numpy as np
    import pandas as pd
    from src.models.metrics import MetricsReporter

    rng = np.random.default_rng(0)
    truth = np.exp(rng.normal(2, 1.5, 200))
    df_true = pd.DataFrame({"bod": truth, "do": rng.normal(6, 1, 200), "water_body_type": "lake"})
    df_pred = pd.DataFrame({"bod": truth * np.exp(rng.normal(0, 0.2, 200)), "do": df_true["do"] + rng.normal(0, 0.5, 200)})
    t = MetricsReporter(targets=["bod", "do"]).report(df_true, df_pred)
    bod = t[(t.target == "bod") & (t.water_body_type == "overall")].iloc[0]
    do = t[(t.target == "do") & (t.water_body_type == "overall")].iloc[0]
    assert bod["R2_log"] > 0.9
    assert np.isnan(do["R2_log"])


# ---------------------------------------------------------------------------
# Constant-per-fold baseline: pooled Spearman is an artifact, not real skill
# ---------------------------------------------------------------------------

def test_constant_within_group_prediction_returns_nan_spearman():
    from src.models.metrics import _safe_spearman

    rng = np.random.default_rng(0)
    y_true = rng.uniform(1, 50, 60)
    groups = np.repeat(["fold_a", "fold_b", "fold_c"], 20)
    # a "predict the fold mean" baseline: every row in a group gets that group's one constant value
    y_pred = np.repeat([10.0, 30.0, 5.0], 20)
    assert np.isnan(_safe_spearman(y_true, y_pred, groups=groups))
    # a real model varies within every group -> genuine Spearman, not NaN
    y_pred_real = y_true + rng.normal(0, 2, 60)
    assert not np.isnan(_safe_spearman(y_true, y_pred_real, groups=groups))
    # no groups given -> unchanged prior behaviour (pooled Spearman computed as before)
    assert not np.isnan(_safe_spearman(y_true, y_pred))


def test_metrics_reporter_report_returns_nan_spearman_for_a_constant_per_fold_baseline():
    from src.models.baselines import mean_baseline_oof
    from src.models.spatial_cv import SpatialKFold

    rng = np.random.default_rng(1)
    n = 60
    sites = [f"s{i}" for i in range(12)]
    df = pd.DataFrame({
        "site": np.repeat(sites, n // 12),
        "lat": np.repeat(np.linspace(10, 30, 12), n // 12),
        "lon": np.repeat(np.linspace(70, 90, 12), n // 12),
        "water_body_type": "lake",
        "bod": rng.uniform(1, 30, n),
    })
    baseline = mean_baseline_oof(df, ["bod"], n_folds=3, random_state=0)
    table = MetricsReporter(targets=["bod"]).report(df, baseline)
    overall = table[table.water_body_type == "overall"].iloc[0]
    assert np.isnan(overall["spearman"])


def test_report_still_computes_spearman_without_a_site_column():
    """group_col defaults to 'site'; when it's absent from df_true, spearman falls back to the old
    (ungrouped) pooled behaviour instead of crashing."""
    y_true, y_pred = _make_pred_df()
    reporter = MetricsReporter(targets=["chl_a"])
    table = reporter.report(y_true, y_pred)          # _make_pred_df's frame has no "site" column
    overall = table[table.water_body_type == "overall"].iloc[0]
    assert not np.isnan(overall["spearman"])
