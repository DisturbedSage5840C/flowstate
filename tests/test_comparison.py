import pandas as pd

from src.models.comparison import compare_models


def rows(model, target, r2, rmse, n=100, wt="overall"):
    return {"model": model, "target": target, "water_body_type": wt, "n": n, "R2": r2, "RMSE": rmse, "MAE": rmse}


def test_xgboost_is_default_and_dl_needs_a_clear_margin():
    xgb = pd.DataFrame([rows("xgboost_oof", "do", 0.30, 1.00), rows("baseline_mean", "do", -0.01, 1.20),
                        rows("baseline_median", "do", -0.02, 1.21)])
    dl_close = pd.DataFrame([rows("dl_oof", "do", 0.32, 0.97)])            # only 3% better -> not clear
    dl_clear = pd.DataFrame([rows("dl_oof", "do", 0.45, 0.90)])            # 10% better -> chosen
    assert compare_models(xgb, dl_close)["production_model"].iloc[0] == "xgboost_oof"
    assert compare_models(xgb, dl_clear)["production_model"].iloc[0] == "dl_oof"


def test_beats_baseline_flag_is_false_when_model_is_no_better_than_the_average():
    xgb = pd.DataFrame([rows("xgboost_oof", "bod", -0.10, 30.0), rows("baseline_mean", "bod", -0.01, 28.0),
                        rows("baseline_median", "bod", -0.05, 29.0)])
    out = compare_models(xgb)
    assert not out["beats_baseline"].iloc[0] and out["production_model"].iloc[0] == "xgboost_oof"


def test_one_row_per_target_and_water_type_filter():
    xgb = pd.DataFrame([rows("xgboost_oof", "do", 0.3, 1.0), rows("baseline_mean", "do", 0.0, 1.2),
                        rows("xgboost_oof", "do", -1.0, 3.0, wt="river"), rows("xgboost_oof", "bod", 0.1, 5.0),
                        rows("baseline_mean", "bod", 0.0, 5.5)])
    out = compare_models(xgb)
    assert sorted(out["target"]) == ["bod", "do"]
    assert out.set_index("target").loc["do", "xgboost_oof_R2"] == 0.3        # 'overall' row, not the river row
    assert compare_models(xgb, water_type="river")["target"].tolist() == ["do"]


def test_works_without_dl_metrics():
    xgb = pd.DataFrame([rows("xgboost_oof", "do", 0.3, 1.0), rows("baseline_mean", "do", 0.0, 1.2)])
    out = compare_models(xgb, None)
    assert out["production_model"].iloc[0] == "xgboost_oof" and out["beats_baseline"].iloc[0]


def test_marginal_rmse_edge_is_not_reported_as_beating_the_baseline_and_skill_is_labelled():
    xgb = pd.DataFrame([rows("xgboost_oof", "turbidity", -0.055, 60.95), rows("baseline_mean", "turbidity", -0.059, 61.06),
                        rows("baseline_median", "turbidity", -0.09, 62.0)])
    out = compare_models(xgb).iloc[0]
    assert not out["beats_baseline"] and out["skill"] == "none"          # 0.2 % lower RMSE is noise, not skill


def test_skill_uses_log_scale_r2_for_heavy_tailed_targets():
    r = rows("xgboost_oof", "bod", -0.02, 16.0)
    r["R2_log"] = 0.40
    base = rows("baseline_mean", "bod", -0.03, 17.5)
    base["R2_log"] = -0.5
    out = compare_models(pd.DataFrame([r, base])).iloc[0]
    assert out["skill"] == "moderate" and out["beats_baseline"]


def test_skill_label_boundaries():
    from src.models.comparison import skill_label
    assert [skill_label(v) for v in (-0.5, 0.05, 0.06, 0.30, 0.31, 0.60, 0.61, float("nan"))] ==         ["none", "none", "weak", "weak", "moderate", "moderate", "good", "unknown"]
