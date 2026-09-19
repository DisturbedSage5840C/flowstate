"""Tests for the real-data dashboard helpers and view (fake bundle; no network, no real files)."""

import numpy as np
import pandas as pd
import pytest

from src.app import real_data
from src.wqi.wqi_engine import classify_cpcb_best_use, compute_wqi_dataframe


def make_table(n_stations=6, visits=4, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_stations):
        for v in range(visits):
            rows.append({
                "site": f"ST{s}", "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=30 * v),
                "state": "Karnataka" if s % 2 == 0 else "Punjab", "lat": 12.0 + s * 0.5, "lon": 77.0 + s * 0.5,
                "water_body_type": "lake" if s % 3 else "river",
                "do": float(rng.uniform(1, 8)), "bod": float(rng.uniform(1, 30)), "turbidity": float(rng.uniform(1, 80)),
                "ph": 7.4, "conductivity": 800.0, "total_coliform": np.nan,
                "day_diff": 1, "n_water_px": 300,
                "do_pred": float(rng.uniform(1, 8)), "bod_pred": float(rng.uniform(1, 30)),
                "turbidity_pred": float(rng.uniform(1, 80)),
                "do_pred_dl": float(rng.uniform(1, 8)), "bod_pred_dl": float(rng.uniform(1, 30)),
                "turbidity_pred_dl": float(rng.uniform(1, 80)),
            })
    df = pd.DataFrame(rows)
    return classify_cpcb_best_use(compute_wqi_dataframe(df))


def test_filter_visits():
    t = make_table()
    assert set(real_data.filter_visits(t, states=["Punjab"])["state"]) == {"Punjab"}
    assert set(real_data.filter_visits(t, types=["river"])["water_body_type"]) == {"river"}
    d = real_data.filter_visits(t, start="2020-02-15", end="2020-03-15")
    assert d["date"].min() >= pd.Timestamp("2020-02-15") and d["date"].max() <= pd.Timestamp("2020-03-15")


def test_latest_per_station_is_the_last_visit():
    t = make_table()
    latest = real_data.latest_per_station(t)
    assert len(latest) == t["site"].nunique()
    assert (latest["date"] == t.groupby("site")["date"].max().reindex(latest["site"]).to_numpy()).all()


def test_model_source_uses_predictions_and_recomputes_wqi_from_predicted_parameters_only():
    t = make_table()
    m = real_data.with_display_columns(t, "xgboost")
    assert np.allclose(m["do"], t["do_pred"]) and np.allclose(m["turbidity"], t["turbidity_pred"])
    d = real_data.with_display_columns(t, "dl")
    assert np.allclose(d["do"], t["do_pred_dl"]) and not np.allclose(d["do"], m["do"])
    assert np.allclose(real_data.with_display_columns(t, "model")["do"], m["do"])      # legacy alias
    assert m["n_wqi_params"].max() == 3                      # do, bod, turbidity - pH/conductivity not injected
    assert not np.allclose(m["wqi"], t["wqi"])               # differs from the measured-value WQI


def test_measured_source_is_unchanged():
    t = make_table()
    m = real_data.with_display_columns(t, "measured")
    pd.testing.assert_frame_equal(m, t)


def test_share_meeting_class_and_tier_counts():
    t = make_table()
    assert 0.0 <= real_data.share_meeting_class(t.assign(cpcb_class_complete=True)) <= 1.0
    partial = t.assign(cpcb_class="A", cpcb_class_complete=False)
    assert np.isnan(real_data.share_meeting_class(partial))                       # partial assignments are not counted
    assert real_data.share_meeting_class(partial, complete_only=False) == 1.0
    assert real_data.complete_class_count(partial) == 0
    assert real_data.tier_counts(t).sum() == t["wqi_tier"].notna().sum()
    assert np.isnan(real_data.share_meeting_class(t.assign(cpcb_class=np.nan)))


def test_metric_pivot():
    m = pd.DataFrame([
        {"model": "xgboost_oof", "target": "do", "water_body_type": "overall", "n": 10, "R2": 0.2, "RMSE": 1.0, "MAE": 0.8},
        {"model": "baseline_mean", "target": "do", "water_body_type": "overall", "n": 10, "R2": -0.1, "RMSE": 1.2, "MAE": 0.9},
    ])
    p = real_data.metric_pivot(m, "overall")
    assert p.loc["do", "R2 - xgboost_oof"] == 0.2 and p.loc["do", "R2 - baseline_mean"] == -0.1


def test_load_real_returns_none_without_a_table(tmp_path):
    assert real_data.load_real(tmp_path) is None


def test_load_real_merges_oof_predictions_and_reads_reports(tmp_path):
    t = make_table().drop(columns=["do_pred", "bod_pred", "turbidity_pred", "do_pred_dl", "bod_pred_dl", "turbidity_pred_dl"])
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "reports" / "real").mkdir(parents=True)
    t.to_parquet(tmp_path / "data" / "processed" / "train_real.parquet")
    oof = t[["site", "date"]].assign(do_pred=1.0, bod_pred=2.0, turbidity_pred=3.0)
    oof.to_parquet(tmp_path / "reports" / "real" / "oof_predictions.parquet")
    oof.assign(do_pred=7.0, bod_pred=8.0, turbidity_pred=9.0).to_parquet(
        tmp_path / "reports" / "real" / "dl_oof_predictions.parquet")
    pd.DataFrame([{"model": "dl_oof", "target": "do", "water_body_type": "overall", "n": 5, "R2": 0.1,
                   "RMSE": 1.0, "MAE": 1.0}]).to_csv(tmp_path / "reports" / "real" / "dl_metrics_table.csv", index=False)
    pd.DataFrame([{"model": "xgboost_oof", "target": "do", "water_body_type": "overall", "n": 5, "R2": 0.2,
                   "RMSE": 0.9, "MAE": 0.9}]).to_csv(tmp_path / "reports" / "real" / "metrics_table.csv", index=False)
    (tmp_path / "reports" / "real" / "dataset_summary.json").write_text('{"rows": 24}')
    b = real_data.load_real(tmp_path)
    assert b.has_predictions and (b.table["do_pred"] == 1.0).all() and b.summary["rows"] == 24
    assert b.has_dl_predictions and (b.table["do_pred_dl"] == 7.0).all()                  # DL columns kept separate
    assert set(b.metrics["model"]) == {"xgboost_oof", "dl_oof"}


def _app():
    import sys
    sys.path.insert(0, ".")
    import pandas as pd
    from tests.test_real_view import make_table
    from src.app.real_data import RealBundle
    from src.app import real_view
    metrics = __import__("pandas").DataFrame([
        {"model": "xgboost_oof", "target": "do", "water_body_type": "overall", "n": 24, "R2": 0.1, "RMSE": 1.0, "MAE": 0.8},
        {"model": "baseline_mean", "target": "do", "water_body_type": "overall", "n": 24, "R2": -0.05, "RMSE": 1.1, "MAE": 0.9}])
    screening = {"shortlist_target": "cpcb_polluted", "targets": {"bod_gt_3": {
        "description": "BOD above 3 mg/L (CPCB Class B/C limit)", "n": 6833, "base_rate": 0.288, "roc_auc": 0.755,
        "average_precision": 0.601, "lift_over_base_rate": 2.1,
        "precision_at_k": {"top_5pct": 0.8, "top_10pct": 0.77, "top_20pct": 0.7},
        "lift_at_k": {"top_5pct": 2.8, "top_10pct": 2.7, "top_20pct": 2.4}}}}
    shortlist = pd.DataFrame({"site": ["A", "B"], "breach_probability": [0.95, 0.91], "visits": [3, 1],
                              "actually_breached": [1.0, 0.0], "state": ["Delhi", "Karnataka"]})
    real_view.render(RealBundle(table=make_table(), metrics=metrics, summary={"rows": 24}, has_predictions=True,
                                has_dl_predictions=True, screening=screening, shortlist=shortlist,
                                summary_metrics={"skill": {"do": "none - not optically active"}}))


def _aoi_app():
    import sys
    sys.path.insert(0, ".")
    import datetime as dt
    import numpy as np
    import pandas as pd
    import rasterio
    import tempfile
    from pathlib import Path
    from rasterio.transform import from_bounds
    from tests.test_real_view import make_table
    from src.app.real_data import RealBundle
    from src.app import real_view
    from src.data import aoi

    def fake_predict_aoi(lat, lon, date, half_size_m=2000.0, **kw):
        d = Path(tempfile.mkdtemp())
        layers = {}
        for name in ("idx_ndci", "do"):
            p = d / f"{name}.tif"
            arr = np.random.default_rng(0).random((20, 20)).astype("float32")
            arr[:5] = -9999.0
            with rasterio.open(p, "w", driver="GTiff", height=20, width=20, count=1, dtype="float32", crs="EPSG:4326",
                               transform=from_bounds(77.65, 12.92, 77.69, 12.96, 20, 20), nodata=-9999.0) as dst:
                dst.write(arr, 1)
            layers[name] = p
        return {"layers": layers, "n_water_px": 300, "water_frac": 0.3, "scene_id": "S2_FAKE", "scene_date": "2020-01-29",
                "scene_cloud": 1.0, "day_diff": 0, "window_cloud_frac": 0.0, "half_size_m": half_size_m,
                "lat": lat, "lon": lon, "note": "experimental"}

    def failing(*a, **k):
        raise aoi.AOIError("no clear Sentinel-2 scene within ±5 days")

    import streamlit as st
    aoi.predict_aoi = failing if st.session_state.get("_fail_aoi") else fake_predict_aoi
    metrics = pd.DataFrame([{"model": "xgboost_oof", "target": "do", "water_body_type": "overall", "n": 24, "R2": 0.05,
                             "RMSE": 1.0, "MAE": 0.8}])
    real_view.render(RealBundle(table=make_table(), metrics=metrics, has_predictions=True))


def test_aoi_tab_runs_shows_skill_warning_and_handles_errors():
    st_testing = pytest.importorskip("streamlit.testing.v1")
    at = st_testing.AppTest.from_function(_aoi_app, default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Skill warning" in w.value and "do R² 0.05" in w.value for w in at.warning)
    at.button(key="aoi_run").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("S2_FAKE" in c.value for c in at.caption)                       # metadata for the fetched scene shown
    at.selectbox(key="aoi_layer").set_value("do").run()
    assert not at.exception, [e.value for e in at.exception]
    at.session_state["_fail_aoi"] = True
    at.button(key="aoi_run").click().run()
    assert any("no clear Sentinel-2 scene" in e.value for e in at.error)       # readable error, no crash


def test_real_view_renders_without_exceptions():
    st_testing = pytest.importorskip("streamlit.testing.v1")
    at = st_testing.AppTest.from_function(_app, default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Real data" in w.value for w in at.warning)
    assert [m.label for m in at.metric][:2] == ["Stations", "Matched visits"]
    assert at.tabs[0].label.endswith("Screening")
    body = " ".join(d.value for d in at.markdown) + " ".join(c.value for c in at.caption)
    assert "inspect" in body.lower()                                   # the shortlist is presented
    assert any("not optically active" in w.value for w in at.warning)  # DO carries its honesty flag
    # switching to model values and another parameter still renders
    for label in ("XGBoost (out-of-fold)", "DL model (out-of-fold)"):
        at.radio(key="rv_source").set_value(label).run()
        at.radio(key="rv_param").set_value("wqi").run()
        assert not at.exception, [e.value for e in at.exception]
