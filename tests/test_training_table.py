import numpy as np
import pandas as pd
import pytest

from src.data.training_table import BANDS, build_training_table, quality_filter


def refl_row(station, date, **kw):
    row = {"station": station, "date": pd.Timestamp(date), "status": "ok", "n_water_px": 500, "cloud_frac": 0.0,
           "scene_cloud": 5.0, "scene_id": "S2A_x", "scene_date": pd.Timestamp(date), "day_diff": 1,
           "B2": 0.04, "B3": 0.06, "B4": 0.04, "B5": 0.07, "B6": 0.06, "B8": 0.02, "B11": 0.01}
    row.update(kw)
    return row


def insitu_row(station, date, **kw):
    row = {"station": station, "date": pd.Timestamp(date), "state": "Karnataka", "lat": 12.9, "lon": 77.6,
           "water_body_type": "lake", "do": 4.0, "bod": 8.0, "turbidity": np.nan, "ph": 7.4, "conductivity": 900.0,
           "total_coliform": np.nan, "ammonia_n": np.nan, "sar": np.nan, "boron": np.nan,
           "source": "CPCB NWDP manual monitoring", "is_proxy": False, "retrieved_on": "2026-09-19"}
    row.update(kw)
    return row


def test_quality_filter_rejects_bad_extractions():
    refl = pd.DataFrame([
        refl_row("a", "2020-01-01"),
        refl_row("b", "2020-01-01", status="cloudy"),
        refl_row("c", "2020-01-01", n_water_px=5),
        refl_row("d", "2020-01-01", B3=0.001),
        refl_row("e", "2020-01-01", B8=0.5),
        refl_row("f", "2020-01-01", B5=np.nan),
    ])
    assert quality_filter(refl)["station"].tolist() == ["a"]


def test_join_only_keeps_matched_visits_and_adds_features_and_provenance():
    ins = pd.DataFrame([insitu_row("a", "2020-01-01"), insitu_row("b", "2020-01-01")])
    refl = pd.DataFrame([refl_row("a", "2020-01-01")])
    t = build_training_table(ins, refl)
    assert t["site"].tolist() == ["a"]
    assert {"ndci", "bdm2", "bdm3", "red_green", "nir", "chl_a_empirical", "turbidity_empirical"} <= set(t.columns)
    assert t["ndci"].iloc[0] == pytest.approx((0.07 - 0.04) / (0.07 + 0.04), abs=1e-5)
    assert not t["is_proxy"].iloc[0] and t["source"].iloc[0].startswith("CPCB")
    assert not {"chl_a", "turbidity_x"} & set(t.columns) or "chl_a" not in t.columns   # no invented Chl-a label


def test_wqi_and_class_use_only_measured_parameters():
    ins = pd.DataFrame([insitu_row("a", "2020-01-01", do=7.0, bod=1.0, ph=7.5, total_coliform=20.0)])
    t = build_training_table(ins, pd.DataFrame([refl_row("a", "2020-01-01")]))
    assert t["cpcb_class"].iloc[0] == "A"
    assert t["n_wqi_params"].iloc[0] == 5            # do, bod, ph, conductivity, coliform (turbidity / chl-a were not measured)
    assert np.isnan(t["qi_chl_a"].iloc[0])


def test_missing_reflectance_yields_empty_table():
    t = build_training_table(pd.DataFrame([insitu_row("a", "2020-01-01")]),
                             pd.DataFrame([refl_row("a", "2020-01-01", status="no_water")]))
    assert len(t) == 0


def test_temperature_is_merged_where_available_and_nan_elsewhere():
    ins = pd.DataFrame([insitu_row("a", "2020-01-01"), insitu_row("b", "2020-01-01")])
    refl = pd.DataFrame([refl_row("a", "2020-01-01"), refl_row("b", "2020-01-01")])
    temp = pd.DataFrame([
        {"station": "a", "date": pd.Timestamp("2020-01-01"), "temp_status": "ok", "temp_c": 26.5, "temp_day_diff": 2},
        {"station": "b", "date": pd.Timestamp("2020-01-01"), "temp_status": "no_clear_water"},
    ])
    t = build_training_table(ins, refl, temperature=temp).set_index("site")
    assert t.loc["a", "temp_surface"] == 26.5 and "Landsat" in t.loc["a", "temp_source"]
    assert np.isnan(t.loc["b", "temp_surface"]) and pd.isna(t.loc["b", "temp_source"])


def test_without_temperature_the_column_is_all_nan_not_a_constant():
    t = build_training_table(pd.DataFrame([insitu_row("a", "2020-01-01")]), pd.DataFrame([refl_row("a", "2020-01-01")]))
    assert t["temp_surface"].isna().all()


def test_real_temperature_changes_the_do_surrogate_only_for_rows_that_have_it():
    ins = pd.DataFrame([insitu_row("a", "2020-01-01"), insitu_row("b", "2020-01-01")])
    refl = pd.DataFrame([refl_row("a", "2020-01-01"), refl_row("b", "2020-01-01")])
    temp = pd.DataFrame([{"station": "a", "date": pd.Timestamp("2020-01-01"), "temp_status": "ok", "temp_c": 35.0,
                          "temp_day_diff": 0}])
    t = build_training_table(ins, refl, temperature=temp).set_index("site")
    assert t.loc["a", "do_empirical"] < t.loc["b", "do_empirical"]          # 35 C holds less oxygen than the 25 C default


def test_implausible_water_temperatures_are_dropped():
    ins = pd.DataFrame([insitu_row(s, "2020-01-01") for s in "abc"])
    refl = pd.DataFrame([refl_row(s, "2020-01-01") for s in "abc"])
    temp = pd.DataFrame([{"station": s, "date": pd.Timestamp("2020-01-01"), "temp_status": "ok", "temp_c": v,
                          "temp_day_diff": 0} for s, v in zip("abc", (27.0, 45.6, -0.4))])
    t = build_training_table(ins, refl, temperature=temp).set_index("site")
    assert t.loc["a", "temp_surface"] == 27.0 and np.isnan(t.loc["b", "temp_surface"]) and np.isnan(t.loc["c", "temp_surface"])
