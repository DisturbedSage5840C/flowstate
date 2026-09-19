import numpy as np
import pandas as pd
import pytest

from src.data import kaggle_sources as ks


def make_csv(tmp_path, name="sangam", n_days=3, per_day=8, bad_ph=True):
    rows = []
    for d in range(n_days):
        for r in range(per_day):
            ts = pd.Timestamp("2019-03-01 09:30") + pd.Timedelta(days=d, minutes=10 * r)
            rows.append({"Date": ts.strftime("%Y-%m-%d %H:%M:%S"), "DO": 8.0 + 0.1 * d, "pH": 11.0 if bad_ph else 8.0,
                         "ORP": 0.1, "Cond": 300.0, "Temp": 25.0 + d, "WQI": 30.0, "Status": "Fair"})
    rows.append({"Date": "2019-03-01 10:00:00", "DO": 19.9, "pH": 13.9, "ORP": 0.1, "Cond": 0.04, "Temp": 25.0,
                 "WQI": 60.0, "Status": "Very Poor"})           # impossible DO
    pd.DataFrame(rows).to_csv(tmp_path / f"{name}.csv", index=False)


def test_load_flags_implausible_values(tmp_path):
    make_csv(tmp_path)
    df = ks.load_iot("sangam", tmp_path)
    assert df["Date"].is_monotonic_increasing
    assert not df.loc[df["DO"] > 14, "do_ok"].any() and df["do_ok"].sum() == len(df) - 1


def test_reliability_report_exposes_the_ph_problem(tmp_path):
    make_csv(tmp_path)
    rep = ks.reliability_report(ks.load_iot("sangam", tmp_path))
    assert rep["share_ph_above_8_5"] == 1.0 and rep["median_ph"] == 11.0
    assert rep["share_do_impossible"] > 0 and rep["days_with_data"] == 3


def test_daily_summary_uses_overpass_window_and_drops_unreliable_columns(tmp_path):
    make_csv(tmp_path)
    s = ks.daily_summary(ks.load_iot("sangam", tmp_path))
    assert list(s.columns) == ["date", "temp_c", "n_temp", "do", "n_do"]          # no pH / Cond / ORP
    assert s["temp_c"].tolist() == [25.0, 26.0, 27.0]
    assert s["do"].iloc[0] == pytest.approx(8.0)                                   # the 19.9 mg/L reading was excluded


def test_days_with_too_few_readings_are_nan(tmp_path):
    make_csv(tmp_path, n_days=1, per_day=2)
    s = ks.daily_summary(ks.load_iot("sangam", tmp_path), min_readings=5)
    assert np.isnan(s["temp_c"].iloc[0]) and np.isnan(s["do"].iloc[0])


def test_station_frame_carries_coordinates_and_provenance(tmp_path):
    make_csv(tmp_path)
    f = ks.station_frame("sangam", tmp_path)
    assert f["station"].iloc[0] == "KAGGLE_SANGAM" and 81.0 < f["lon"].iloc[0] < 82.5 and not f["is_proxy"].iloc[0]
    assert "Kaggle" in f["source"].iloc[0]


def test_unknown_series_and_missing_file_are_clear_errors(tmp_path):
    with pytest.raises(ValueError):
        ks.load_iot("yamuna", tmp_path)
    with pytest.raises(FileNotFoundError, match="Kaggle token"):
        ks.load_iot("ganga", tmp_path)


def test_coordinates_are_at_prayagraj_not_the_typo_longitude():
    for lat, lon, _ in ks.LOCATIONS.values():
        assert 25.3 < lat < 25.6 and 81.7 < lon < 82.1           # Prayagraj is near 81.9 E, not 78.9 E


def test_validation_stats_match_hand_computation():
    from scripts.validate_landsat_temperature import stats
    insitu = np.array([20.0, 25.0, 30.0])
    sat = np.array([22.0, 25.0, 34.0])
    s = stats(insitu, sat)
    assert s["n"] == 3 and s["bias_c"] == pytest.approx(2.0) and s["mae_c"] == pytest.approx(2.0)
    assert s["rmse_c"] == pytest.approx(np.sqrt((4 + 0 + 16) / 3), abs=0.01) and s["pearson_r"] > 0.95
