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


# ---------------------------------------------------------------------------
# Open-Meteo rate limiting and the antecedent-rainfall lookup
# ---------------------------------------------------------------------------

def test_rate_limit_wait_matches_the_limit_that_was_actually_hit():
    import datetime as dt
    from src.data.weather import QuotaExhausted, rate_limit_wait

    now = dt.datetime(2026, 9, 19, 17, 44, tzinfo=dt.timezone.utc)
    # the live error is hourly, not daily; exponential backoff could never clear it
    assert 900 < rate_limit_wait("Hourly API request limit exceeded. Please try again in the next hour.", 0, now) < 1000
    assert rate_limit_wait("Minutely API request limit exceeded.", 0, now) == 2.0
    assert rate_limit_wait("Minutely API request limit exceeded.", 3, now) == 16.0
    assert rate_limit_wait("something else", 10, now) == 60.0                      # capped
    with pytest.raises(QuotaExhausted, match="UTC midnight"):
        rate_limit_wait("Daily API request limit exceeded.", 0, now)


def test_antecedent_rainfall_uses_the_last_day_at_or_before_the_window_edge():
    """A gap in the cache must not silently null the feature: reindex().ffill() returned NaN whenever the
    exact window-edge day was missing, asof() correctly falls back to the previous day."""
    from src.data.weather import antecedent_rainfall_features

    rain = pd.DataFrame({"site": ["a"] * 4,
                         "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-04", "2020-01-05"]),
                         "precip_mm": [1.0, 2.0, 3.0, 4.0]})
    visits = pd.DataFrame({"site": ["a"], "date": [pd.Timestamp("2020-01-04")]})   # needs 2020-01-03, absent
    out = antecedent_rainfall_features(visits, rain)
    assert out["rain_3d_mm"].iloc[0] == pytest.approx(2.0)        # days 1-3 present in cache: 1+2, day 3 missing
    assert out["rain_30d_mm"].iloc[0] == pytest.approx(3.0)       # everything before the visit
    assert out.notna().all().all()


def test_rainfall_features_are_nan_for_an_unknown_site():
    from src.data.weather import antecedent_rainfall_features

    rain = pd.DataFrame({"site": ["a"], "date": [pd.Timestamp("2020-01-01")], "precip_mm": [5.0]})
    out = antecedent_rainfall_features(pd.DataFrame({"site": ["b"], "date": [pd.Timestamp("2020-02-01")]}), rain)
    assert out.isna().all().all()
