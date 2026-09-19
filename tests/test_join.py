import numpy as np
import pandas as pd
import pytest

from src.features.spatial_temporal_join import _haversine_km, get_day_tolerance, spatial_temporal_join


@pytest.mark.parametrize("sensor, days", [("S2", 3), ("Sentinel-2", 3), ("LS", 5), ("L8", 5), ("L9", 5),
                                           ("LANDSAT-8", 5), ("landsat_9", 5), ("unknown", 3)])
def test_sensor_tolerances(sensor, days):
    assert get_day_tolerance(sensor) == days


def test_haversine_accepts_arrays_and_matches_scalar():
    lat, lon = np.array([12.93, 13.0]), np.array([77.67, 77.6])
    d = _haversine_km(12.93, 77.67, lat, lon)
    assert d[0] == pytest.approx(0.0, abs=1e-6) and 8 < d[1] < 12
    assert float(_haversine_km(12.93, 77.67, 12.93, 77.67)) == pytest.approx(0.0, abs=1e-6)


def _features(sensor, date, lat=12.93, lon=77.67, site="a"):
    return {"site": site, "lat": lat, "lon": lon, "date": date, "sensor": sensor, "ndci": 0.2}


def _gt(date, site="a"):
    return {"site": site, "lat": 12.93, "lon": 77.67, "date": date, "chl_a": 1.0, "turbidity": 2.0, "do": 3.0,
            "bod": 4.0, "source": "x"}


def test_landsat_uses_five_day_window_and_sentinel_three():
    gts = pd.concat([pd.DataFrame([_gt("2020-01-10")])] * 3)                     # 4 days before the scenes
    ls = spatial_temporal_join(pd.concat([pd.DataFrame([_features("LS", "2020-01-14")])] * 3), gts)
    s2 = spatial_temporal_join(pd.concat([pd.DataFrame([_features("S2", "2020-01-14")])] * 3), gts)
    assert len(ls) == 3          # Landsat: 4 days <= 5
    assert len(s2) == 0          # Sentinel-2: 4 days > 3


def test_distance_filter_uses_radius():
    far = _features("S2", "2020-01-10", lat=13.5)
    near = _features("S2", "2020-01-10")
    feats = pd.concat([pd.DataFrame([far])] * 3 + [pd.DataFrame([near])] * 3)
    gts = pd.concat([pd.DataFrame([_gt("2020-01-10")])] * 3)
    out = spatial_temporal_join(feats, gts, radius_km=0.5)
    assert len(out) == 3 and (out["match_dist_km"] <= 0.5).all()
