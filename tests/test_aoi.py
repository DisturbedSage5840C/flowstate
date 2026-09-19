"""Offline tests for the AOI pipeline (fake scene / fake STAC extractor; no network)."""

import datetime as dt
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds

from src.data import aoi
from src.data.satellite_extract import ExtractionConfig

N = 60


def make_scene(with_water=True, cloudy=False):
    bands = {b: np.full((N, N), 0.05) for b in ("B2", "B3", "B4", "B5", "B6", "B8", "B11")}
    if with_water:
        bands["B3"][:, :30], bands["B11"][:, :30], bands["B8"][:, :30] = 0.08, 0.01, 0.02   # left half water
        bands["B5"][:, :30], bands["B4"][:, :30] = 0.09, 0.05
        bands["B3"][:, 30:], bands["B11"][:, 30:], bands["B8"][:, 30:] = 0.10, 0.25, 0.30   # right half land
    scl = np.full((N, N), 6 if with_water else 5, dtype="uint8")
    if cloudy:
        scl[:] = 9
    # UTM-like projected grid (metres), 10 m pixels, near Bengaluru
    transform = from_bounds(780000, 1430000, 780000 + N * 10, 1430000 + N * 10, N, N)
    return aoi.AOIScene(bands, scl, transform, "EPSG:32643", "S2A_TEST", dt.date(2020, 1, 29), 3.0,
                        12.93, 77.67, 300.0, {"day_diff": 1, "window_cloud_frac": 0.0})


def fake_predict(feats: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"do": 4.0 + 5 * feats["ndci"].to_numpy(), "bod": 9.0, "turbidity": 20.0,
                         "wqi": 55.0}, index=feats.index)


def test_predict_from_scene_writes_wgs84_geotiffs_only_over_water(tmp_path):
    res = aoi.predict_aoi_from_scene(make_scene(), fake_predict, tmp_path, "t", smooth_px=0)
    assert set(res["layers"]) == {"do", "bod", "turbidity", "wqi", "idx_ndci", "idx_turbidity_empirical"}
    assert 0.4 < res["water_frac"] < 0.6 and res["scene_id"] == "S2A_TEST" and "experimental" in res["note"]
    with rasterio.open(res["layers"]["bod"]) as src:
        assert src.crs.to_string() == "EPSG:4326" and src.nodata == aoi.NODATA
        data = src.read(1)
        lon0, lat0, lon1, lat1 = src.bounds
    assert 77.0 < lon0 < lon1 < 78.5 and 12.0 < lat0 < lat1 < 14.0       # lat/lon, not metres
    valid = data != aoi.NODATA
    assert 0.3 < valid.mean() < 0.7 and np.allclose(data[valid], 9.0)


def test_no_water_raises_a_readable_error(tmp_path):
    with pytest.raises(aoi.AOIError, match="water pixels"):
        aoi.predict_aoi_from_scene(make_scene(with_water=False), fake_predict, tmp_path, "t")


def test_write_wgs84_keeps_nan_as_nodata(tmp_path):
    arr = np.full((N, N), np.nan, dtype="float32")
    arr[10:20, 10:20] = 3.0
    path = aoi.write_wgs84(arr, make_scene().transform, "EPSG:32643", tmp_path / "x.tif")
    with rasterio.open(path) as src:
        d = src.read(1)
    assert set(np.unique(d)) <= {3.0, aoi.NODATA} and (d == 3.0).any()


class FakeExtractor:
    def __init__(self, scenes):
        self.scenes = scenes          # list of (date, cloud_cover, window)
        self.reads = 0

    def search(self, lat, lon, start, end):
        return [SimpleNamespace(id=f"item{i}", datetime=dt.datetime.combine(d, dt.time(5, 0), tzinfo=dt.timezone.utc),
                                properties={"eo:cloud_cover": cc}, window=w) for i, (d, cc, w) in enumerate(self.scenes)]

    def _read_window(self, item, lat, lon, radius_m=None):
        self.reads += 1
        s = item.window
        return {"bands": s.bands, "scl": s.scl, "transform": s.transform, "crs": s.crs}


def test_fetch_prefers_nearest_scene_and_skips_cloudy_windows():
    cloudy, clear = make_scene(cloudy=True), make_scene()
    ex = FakeExtractor([(dt.date(2020, 1, 29), 5.0, cloudy), (dt.date(2020, 1, 31), 10.0, clear)])
    scene = aoi.fetch_aoi_scene(12.93, 77.67, "2020-01-29", half_size_m=300, extractor=ex,
                                cfg=ExtractionConfig(day_tolerance=5, candidates=4))
    assert scene is not None and scene.item_id == "item1" and scene.extra["day_diff"] == 2
    assert ex.reads == 2                                                  # tried the cloudy one first, then moved on


def test_fetch_returns_none_when_nothing_is_within_tolerance_or_clear():
    ex = FakeExtractor([(dt.date(2020, 3, 1), 5.0, make_scene())])
    assert aoi.fetch_aoi_scene(12.9, 77.6, "2020-01-29", extractor=ex, tolerance_days=5) is None
    ex2 = FakeExtractor([(dt.date(2020, 1, 29), 5.0, make_scene(cloudy=True))])
    assert aoi.fetch_aoi_scene(12.9, 77.6, "2020-01-29", extractor=ex2, tolerance_days=5) is None


def test_predict_aoi_reports_missing_models(tmp_path):
    with pytest.raises(aoi.AOIError, match="no trained models"):
        aoi.predict_aoi(12.9, 77.6, "2020-01-29", models_dir=tmp_path, spatial_models_dir=tmp_path / "spatial")


# ---------------------------------------------------------------------------
# with_context: the any-AOI serving path supplies urban/season/type context per pixel
# ---------------------------------------------------------------------------

def test_with_context_adds_urban_season_and_default_type_columns():
    def base_predict(feats: pd.DataFrame) -> pd.DataFrame:
        needed = {"ndci", "dist_nearest_city_km", "urban_load_index",
                 "is_winter", "is_summer", "is_monsoon", "is_river", "is_lake", "is_reservoir"}
        assert needed <= set(feats.columns)
        assert (feats["is_river"] == 0).all() and (feats["is_lake"] == 0).all() and (feats["is_reservoir"] == 0).all()
        assert (feats["dist_nearest_city_km"] < 5.0).all()           # Bengaluru itself: essentially 0 km away
        return pd.DataFrame({"bod": 5.0}, index=feats.index)

    wrapped, meta = aoi.with_context(base_predict, lat=12.9716, lon=77.5946, date="2020-07-15")   # Bengaluru, monsoon
    feats = pd.DataFrame({"ndci": [0.1, 0.2, 0.3]})
    out = wrapped(feats)
    assert (out["bod"] == 5.0).all() and len(out) == 3
    assert meta["nearest_station_km"] is None          # no spatial_models given -> nothing to report


def test_with_context_derives_season_from_the_scene_date():
    seen = {}

    def base_predict(feats: pd.DataFrame) -> pd.DataFrame:
        seen.update(feats.iloc[0].to_dict())
        return pd.DataFrame({"bod": 1.0}, index=feats.index)

    wrapped, _ = aoi.with_context(base_predict, lat=20.0, lon=78.0, date="2020-01-15")
    wrapped(pd.DataFrame({"ndci": [0.1]}))
    assert seen["is_winter"] == 1 and seen["is_summer"] == 0 and seen["is_monsoon"] == 0


def test_with_context_leaves_the_original_frame_untouched():
    def base_predict(feats):
        return pd.DataFrame({"bod": 1.0}, index=feats.index)

    feats = pd.DataFrame({"ndci": [0.1, 0.2]})
    wrapped, _ = aoi.with_context(base_predict, lat=12.9, lon=77.6, date="2020-01-29")
    wrapped(feats)
    assert list(feats.columns) == ["ndci"]            # the caller's frame was not mutated in place


def test_with_context_reports_nearest_station_km_when_spatial_models_given():
    class FakeSpatialModel:
        def __init__(self, km):
            self._km = km

        def nearest_known_station_km(self, lat, lon):
            return self._km

    def base_predict(feats):
        assert "lat" in feats.columns and "lon" in feats.columns   # needed by a real SpatialKNNRegressor
        return pd.DataFrame({"bod": 1.0}, index=feats.index)

    spatial_models = {"bod": FakeSpatialModel(12.3), "do": FakeSpatialModel(4.5)}
    wrapped, meta = aoi.with_context(base_predict, lat=12.9, lon=77.6, date="2020-01-29",
                                     spatial_models=spatial_models)
    wrapped(pd.DataFrame({"ndci": [0.1]}))
    assert meta["nearest_station_km"] == pytest.approx(4.5)     # the minimum across served targets
