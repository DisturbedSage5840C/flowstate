"""Offline tests for the Sentinel-2 extraction logic (no network)."""

import datetime as dt
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data.satellite_extract import (
    ExtractionConfig,
    circle_mask,
    rank_items,
    reflectance_from_dn,
    summarize_window,
)

N = 100


def make_window(water_value=0.06, land_value=0.30):
    """Left half water (low NIR, MNDWI > 0), right half land (high NIR, MNDWI < 0)."""
    bands = {}
    for b in ("B2", "B3", "B4", "B5", "B6", "B8", "B11"):
        bands[b] = np.full((N, N), 0.05)
    # water: green >> swir, nir low
    bands["B3"][:, : N // 2] = 0.08
    bands["B11"][:, : N // 2] = 0.01
    bands["B8"][:, : N // 2] = 0.02
    bands["B4"][:, : N // 2] = water_value
    # land: swir > green, nir high
    bands["B3"][:, N // 2:] = 0.10
    bands["B11"][:, N // 2:] = 0.25
    bands["B8"][:, N // 2:] = land_value
    scl = np.full((N, N), 5, dtype="uint8")
    return bands, scl


def test_circle_mask_area():
    m = circle_mask(100, 50)
    assert abs(m.sum() - np.pi * 50 ** 2) / (np.pi * 50 ** 2) < 0.02
    assert not m[0, 0] and m[50, 50]


def test_summary_uses_only_water_pixels():
    bands, scl = make_window(water_value=0.06)
    s = summarize_window(bands, scl, ExtractionConfig())
    assert s["status"] == "ok"
    assert s["B4"] == pytest.approx(0.06) and s["B8"] == pytest.approx(0.02)        # land (0.30 NIR) excluded
    assert 0.3 < s["water_frac"] < 0.6


def test_clouds_shadows_and_snow_are_excluded():
    bands, scl = make_window()
    scl[:, : N // 2] = 5
    scl[:60, : N // 2] = 9                      # cloud over part of the water
    s = summarize_window(bands, scl, ExtractionConfig(max_cloud_frac=0.9))
    assert s["status"] == "ok" and s["n_water_px"] < 0.5 * s["n_circle_px"]


def test_too_cloudy_is_rejected():
    bands, scl = make_window()
    scl[:] = 8
    assert summarize_window(bands, scl, ExtractionConfig())["status"] == "cloudy"


def test_no_water_is_reported():
    bands, scl = make_window()
    bands["B11"][:] = 0.30                       # everything looks like land
    assert summarize_window(bands, scl, ExtractionConfig())["status"] == "no_water"


def test_turbid_water_labelled_vegetation_by_scl_is_still_used():
    bands, scl = make_window()
    scl[:, : N // 2] = 4                         # Sen2Cor calls the water "vegetation"
    assert summarize_window(bands, scl, ExtractionConfig())["status"] == "ok"


def test_min_water_pixels_threshold():
    bands, scl = make_window()
    assert summarize_window(bands, scl, ExtractionConfig(min_water_px=10 ** 6))["status"] == "no_water"


def test_nodata_zero_dn_becomes_nan():
    dn = np.array([[0, 1000], [2000, 0]], dtype="uint16")
    out = reflectance_from_dn(dn, "02.12")
    assert np.isnan(out[0, 0]) and out[0, 1] == pytest.approx(0.1) and out[1, 0] == pytest.approx(0.2)


def test_baseline_4_offset_applied():
    dn = np.array([[1000, 2000]], dtype="uint16")
    out = reflectance_from_dn(dn, "04.00")
    assert out[0, 0] == pytest.approx(0.0) and out[0, 1] == pytest.approx(0.1)
    assert reflectance_from_dn(dn, "05.10")[0, 1] == pytest.approx(0.1)


def _item(date, cloud):
    return SimpleNamespace(datetime=dt.datetime.combine(date, dt.time(5, 10), tzinfo=dt.timezone.utc),
                           properties={"eo:cloud_cover": cloud}, id=f"S2_{date}")


def test_rank_items_nearest_first_and_tolerance():
    target = dt.date(2021, 3, 10)
    items = [_item(dt.date(2021, 3, 5), 1), _item(dt.date(2021, 3, 9), 40), _item(dt.date(2021, 3, 12), 5),
             _item(dt.date(2021, 3, 14), 0)]
    ranked = rank_items(items, target, tolerance_days=3, limit=5)
    assert [d for _, d, _ in ranked] == [dt.date(2021, 3, 9), dt.date(2021, 3, 12)]      # 5 Mar and 14 Mar out of range
    assert [diff for diff, _, _ in ranked] == [1, 2]


def test_rank_items_tie_prefers_less_cloud():
    target = dt.date(2021, 3, 10)
    items = [_item(dt.date(2021, 3, 8), 50), _item(dt.date(2021, 3, 12), 3)]
    assert rank_items(items, target, 3, 1)[0][1] == dt.date(2021, 3, 12)


def test_item_crs_supports_both_projection_extension_versions():
    from src.data.satellite_extract import item_crs
    assert item_crs(SimpleNamespace(properties={"proj:code": "EPSG:32643"})) == "EPSG:32643"
    assert item_crs(SimpleNamespace(properties={"proj:epsg": 32644})) == "EPSG:32644"
    assert item_crs(SimpleNamespace(properties={})) is None
