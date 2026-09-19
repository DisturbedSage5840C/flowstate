import numpy as np
import pytest

from src.data.landsat_temp import (
    QA_WATER_BIT,
    ST_OFFSET,
    ST_SCALE,
    LandsatTempExtractor,
    kelvin_from_dn,
    summarize_temperature,
)

N = 20
CLEAR_WATER = (1 << QA_WATER_BIT) | (1 << 6)          # water + "clear" flag, no cloud bits


def window(temp_c=27.0, qa_value=CLEAR_WATER):
    return np.full((N, N), temp_c + 273.15), np.full((N, N), qa_value, dtype="uint16")


def test_dn_to_kelvin_and_nodata():
    dn = np.array([0, 44000], dtype="uint16")
    k = kelvin_from_dn(dn)
    assert np.isnan(k[0]) and k[1] == pytest.approx(44000 * ST_SCALE + ST_OFFSET)
    assert 290 < k[1] < 310                              # a plausible surface temperature (~20-35 C)


def test_median_water_temperature_in_celsius():
    st, qa = window(27.0)
    r = summarize_temperature(st, qa)
    assert r["temp_status"] == "ok" and r["temp_c"] == pytest.approx(27.0, abs=1e-6)


def test_land_cloud_and_implausible_pixels_are_excluded():
    st, qa = window(27.0)
    st[:, :10] = 340.0                                  # hot land / rock: not water (QA says land)
    qa[:, :10] = 1 << 6                                 # clear but NOT water
    qa[:5, 10:] |= 1 << 3                               # cloud over part of the water
    st[15:, 10:] = 200.0                                # implausible retrieval
    r = summarize_temperature(st, qa)
    assert r["temp_status"] == "ok" and r["temp_c"] == pytest.approx(27.0, abs=1e-6)
    assert r["temp_n_px"] < 0.5 * N * N


def test_no_clear_water_gives_no_value_not_a_constant():
    st, qa = window(27.0, qa_value=1 << 6)             # no water flag anywhere
    r = summarize_temperature(st, qa)
    assert r["temp_status"] == "no_clear_water" and "temp_c" not in r


def test_extractor_uses_landsat_collection_and_five_day_tolerance():
    ex = LandsatTempExtractor(workers=1)
    assert ex.collection == "landsat-c2-l2" and ex.cfg.day_tolerance == 5


def test_pixels_outside_liquid_water_range_are_ignored():
    st, qa = window(27.0)
    st[:, :10] = 318.0                                   # 44.9 C: not plausible for water
    r = summarize_temperature(st, qa)
    assert r["temp_c"] == pytest.approx(27.0, abs=1e-6)
