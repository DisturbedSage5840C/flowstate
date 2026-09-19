"""Hand-computed checks for src/features/feature_engineering.py."""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import (
    NECHAD_NIR,
    NECHAD_RED,
    compute_2bdm,
    compute_3bdm,
    compute_all_features,
    compute_chl_a_from_ndci,
    compute_do_surrogate,
    compute_mndwi,
    compute_ndci,
    compute_nechad_turbidity,
    compute_red_green_ratio,
    compute_turbidity_dogliotti,
)
from src.wqi.water_chemistry import do_saturation_mg_l


def f(x):
    return float(np.asarray(x).ravel()[0])


def test_ndci():
    assert f(compute_ndci(0.10, 0.06)) == pytest.approx(0.25, abs=1e-6)


def test_2bdm_is_705_over_665_ratio_as_in_plan():
    assert f(compute_2bdm(0.10, 0.05)) == pytest.approx(2.0, abs=1e-6)


def test_3bdm_needs_b6_and_never_substitutes():
    assert f(compute_3bdm(0.05, 0.10, 0.08)) == pytest.approx(0.8, abs=1e-5)   # (20 - 10) * 0.08
    assert np.isnan(f(compute_3bdm(np.array([0.05]), np.array([0.10]), None)))


def test_mndwi_and_red_green():
    assert f(compute_mndwi(0.06, 0.02)) == pytest.approx(0.5, abs=1e-6)
    assert f(compute_red_green_ratio(0.03, 0.06)) == pytest.approx(0.5, abs=1e-6)


def test_divide_by_zero_gives_nan():
    assert np.isnan(f(compute_ndci(0.0, 0.0)))


def test_chl_a_polynomial_and_clipping():
    assert f(compute_chl_a_from_ndci(0.25)) == pytest.approx(14.039 + 86.115 * 0.25 + 194.325 * 0.0625, abs=1e-3)
    # outside the validity range the input is clipped rather than extrapolated
    assert f(compute_chl_a_from_ndci(-0.9)) == pytest.approx(f(compute_chl_a_from_ndci(-0.1)))
    assert f(compute_chl_a_from_ndci(0.9)) == pytest.approx(f(compute_chl_a_from_ndci(0.5)))


def test_nechad_formula_and_saturation():
    a, c = NECHAD_RED["A"], NECHAD_RED["C"]
    assert f(compute_nechad_turbidity(0.05, a, c)) == pytest.approx(a * 0.05 / (1 - 0.05 / c), rel=1e-5)
    assert np.isnan(f(compute_nechad_turbidity(c + 0.01, a, c)))      # saturated -> undefined


def test_dogliotti_blend_endpoints():
    t_red = f(compute_nechad_turbidity(0.04, NECHAD_RED["A"], NECHAD_RED["C"]))
    assert f(compute_turbidity_dogliotti(0.04, 0.03)) == pytest.approx(t_red, rel=1e-5)     # w = 0
    t_nir = f(compute_nechad_turbidity(0.09, NECHAD_NIR["A"], NECHAD_NIR["C"]))
    assert f(compute_turbidity_dogliotti(0.08, 0.09)) == pytest.approx(t_nir, rel=1e-5)     # w = 1


@pytest.mark.parametrize("temp", [20, 25, 30, 35])
def test_do_surrogate_equals_saturation_for_clean_water(temp):
    out = f(compute_do_surrogate(5.0, 10.0, month=1, t_water_c=temp))
    assert out == pytest.approx(float(do_saturation_mg_l(temp)), abs=1e-3)


def test_do_surrogate_no_longer_collapses_in_hot_water():
    # the old linear formula gave ~1 mg/L at 35 C; saturation there is ~6.95
    assert f(compute_do_surrogate(5.0, 10.0, month=1, t_water_c=35)) > 6.5


def test_do_surrogate_depression_monsoon_and_altitude():
    base = f(compute_do_surrogate(5.0, 10.0, month=1, t_water_c=25))
    assert f(compute_do_surrogate(120.0, 10.0, month=1, t_water_c=25)) == pytest.approx(base - 1.5, abs=1e-3)
    assert f(compute_do_surrogate(5.0, 210.0, month=1, t_water_c=25)) == pytest.approx(base - 0.8, abs=1e-3)
    assert f(compute_do_surrogate(5.0, 10.0, month=7, t_water_c=25)) == pytest.approx(base + 0.5, abs=1e-3)
    assert f(compute_do_surrogate(5.0, 10.0, month=1, t_water_c=25, altitude_m=1580)) < base * 0.85


def _s2_frame(n=3, **extra):
    d = {"B2": [0.03] * n, "B3": [0.05] * n, "B4": [0.04] * n, "B5": [0.06] * n,
         "B8": [0.03] * n, "B11": [0.01] * n}
    d.update(extra)
    return pd.DataFrame(d)


def test_all_features_s2_columns_and_no_target_named_columns():
    out = compute_all_features(_s2_frame(B6=[0.05] * 3))
    for c in ("ndci", "bdm2", "bdm3", "red_green", "mndwi", "nir",
              "turbidity_empirical", "chl_a_empirical", "do_empirical"):
        assert c in out.columns
    assert not {"chl_a", "turbidity", "do"} & set(out.columns)     # would collide with model targets
    assert out["ndci"].iloc[0] == pytest.approx(0.2, abs=1e-6)


def test_all_features_bdm3_nan_without_b6():
    assert compute_all_features(_s2_frame())["bdm3"].isna().all()


def test_missing_bands_raise_a_clear_error():
    river_like = _s2_frame().drop(columns=["B5", "B11"])
    with pytest.raises(KeyError, match="missing"):
        compute_all_features(river_like)


def test_landsat_path_never_treats_b5_as_red_edge():
    ls = pd.DataFrame({"B3": [0.05], "B4": [0.04], "B5": [0.30], "B6": [0.01]})     # B5 = NIR here
    out = compute_all_features(ls, sensor="LS")
    assert "ndci" not in out.columns and "chl_a_empirical" not in out.columns
    assert out["nir"].iloc[0] == pytest.approx(0.30, abs=1e-6)
    assert out["mndwi"].iloc[0] == pytest.approx((0.05 - 0.01) / (0.05 + 0.01), abs=1e-6)


def test_row_level_month_and_temperature_drive_do():
    df = _s2_frame(3, date=["2024-01-10", "2024-07-10", "2024-01-10"], temp_surface=[25.0, 25.0, 35.0])
    out = compute_all_features(df)
    do = out["do_empirical"].to_numpy()
    assert do[1] > do[0]          # monsoon bonus applies per row, not once for the whole frame
    assert do[2] < do[0]          # warmer water holds less oxygen
