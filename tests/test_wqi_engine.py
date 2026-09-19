"""Unit tests for the WQI engine and CPCB best-use classifier (hand-computed cases)."""

import numpy as np
import pandas as pd
import pytest

from src.wqi.wqi_engine import (
    CPCB_BELOW_E,
    PARAMETERS,
    WQI_TIERS,
    classify_cpcb_best_use,
    compute_wqi,
    compute_wqi_dataframe,
    wqi_tier_label,
)
from src.wqi.water_chemistry import do_saturation_mg_l, free_ammonia_n


# ---------------------------------------------------------------------------
# Water chemistry helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("temp, expected", [(0, 14.62), (10, 11.29), (20, 9.09), (25, 8.26),
                                             (30, 7.56), (35, 6.95)])
def test_do_saturation_matches_standard_methods_table(temp, expected):
    assert float(do_saturation_mg_l(temp)) == pytest.approx(expected, abs=0.02)


def test_do_saturation_drops_with_altitude():
    assert float(do_saturation_mg_l(20, 1580)) < 0.9 * float(do_saturation_mg_l(20, 0))


def test_free_ammonia_fraction_rises_with_ph():
    lo = float(free_ammonia_n(1.0, 7.0, 25))
    hi = float(free_ammonia_n(1.0, 8.5, 25))
    assert lo < 0.01 and 0.14 < hi < 0.17


# ---------------------------------------------------------------------------
# WQI tiers
# ---------------------------------------------------------------------------

def test_tiers_cover_0_to_100_contiguously():
    assert WQI_TIERS[0].lower == 0.0 and WQI_TIERS[-1].upper == 100.0
    for a, b in zip(WQI_TIERS, WQI_TIERS[1:]):
        assert a.upper == b.lower


@pytest.mark.parametrize("score, label", [(0, "Excellent"), (19.9, "Excellent"), (20, "Good"),
                                           (59.9, "Moderate"), (60, "Poor"), (80, "Very Poor"),
                                           (100, "Very Poor")])
def test_tier_boundaries(score, label):
    assert wqi_tier_label(score) == label


def test_nan_has_no_tier():
    assert wqi_tier_label(float("nan")) is None


# ---------------------------------------------------------------------------
# WQI index
# ---------------------------------------------------------------------------

def test_hand_computed_two_parameter_case():
    # DO at the standard (5 mg/L, 25 C) -> Qi 100, weight 1/5; BOD 0 -> Qi 0, weight 1/3.
    # WQI = (0.2*100 + (1/3)*0) / (0.2 + 1/3) = 37.5
    r = compute_wqi(do=5.0, bod=0.0)
    assert r["wqi"] == pytest.approx(37.5, abs=1e-6)
    assert r["sub_index"]["do"] == pytest.approx(100.0)
    assert r["sub_index"]["bod"] == pytest.approx(0.0)
    assert r["n_params"] == 2
    assert sum(r["weights"].values()) == pytest.approx(1.0)


def test_all_parameters_at_standard_gives_100():
    r = compute_wqi(do=5.0, bod=3.0, ph=8.5, turbidity=5.0, chl_a=10.0)
    assert r["wqi"] == pytest.approx(100.0)
    assert r["wqi_tier"] == "Very Poor"


def test_pristine_water_is_excellent():
    sat = float(do_saturation_mg_l(25))
    r = compute_wqi(do=sat, bod=0.0, ph=7.0, turbidity=0.0, chl_a=0.0)
    assert r["wqi"] < 1.0 and r["wqi_tier"] == "Excellent"


def test_missing_parameters_are_skipped_not_imputed():
    with_ph = compute_wqi(do=5.0, bod=0.0, ph=None)
    assert with_ph["wqi"] == pytest.approx(37.5, abs=1e-6)
    assert "ph" not in with_ph["sub_index"]


def test_too_few_parameters_gives_nan():
    r = compute_wqi(do=6.0)
    assert np.isnan(r["wqi"]) and r["wqi_tier"] is None


def test_index_is_bounded_even_for_extreme_pollution():
    r = compute_wqi(do=0.1, bod=500.0, ph=12.0, turbidity=5000.0, chl_a=900.0, total_coliform=1e7)
    assert 0.0 <= r["wqi"] <= 100.0
    assert all(0.0 <= q <= 100.0 for q in r["sub_index"].values())


@pytest.mark.parametrize("worse_kw, better_kw", [
    (dict(do=2.0, bod=3.0), dict(do=7.0, bod=3.0)),
    (dict(do=6.0, bod=10.0), dict(do=6.0, bod=2.0)),
    (dict(do=6.0, bod=2.0, turbidity=80.0), dict(do=6.0, bod=2.0, turbidity=3.0)),
    (dict(do=6.0, bod=2.0, chl_a=90.0), dict(do=6.0, bod=2.0, chl_a=4.0)),
])
def test_more_pollution_raises_wqi(worse_kw, better_kw):
    assert compute_wqi(**worse_kw)["wqi"] > compute_wqi(**better_kw)["wqi"]


def test_ideal_do_depends_on_temperature():
    # The same 6 mg/L is less alarming in 35 C water (saturation 6.95) than in 25 C water (8.26).
    hot = compute_wqi(do=6.0, bod=0.0, temp_c=35)["sub_index"]["do"]
    mild = compute_wqi(do=6.0, bod=0.0, temp_c=25)["sub_index"]["do"]
    assert hot == pytest.approx(48.7, abs=0.2) and mild == pytest.approx(69.4, abs=0.2)


def test_dataframe_matches_scalar_and_handles_partial_rows():
    df = pd.DataFrame({
        "do":  [5.0, 8.0, np.nan],
        "bod": [0.0, 1.0, 2.0],
        "ph":  [np.nan, 7.5, np.nan],
    })
    out = compute_wqi_dataframe(df)
    assert out.loc[0, "wqi"] == pytest.approx(37.5, abs=1e-6)
    assert out.loc[1, "wqi"] == pytest.approx(compute_wqi(do=8.0, bod=1.0, ph=7.5)["wqi"])
    assert np.isnan(out.loc[2, "wqi"]) and pd.isna(out.loc[2, "wqi_tier"])   # only BOD available
    assert set(f"qi_{k}" for k in PARAMETERS) <= set(out.columns)
    assert out["wqi"].dropna().between(0, 100).all()


def test_dataframe_uses_temperature_column():
    base = pd.DataFrame({"do": [6.0], "bod": [0.0]})
    hot = compute_wqi_dataframe(base.assign(temp_c=35.0))["wqi"].iloc[0]
    mild = compute_wqi_dataframe(base)["wqi"].iloc[0]
    assert hot < mild


def test_no_invented_inputs_when_columns_absent():
    out = compute_wqi_dataframe(pd.DataFrame({"chl_a": [5.0], "turbidity": [3.0]}))
    assert out["n_wqi_params"].iloc[0] == 2
    assert np.isnan(out["qi_bod"].iloc[0]) and np.isnan(out["qi_do"].iloc[0])


# ---------------------------------------------------------------------------
# CPCB designated-best-use class
# ---------------------------------------------------------------------------

def _classify(**kw):
    return classify_cpcb_best_use(pd.DataFrame([kw])).iloc[0]


def test_class_a_full_criteria():
    r = _classify(do=7.0, bod=1.0, ph=7.5, total_coliform=20)
    assert r["cpcb_class"] == "A" and bool(r["cpcb_class_complete"]) and r["cpcb_criteria_assessed"] == 4


def test_class_b_when_do_below_6():
    assert _classify(do=5.5, bod=2.5, ph=7.5, total_coliform=300)["cpcb_class"] == "B"


def test_class_c():
    assert _classify(do=4.5, bod=3.0, ph=8.0, total_coliform=4000)["cpcb_class"] == "C"


def test_class_d_partial_without_ammonia():
    r = _classify(do=4.2, bod=20.0, ph=7.5)
    assert r["cpcb_class"] == "D" and not bool(r["cpcb_class_complete"])


def test_free_ammonia_derived_from_total_ammonia_pushes_class_below_d():
    # 10 mg/L total ammonia-N at pH 8.5 / 25 C -> ~1.5 mg/L free ammonia > 1.2 (Class D limit)
    r = _classify(do=4.5, bod=20.0, ph=8.5, ammonia_n=10.0, conductivity=900.0)
    assert r["cpcb_class"] == "E"


def test_class_e_when_do_too_low_for_d():
    assert _classify(do=1.0, bod=50.0, ph=7.0, conductivity=800.0, sar=5.0, boron=0.5)["cpcb_class"] == "E"


def test_below_e_when_even_irrigation_criteria_fail():
    assert _classify(do=1.0, bod=50.0, ph=5.0, conductivity=800.0)["cpcb_class"] == CPCB_BELOW_E


def test_indeterminate_with_a_single_measurement():
    assert pd.isna(_classify(do=7.0)["cpcb_class"])


def test_cpcb_class_is_independent_of_wqi_score():
    df = pd.DataFrame({"do": [7.0], "bod": [1.0], "ph": [7.5], "total_coliform": [20]})
    out = classify_cpcb_best_use(compute_wqi_dataframe(df))
    assert out["cpcb_class"].iloc[0] == "A" and out["wqi_tier"].iloc[0] in {t.label for t in WQI_TIERS}
