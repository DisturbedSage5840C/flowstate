"""Offline tests for the Earth Engine extractor's decision logic (no Earth Engine calls)."""

import pandas as pd
import pytest

from src.data.gee_extract import BANDS, interpret_candidates
from src.data.satellite_extract import ExtractionConfig

T = int(pd.Timestamp("2020-01-29 05:10").value // 1_000_000)


def cand(dd=0, n_all=7800, n_water=3000, n_cloudy=0, **over):
    c = {"id": "20200129T051041_T43PGQ", "t": T, "dd": dd, "scene_cloud": 1.5, "n_all": n_all, "n_water": n_water,
         "n_cloudy": n_cloudy, **{b: 0.05 + 0.01 * i for i, b in enumerate(BANDS)}}
    c.update(over)
    return c


CFG = ExtractionConfig()


def test_no_candidates_means_no_scene():
    assert interpret_candidates([], CFG)["status"] == "no_scene"
    assert interpret_candidates(None, CFG)["status"] == "no_scene"


def test_first_clear_candidate_wins_and_matches_stac_schema():
    r = interpret_candidates([cand(dd=1)], CFG)
    assert r["status"] == "ok" and r["day_diff"] == 1 and r["scene_date"] == pd.Timestamp("2020-01-29")
    assert r["B2"] == pytest.approx(0.05) and r["B11"] == pytest.approx(0.11)
    assert r["n_water_px"] == 3000 and r["water_frac"] == pytest.approx(3000 / 7800) and r["cloud_frac"] == 0.0
    for key in ("scene_id", "scene_cloud", "sensor", "n_circle_px"):
        assert key in r


def test_cloudy_first_candidate_falls_through_to_the_next():
    r = interpret_candidates([cand(dd=0, n_cloudy=4000), cand(dd=2, id="second")], CFG)
    assert r["status"] == "ok" and r["scene_id"] == "second" and r["day_diff"] == 2


def test_all_cloudy_and_all_dry_report_the_most_informative_failure():
    assert interpret_candidates([cand(n_cloudy=5000)], CFG)["status"] == "cloudy"
    r = interpret_candidates([cand(n_water=3)], CFG)
    assert r["status"] == "no_water" and "B2" not in r


def test_missing_band_medians_count_as_no_water():
    c = cand()
    c["B5"] = None                                              # reducer returned nothing for a band
    assert interpret_candidates([c], CFG)["status"] == "no_water"


def test_zero_pixels_in_buffer_is_treated_as_cloudy_not_a_crash():
    assert interpret_candidates([cand(n_all=0, n_water=0)], CFG)["status"] == "cloudy"


def test_thresholds_follow_the_config():
    strict = ExtractionConfig(max_cloud_frac=0.01, min_water_px=5000)
    assert interpret_candidates([cand(n_cloudy=200)], strict)["status"] == "cloudy"
    assert interpret_candidates([cand(n_water=4000)], strict)["status"] == "no_water"


def test_compare_reports_agreement_on_identical_and_perturbed_inputs():
    from scripts.compare_backends import compare
    n = 12
    stac = pd.DataFrame({"station": [f"s{i}" for i in range(n)], "date": pd.Timestamp("2020-01-01"),
                         "status": "ok", "day_diff": 1.0, "n_water_px": 1000.0,
                         **{b: 0.05 + 0.002 * i for i in range(n) for b in [BANDS[0]]}})
    for j, b in enumerate(BANDS):
        stac[b] = [0.03 + 0.004 * i + 0.001 * j for i in range(n)]
    gee = stac.copy()
    for b in BANDS:
        gee[b] = stac[b] * 1.02                                  # a uniform 2 % difference
    gee["n_water_px"] = 1100.0
    r = compare(stac, gee)
    assert r["both_ok"] == n and r["same_day_diff_when_both_ok"] == 1.0
    assert r["bands"]["B4"]["median_rel_diff"] == pytest.approx(0.02, abs=1e-3) and r["bands"]["B4"]["pearson_r"] > 0.999
    assert r["n_water_px_median_ratio_gee_over_stac"] == pytest.approx(1.1)


def test_compare_ignores_mismatched_scenes():
    from scripts.compare_backends import compare
    base = {"station": [f"s{i}" for i in range(5)], "date": pd.Timestamp("2020-01-01"), "status": "ok",
            "n_water_px": 100.0, **{b: [0.05] * 5 for b in BANDS}}
    stac = pd.DataFrame({**base, "day_diff": 1.0})
    gee = pd.DataFrame({**base, "day_diff": 2.0})                # the backends picked different scenes
    r = compare(stac, gee)
    assert r["both_ok"] == 5 and r["same_day_diff_when_both_ok"] == 0.0 and r["bands"] == {}
