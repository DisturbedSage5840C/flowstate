"""tests/test_tier_classifier.py — unit tests for src/models/tier_classifier.py: the binary
pollution-screening classifier (headline deliverable) and its leakage-free target derivation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.tier_classifier import (
    ScreeningClassifier,
    WQITierClassifier,
    add_screening_targets,
)


def _separable_df(n_sites=15, per_site=8, seed=0):
    """Station-level signal drives both the features and the label, so a model that learns "which
    station" is effectively separable -- the same structure this project's real ablations found
    (most of BOD's variance is between-station, see src/models/schema.py)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_sites):
        lat, lon = 10 + i * 1.3, 70 + i * 1.1
        site_signal = rng.uniform(-3, 3)
        for _ in range(per_site):
            f1 = site_signal + rng.normal(0, 0.3)
            f2 = rng.normal(0, 1)               # uninformative
            bod = max(0.1, 3.0 + 2.0 * site_signal + rng.normal(0, 0.5))
            rows.append({"site": f"s{i}", "lat": lat, "lon": lon, "f1": f1, "f2": f2, "bod": bod})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# add_screening_targets: leakage-free binary labels from measured chemistry only
# ---------------------------------------------------------------------------

def test_add_screening_targets_derives_binary_labels_and_preserves_nan():
    df = pd.DataFrame({
        "bod": [1.0, 5.0, np.nan, 8.0],
        "do": [6.0, 3.5, 5.0, np.nan],
        "cpcb_class": ["B", "D", None, "Below E"],
    })
    out = add_screening_targets(df)
    np.testing.assert_array_equal(out["bod_gt_3"].to_numpy(), np.array([0.0, 1.0, np.nan, 1.0]))
    np.testing.assert_array_equal(out["bod_gt_6"].to_numpy(), np.array([0.0, 0.0, np.nan, 1.0]))
    np.testing.assert_array_equal(out["do_lt_4"].to_numpy(), np.array([0.0, 1.0, 0.0, np.nan]))
    np.testing.assert_array_equal(out["cpcb_polluted"].to_numpy(), np.array([0.0, 1.0, np.nan, 1.0]))


def test_screening_targets_are_never_inputs_to_themselves():
    # regression guard: none of the derived target names collide with the chemistry columns they're built from
    df = pd.DataFrame({"bod": [1.0], "do": [5.0], "cpcb_class": ["A"]})
    out = add_screening_targets(df)
    from src.models.tier_classifier import SCREENING_TARGETS
    assert not set(SCREENING_TARGETS) & {"bod", "do", "cpcb_class"}
    assert set(SCREENING_TARGETS) <= set(out.columns)


# ---------------------------------------------------------------------------
# ScreeningClassifier: AUC / lift on a synthetic, separable set
# ---------------------------------------------------------------------------

def test_screening_classifier_shows_high_auc_and_lift_on_a_separable_synthetic_set():
    df = _separable_df()
    df["bod_gt_3"] = (df["bod"] > 3.0).astype(int)
    assert df["bod_gt_3"].nunique() == 2

    clf = ScreeningClassifier(feature_cols=["f1", "f2"], target="bod_gt_3", n_folds=3, random_state=0)
    result = clf.evaluate(df)

    assert result["n"] == len(df)
    assert 0.0 < result["base_rate"] < 1.0
    assert result["auc"] > 0.85
    assert result["lift_vs_base_rate"] > 1.5          # average precision well above the base rate
    assert result["average_precision"] > result["base_rate"]
    assert result["precision_at_k"]["10%"] >= result["base_rate"]
    assert len(result["calibration"]) > 0
    for b in result["calibration"]:
        assert {"mean_predicted", "mean_actual", "n"} <= set(b)


def test_screening_classifier_does_worse_on_an_uninformative_feature():
    df = _separable_df(seed=2)
    df["bod_gt_3"] = (df["bod"] > 3.0).astype(int)
    rng = np.random.default_rng(3)
    df["noise"] = rng.normal(size=len(df))

    informative = ScreeningClassifier(feature_cols=["f1"], target="bod_gt_3", n_folds=3, random_state=0).evaluate(df)
    uninformative = ScreeningClassifier(feature_cols=["noise"], target="bod_gt_3", n_folds=3, random_state=0).evaluate(df)
    assert informative["auc"] > uninformative["auc"]
    assert informative["auc"] > 0.8
    assert uninformative["auc"] < 0.7


def test_screening_classifier_fit_and_predict_proba():
    df = _separable_df(seed=4)
    df["bod_gt_3"] = (df["bod"] > 3.0).astype(int)
    clf = ScreeningClassifier(feature_cols=["f1", "f2"], target="bod_gt_3", n_folds=3, random_state=0).fit(df)
    proba = clf.predict_proba(df)
    assert len(proba) == len(df) and ((proba >= 0) & (proba <= 1)).all()
    with pytest.raises(RuntimeError):
        ScreeningClassifier(feature_cols=["f1"], target="bod_gt_3").predict_proba(df)


def test_screening_classifier_handles_a_degenerate_single_class_target():
    df = _separable_df(seed=5)
    df["always_zero"] = 0
    result = ScreeningClassifier(feature_cols=["f1"], target="always_zero", n_folds=3, random_state=0).evaluate(df)
    assert np.isnan(result["auc"]) and np.isnan(result["average_precision"])


# ---------------------------------------------------------------------------
# WQITierClassifier: still works as the secondary 5-class deliverable
# ---------------------------------------------------------------------------

def test_wqi_tier_classifier_evaluate_reports_model_and_baseline():
    rng = np.random.default_rng(6)
    n_sites = 12
    rows = []
    tiers = ["Excellent", "Good", "Moderate", "Poor", "Very Poor"]
    for i in range(n_sites):
        tier = tiers[i % len(tiers)]
        for _ in range(6):
            rows.append({"site": f"s{i}", "lat": 10 + i, "lon": 70 + i,
                        "f1": tiers.index(tier) + rng.normal(0, 0.2), "wqi_tier": tier})
    df = pd.DataFrame(rows)
    clf = WQITierClassifier(feature_cols=["f1"], target="wqi_tier", n_folds=3, random_state=0)
    result = clf.evaluate(df)
    assert result["target"] == "wqi_tier" and result["n"] == len(df)
    assert set(result["classes"]) == set(tiers)
    assert "accuracy" in result["model"] and "macro_f1" in result["model"]
    assert "accuracy" in result["majority_class_baseline"]
