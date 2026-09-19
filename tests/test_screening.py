"""Tests for the screening classifier and the per-target feature sets it shares with the regressors."""

import numpy as np
import pandas as pd
import pytest

from src.models.schema import FEATURE_SETS, RAINFALL_COLS, REAL_FEATURE_COLS, features_for
from src.models.tier_classifier import SCREENING_TARGETS, ScreeningClassifier, screening_label


# ---------------------------------------------------------------------------
# Per-target feature sets
# ---------------------------------------------------------------------------

def test_feature_sets_match_the_ablation_conclusions():
    assert features_for("do") == FEATURE_SETS["do"]
    # DO was best on spectral alone; BOD gained from urban context; turbidity kept only the type flags
    assert not any(c.startswith("is_") or c.startswith("dist_") or c.startswith("urban") for c in FEATURE_SETS["do"])
    assert "urban_load_index" in FEATURE_SETS["bod"] and "is_monsoon" in FEATURE_SETS["bod"]
    assert "is_river" in FEATURE_SETS["turbidity"] and "urban_load_index" not in FEATURE_SETS["turbidity"]


def test_rainfall_is_in_no_feature_set_but_the_union_covers_every_target():
    for target, cols in FEATURE_SETS.items():
        assert not set(cols) & set(RAINFALL_COLS), target      # measured as neutral-to-harmful
        assert set(cols) <= set(REAL_FEATURE_COLS), target
    assert not set(REAL_FEATURE_COLS) & set(RAINFALL_COLS)


def test_unknown_target_falls_back_to_the_union():
    assert features_for("chl_a") == REAL_FEATURE_COLS


# ---------------------------------------------------------------------------
# Screening labels
# ---------------------------------------------------------------------------

def test_labels_use_the_documented_direction_and_threshold():
    df = pd.DataFrame({"bod": [1.0, 3.0, 3.1, np.nan], "do": [8.0, 4.0, 3.9, 2.0],
                       "turbidity": [5.0, 10.0, 11.0, 1.0],
                       "cpcb_class": ["A", "C", "D", "Below E"]})
    assert screening_label(df, "bod_gt_3").tolist()[:3] == [0.0, 0.0, 1.0]       # strictly above the limit
    assert np.isnan(screening_label(df, "bod_gt_3").iloc[3])                     # unmeasured stays unlabelled
    assert screening_label(df, "do_lt_4").tolist() == [0.0, 0.0, 1.0, 1.0]       # below, not above
    assert screening_label(df, "turbidity_gt_10").tolist() == [0.0, 0.0, 1.0, 0.0]
    assert screening_label(df, "cpcb_polluted").tolist() == [0.0, 0.0, 1.0, 1.0]


def test_unknown_screening_target_raises():
    with pytest.raises(KeyError):
        ScreeningClassifier(["a"], "not_a_target")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _frame(n_sites=24, per_site=12, signal=True, seed=0):
    """Stations whose BOD depends on a feature, so a working classifier must beat the base rate."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sites):
        level = rng.normal()
        for v in range(per_site):
            x = level + rng.normal(scale=0.3)
            bod = np.exp(1.0 + (1.6 * x if signal else 0.0) + rng.normal(scale=0.3))
            rows.append({"site": f"s{s}", "lat": 10 + s * 0.7, "lon": 70 + s * 0.7,
                         "water_body_type": "lake", "f1": x, "f2": rng.normal(), "bod": bod})
    return pd.DataFrame(rows)


def test_separable_problem_scores_well_above_the_base_rate():
    clf = ScreeningClassifier(["f1", "f2"], "bod_gt_3", n_folds=4, random_state=0)
    res = clf.evaluate(_frame())
    assert res["roc_auc"] > 0.85
    assert res["lift_over_base_rate"] > 1.5
    assert res["precision_at_k"]["top_10pct"] > res["base_rate"]
    assert res["brier_score"] < res["brier_score_base_rate_guess"]          # better calibrated than guessing


def test_pure_noise_scores_near_chance():
    res = ScreeningClassifier(["f1", "f2"], "bod_gt_3", n_folds=4, random_state=0).evaluate(_frame(signal=False))
    assert 0.35 < res["roc_auc"] < 0.65
    assert res["lift_over_base_rate"] < 1.4


def test_reported_fields_and_row_selection():
    df = _frame()
    df.loc[:9, "bod"] = np.nan                                   # unlabelled rows must be dropped, not imputed
    df.loc[10:14, "f1"] = np.nan                                 # and so must rows missing a feature
    clf = ScreeningClassifier(["f1", "f2"], "bod_gt_3", n_folds=4, random_state=0)
    assert len(clf.frame(df)) == len(df) - 15
    res = clf.evaluate(df)
    assert res["n"] == len(df) - 15 and res["positives"] <= res["n"]
    assert set(res["precision_at_k"]) == {"top_5pct", "top_10pct", "top_20pct"}
    assert "accuracy" not in res                                  # deliberately not reported for imbalanced screening
    assert res["description"] == SCREENING_TARGETS["bod_gt_3"]["label"]


def test_out_of_fold_predictions_never_come_from_a_model_that_saw_the_station():
    df = _frame(n_sites=12, per_site=8)
    clf = ScreeningClassifier(["f1", "f2"], "bod_gt_3", n_folds=4, random_state=0)
    frame = clf.frame(df)
    p = clf.predict_oof(frame)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    # every station's rows are scored by exactly one fold's model, so a station is never in its own training set
    from src.models.spatial_cv import SpatialKFold
    for train_idx, val_idx in SpatialKFold(n_folds=4, random_state=0).split(frame):
        assert not set(frame["site"].iloc[train_idx]) & set(frame["site"].iloc[val_idx])
