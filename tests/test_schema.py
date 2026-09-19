"""tests/test_schema.py — unit tests for src/models/schema.py's per-target FEATURE_SETS."""

from __future__ import annotations

from src.models import schema


def test_feature_sets_cover_do_turbidity_bod_with_increasing_context():
    assert set(schema.FEATURE_SETS) == {"do", "turbidity", "bod"}
    do, turb, bod = schema.FEATURE_SETS["do"], schema.FEATURE_SETS["turbidity"], schema.FEATURE_SETS["bod"]
    assert set(do) == set(schema.SPECTRAL_COLS)                                  # DO: spectral only
    assert set(turb) == set(schema.SPECTRAL_COLS) | set(schema.TYPE_COLS)        # turbidity: + water-body type
    assert set(bod) == (set(schema.SPECTRAL_COLS) | set(schema.TYPE_COLS)
                        | set(schema.SEASON_COLS) | set(schema.URBAN_PROXY_COLS))  # BOD: + season + urban
    assert set(do) < set(turb) < set(bod)                                         # strictly increasing context


def test_rainfall_columns_are_in_no_feature_set():
    all_feature_cols = set().union(*schema.FEATURE_SETS.values())
    assert not set(schema.RAINFALL_COLS) & all_feature_cols
    assert not set(schema.RAINFALL_COLS) & set(schema.REAL_FEATURE_COLS)


def test_real_feature_cols_is_the_union_of_every_feature_set():
    union = set().union(*schema.FEATURE_SETS.values())
    assert set(schema.REAL_FEATURE_COLS) == union
    assert len(schema.REAL_FEATURE_COLS) == len(set(schema.REAL_FEATURE_COLS))    # no duplicates
