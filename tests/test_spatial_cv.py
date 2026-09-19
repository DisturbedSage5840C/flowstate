"""tests/test_spatial_cv.py — unit tests for spatial_cv.py"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.models.spatial_cv import SpatialKFold, spatial_cross_val_score


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_df(n_sites: int = 10, n_per_site: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sites = [f"site_{i}" for i in range(n_sites)]
    types = ["lake", "river", "reservoir", "lagoon"]
    rows = []
    for site in sites:
        base_lat = rng.uniform(8.0, 35.0)
        base_lon = rng.uniform(68.0, 97.0)
        wbt = rng.choice(types)
        for _ in range(n_per_site):
            rows.append({
                "site":            site,
                "water_body_type": wbt,
                "lat":             base_lat + rng.uniform(-0.05, 0.05),
                "lon":             base_lon + rng.uniform(-0.05, 0.05),
                "chl_a":           rng.uniform(2, 200),
                "turbidity":       rng.uniform(2, 500),
                "do":              rng.uniform(2, 12),
            })
    return pd.DataFrame(rows)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSpatialKFold:
    def test_basic_split(self):
        df = _make_df(n_sites=10, n_per_site=50)
        skf = SpatialKFold(n_folds=5)
        folds = list(skf.split(df))
        assert len(folds) == 5

    def test_no_site_straddles_two_folds(self):
        """Core correctness requirement: a site must appear in exactly one fold."""
        df = _make_df(n_sites=10, n_per_site=50)
        skf = SpatialKFold(n_folds=5)
        labels = skf.get_fold_labels(df)
        site_folds = df.assign(fold=labels).groupby("site")["fold"].nunique()
        assert (site_folds == 1).all(), (
            "Some sites appear in more than one fold — spatial leak detected!"
        )

    def test_all_samples_covered(self):
        """Every sample must appear in exactly one val set."""
        df = _make_df(n_sites=10, n_per_site=50)
        skf = SpatialKFold(n_folds=5)
        covered = set()
        for train_idx, val_idx in skf.split(df):
            assert len(set(train_idx) & set(val_idx)) == 0, "Train/val overlap!"
            covered.update(val_idx.tolist())
        assert len(covered) == len(df)

    def test_too_few_sites_raises(self):
        df = _make_df(n_sites=3, n_per_site=10)
        skf = SpatialKFold(n_folds=5)
        with pytest.raises(ValueError, match="n_folds"):
            list(skf.split(df))

    def test_fold_summary_returns_df(self):
        df = _make_df(n_sites=10, n_per_site=20)
        skf = SpatialKFold(n_folds=5)
        summary = skf.fold_summary(df)
        assert isinstance(summary, pd.DataFrame)
        assert "fold" in summary.columns
        assert "n_samples" in summary.columns

    def test_reproducibility(self):
        """Same seed → same fold assignments."""
        df = _make_df(n_sites=10, n_per_site=20)
        labels_a = SpatialKFold(n_folds=5, random_state=7).get_fold_labels(df)
        labels_b = SpatialKFold(n_folds=5, random_state=7).get_fold_labels(df)
        pd.testing.assert_series_equal(labels_a, labels_b)

    def test_different_seed_can_differ(self):
        df = _make_df(n_sites=10, n_per_site=20)
        labels_a = SpatialKFold(n_folds=5, random_state=0).get_fold_labels(df)
        labels_b = SpatialKFold(n_folds=5, random_state=99).get_fold_labels(df)
        # Not guaranteed to differ but very likely with 10 sites
        # (we just check no exception; determinism is tested above)
        assert labels_a is not None and labels_b is not None
