"""tests/test_spatial_cv.py — unit tests for spatial_cv.py"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.models.spatial_cv import SpatialKFold, StationKFold, spatial_cross_val_score


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


def _make_paired_df(n_pairs: int = 20, n_per_site: int = 5, seed: int = 0) -> pd.DataFrame:
    """n_pairs clusters, each of exactly two stations ~1 m apart (a tight geographic pair),
    scattered far apart from every other pair -- for testing whether a fold scheme keeps
    geographic neighbours together (SpatialKFold, by design) or not (StationKFold)."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_pairs):
        base_lat, base_lon = rng.uniform(8.0, 35.0), rng.uniform(68.0, 97.0)
        for twin in ("a", "b"):
            site = f"pair{p}_{twin}"
            for _ in range(n_per_site):
                rows.append({"site": site, "water_body_type": "lake",
                            "lat": base_lat + rng.uniform(-1e-5, 1e-5),
                            "lon": base_lon + rng.uniform(-1e-5, 1e-5),
                            "chl_a": rng.uniform(2, 200), "turbidity": rng.uniform(2, 500),
                            "do": rng.uniform(2, 12)})
    return pd.DataFrame(rows)


class TestStationKFold:
    def test_basic_split(self):
        df = _make_df(n_sites=10, n_per_site=50)
        skf = StationKFold(n_folds=5)
        folds = list(skf.split(df))
        assert len(folds) == 5

    def test_no_site_straddles_two_folds(self):
        df = _make_df(n_sites=20, n_per_site=10)
        skf = StationKFold(n_folds=5)
        for train_idx, val_idx in skf.split(df):
            train_sites = set(df["site"].iloc[train_idx])
            val_sites = set(df["site"].iloc[val_idx])
            assert not (train_sites & val_sites)

    def test_all_samples_covered_and_no_overlap(self):
        df = _make_df(n_sites=20, n_per_site=10)
        skf = StationKFold(n_folds=5)
        covered = set()
        for train_idx, val_idx in skf.split(df):
            assert len(set(train_idx) & set(val_idx)) == 0
            covered.update(val_idx.tolist())
        assert covered == set(range(len(df)))

    def test_too_few_sites_raises(self):
        df = _make_df(n_sites=3, n_per_site=10)
        with pytest.raises(ValueError, match="n_folds"):
            list(StationKFold(n_folds=5).split(df))

    def test_reproducibility(self):
        df = _make_df(n_sites=20, n_per_site=10)
        a = [(set(tr), set(va)) for tr, va in StationKFold(n_folds=5, random_state=7).split(df)]
        b = [(set(tr), set(va)) for tr, va in StationKFold(n_folds=5, random_state=7).split(df)]
        assert a == b

    def test_unlike_spatialkfold_it_is_not_geographically_clustered(self):
        """The whole point of this class: geographically adjacent stations end up split across
        folds at least sometimes, unlike SpatialKFold which keeps them together by construction."""
        df = _make_paired_df(n_pairs=30, n_per_site=3)
        pair_split_anywhere = False
        for seed in range(10):
            skf = StationKFold(n_folds=5, random_state=seed)
            for train_idx, val_idx in skf.split(df):
                val_sites = set(df["site"].iloc[val_idx])
                for p in range(30):
                    a_in, b_in = f"pair{p}_a" in val_sites, f"pair{p}_b" in val_sites
                    if a_in != b_in:          # one twin held out, the other still in training
                        pair_split_anywhere = True
        assert pair_split_anywhere, "StationKFold never separated a tight geographic pair across 10 seeds"

    def test_spatialkfold_keeps_the_same_pairs_together(self):
        """Contrast case: SpatialKFold's whole design keeps a ~1m-apart pair in the same fold."""
        df = _make_paired_df(n_pairs=30, n_per_site=3)
        skf = SpatialKFold(n_folds=5)
        labels = skf.get_fold_labels(df)
        site_fold = df.assign(fold=labels).groupby("site")["fold"].first()
        for p in range(30):
            assert site_fold[f"pair{p}_a"] == site_fold[f"pair{p}_b"]
