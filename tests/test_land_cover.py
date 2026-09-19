"""tests/test_land_cover.py — unit tests for src/data/land_cover.py (ESA WorldCover fractions). No real
network calls: the STAC search is a fake catalog, and the classification "window" is a synthetic array
rather than a real raster read."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data import land_cover


class FakeItem:
    def __init__(self, item_id, start_datetime=""):
        self.id = item_id
        self.properties = {"start_datetime": start_datetime}
        self.assets = {"map": SimpleNamespace(href="https://example.invalid/map.tif")}


class FakeCatalog:
    def __init__(self, items):
        self._items = items

    def search(self, collections, intersects, limit):
        return SimpleNamespace(items=lambda: iter(self._items))


# ---------------------------------------------------------------------------
# search_worldcover_item
# ---------------------------------------------------------------------------

def test_search_returns_none_when_no_coverage():
    assert land_cover.search_worldcover_item(1.0, 1.0, catalog=FakeCatalog([])) is None


def test_search_prefers_the_newest_release_year():
    old, new = FakeItem("2020", "2020-01-01"), FakeItem("2021", "2021-01-01")
    item = land_cover.search_worldcover_item(1.0, 1.0, catalog=FakeCatalog([old, new]))
    assert item.id == "2021"


# ---------------------------------------------------------------------------
# landcover_fractions: pure array logic, no network
# ---------------------------------------------------------------------------

def test_landcover_fractions_computes_correct_percentages():
    # 40 = cropland, 50 = built, 10 = tree, 0 = nodata (excluded from the denominator)
    arr = np.array([40, 40, 40, 50, 10, 0, 0])
    out = land_cover.landcover_fractions(arr)
    assert out["landcover_cropland_pct"] == pytest.approx(3 / 5)
    assert out["landcover_built_pct"] == pytest.approx(1 / 5)
    assert out["landcover_tree_pct"] == pytest.approx(1 / 5)


def test_landcover_fractions_all_nodata_returns_nan():
    out = land_cover.landcover_fractions(np.zeros(9, dtype=int))
    assert all(v != v for v in out.values())      # every reported class is NaN


def test_landcover_fractions_keys_match_reported_classes():
    out = land_cover.landcover_fractions(np.array([40, 50, 10]))
    assert set(out) == {f"landcover_{c}_pct" for c in land_cover.REPORTED_CLASSES}


# ---------------------------------------------------------------------------
# fetch_land_cover_for_sites: resumable cache / offline mode
# ---------------------------------------------------------------------------

def test_fetch_land_cover_for_sites_skips_already_cached_stations(monkeypatch, tmp_path):
    cache_path = tmp_path / "landcover.parquet"
    pd.DataFrame({"site": ["a"], "landcover_cropland_pct": [0.5], "landcover_built_pct": [0.1],
                 "landcover_tree_pct": [0.4]}).to_parquet(cache_path)

    calls = []

    def fake_search(lat, lon, catalog=None):
        calls.append((lat, lon))
        return FakeItem("x")

    monkeypatch.setattr(land_cover, "search_worldcover_item", fake_search)
    monkeypatch.setattr(land_cover, "read_classification_window", lambda *a, **k: np.array([40, 50, 10]))

    sites = pd.DataFrame({"site": ["a", "b"], "lat": [1.0, 2.0], "lon": [1.0, 2.0]})
    result = land_cover.fetch_land_cover_for_sites(sites, cache_path=cache_path)

    assert len(calls) == 1                         # only "b" fetched
    assert set(result["site"]) == {"a", "b"}
    assert result.set_index("site").loc["a", "landcover_cropland_pct"] == 0.5   # cached value preserved


def test_fetch_land_cover_for_sites_offline_returns_cache_with_no_network_calls(monkeypatch, tmp_path):
    cache_path = tmp_path / "landcover.parquet"
    pd.DataFrame({"site": ["a"], "landcover_cropland_pct": [0.5], "landcover_built_pct": [0.1],
                 "landcover_tree_pct": [0.4]}).to_parquet(cache_path)

    def fail(*a, **k):
        raise AssertionError("should not be called in offline mode")

    monkeypatch.setattr(land_cover, "search_worldcover_item", fail)
    sites = pd.DataFrame({"site": ["a", "b"], "lat": [1.0, 2.0], "lon": [1.0, 2.0]})
    result = land_cover.fetch_land_cover_for_sites(sites, cache_path=cache_path, offline=True)
    assert list(result["site"]) == ["a"]


def test_fetch_land_cover_for_sites_skips_a_point_with_no_coverage(monkeypatch, tmp_path):
    monkeypatch.setattr(land_cover, "search_worldcover_item", lambda lat, lon, catalog=None: None)
    sites = pd.DataFrame({"site": ["a"], "lat": [1.0], "lon": [1.0]})
    result = land_cover.fetch_land_cover_for_sites(sites, cache_path=tmp_path / "lc.parquet")
    assert result.empty or "a" not in set(result["site"])


# ---------------------------------------------------------------------------
# land_cover_features: join
# ---------------------------------------------------------------------------

def test_land_cover_features_joins_by_site_and_preserves_visit_index():
    visits = pd.DataFrame({"site": ["a", "a", "b"]}, index=[10, 11, 12])
    lookup = pd.DataFrame({"site": ["a", "b"], "landcover_cropland_pct": [0.6, 0.2],
                          "landcover_built_pct": [0.1, 0.7], "landcover_tree_pct": [0.3, 0.1]})
    out = land_cover.land_cover_features(visits, lookup)
    assert list(out.index) == [10, 11, 12]
    assert out.loc[10, "landcover_cropland_pct"] == 0.6
    assert out.loc[12, "landcover_built_pct"] == 0.7


def test_land_cover_features_is_nan_for_an_unknown_station():
    visits = pd.DataFrame({"site": ["unknown"]})
    lookup = pd.DataFrame({"site": ["a"], "landcover_cropland_pct": [0.6],
                          "landcover_built_pct": [0.1], "landcover_tree_pct": [0.3]})
    out = land_cover.land_cover_features(visits, lookup)
    assert out["landcover_cropland_pct"].isna().all()
