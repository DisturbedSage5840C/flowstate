"""tests/test_soil.py — unit tests for src/data/soil.py (ISRIC SoilGrids fetch). No real network calls;
requests.get is monkeypatched, mirroring tests/test_weather.py's pattern."""

from __future__ import annotations

import pandas as pd
import pytest
import requests

from src.data import soil


class FakeResponse:
    def __init__(self, status_code, body=None):
        self.status_code = status_code
        self._body = body if body is not None else {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


def _soilgrids_body(soc=120.0, clay=250.0, phh2o=68.0, bdod=140.0):
    def layer(name, value):
        return {"name": name, "depths": [{"label": "0-5cm", "values": {"mean": value}}]}

    return {"properties": {"layers": [layer("soc", soc), layer("clay", clay), layer("phh2o", phh2o),
                                     layer("bdod", bdod)]}}


# ---------------------------------------------------------------------------
# _parse_soilgrids_response
# ---------------------------------------------------------------------------

def test_parse_response_extracts_all_four_properties():
    out = soil._parse_soilgrids_response(_soilgrids_body())
    assert out == {"soc_mean": 120.0, "clay_mean": 250.0, "phh2o_mean": 68.0, "bdod_mean": 140.0}


def test_parse_response_missing_layer_is_nan():
    body = _soilgrids_body()
    body["properties"]["layers"] = [l for l in body["properties"]["layers"] if l["name"] != "clay"]
    out = soil._parse_soilgrids_response(body)
    assert out["clay_mean"] != out["clay_mean"]   # NaN
    assert out["soc_mean"] == 120.0


def test_parse_response_missing_depth_is_nan():
    body = _soilgrids_body()
    body["properties"]["layers"][0]["depths"] = []      # soc has no 0-5cm entry
    out = soil._parse_soilgrids_response(body)
    assert out["soc_mean"] != out["soc_mean"]


# ---------------------------------------------------------------------------
# fetch_soil_point: rate-limit retry
# ---------------------------------------------------------------------------

def test_fetch_soil_point_retries_on_429_then_succeeds(monkeypatch):
    calls = {"n": 0}
    sleeps = []

    def fake_get(url, params, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResponse(429, {"detail": "rate limit exceeded"})
        return FakeResponse(200, _soilgrids_body())

    monkeypatch.setattr(soil.requests, "get", fake_get)
    monkeypatch.setattr(soil.time, "sleep", lambda s: sleeps.append(s))
    out = soil.fetch_soil_point(12.9, 77.6)
    assert out["soc_mean"] == 120.0
    assert calls["n"] == 2 and len(sleeps) == 1


def test_fetch_soil_point_gives_up_after_max_retries(monkeypatch):
    def fake_get(url, params, timeout):
        return FakeResponse(429, {"detail": "still limited"})

    monkeypatch.setattr(soil.requests, "get", fake_get)
    monkeypatch.setattr(soil.time, "sleep", lambda s: None)
    with pytest.raises(soil.SoilGridsRateLimited):
        soil.fetch_soil_point(12.9, 77.6)


# ---------------------------------------------------------------------------
# fetch_soil_for_sites: resumable cache / offline mode
# ---------------------------------------------------------------------------

def test_fetch_soil_for_sites_skips_already_cached_stations(monkeypatch, tmp_path):
    cache_path = tmp_path / "soil.parquet"
    pd.DataFrame({"site": ["a"], "soc_mean": [100.0], "clay_mean": [200.0],
                 "phh2o_mean": [65.0], "bdod_mean": [130.0]}).to_parquet(cache_path)

    calls = []

    def fake_fetch(lat, lon, timeout=30.0):
        calls.append((lat, lon))
        return {"soc_mean": 1.0, "clay_mean": 2.0, "phh2o_mean": 3.0, "bdod_mean": 4.0}

    monkeypatch.setattr(soil, "fetch_soil_point", fake_fetch)
    monkeypatch.setattr(soil.time, "sleep", lambda s: None)
    sites = pd.DataFrame({"site": ["a", "b"], "lat": [1.0, 2.0], "lon": [1.0, 2.0]})
    result = soil.fetch_soil_for_sites(sites, cache_path=cache_path, request_delay_s=0)

    assert len(calls) == 1                        # only "b" fetched
    assert set(result["site"]) == {"a", "b"}
    assert result.set_index("site").loc["a", "soc_mean"] == 100.0   # cached value preserved


def test_fetch_soil_for_sites_offline_returns_cache_with_no_network_calls(monkeypatch, tmp_path):
    cache_path = tmp_path / "soil.parquet"
    pd.DataFrame({"site": ["a"], "soc_mean": [1.0], "clay_mean": [2.0],
                 "phh2o_mean": [3.0], "bdod_mean": [4.0]}).to_parquet(cache_path)

    def fail(*a, **k):
        raise AssertionError("should not be called in offline mode")

    monkeypatch.setattr(soil, "fetch_soil_point", fail)
    sites = pd.DataFrame({"site": ["a", "b"], "lat": [1.0, 2.0], "lon": [1.0, 2.0]})
    result = soil.fetch_soil_for_sites(sites, cache_path=cache_path, offline=True)
    assert list(result["site"]) == ["a"]           # "b" never fetched


# ---------------------------------------------------------------------------
# soil_features: unit conversion + join
# ---------------------------------------------------------------------------

def test_soil_features_converts_to_human_units_and_joins_by_site():
    visits = pd.DataFrame({"site": ["a", "a", "b"]})
    lookup = pd.DataFrame({"site": ["a", "b"], "soc_mean": [200.0, 100.0], "clay_mean": [300.0, 150.0],
                          "phh2o_mean": [70.0, 65.0], "bdod_mean": [140.0, 130.0]})
    out = soil.soil_features(visits, lookup)
    assert out.loc[0, "soil_organic_carbon_pct"] == pytest.approx(2.0)     # 200 dg/kg -> 2.0 %
    assert out.loc[0, "soil_clay_pct"] == pytest.approx(30.0)              # 300 g/kg -> 30 %
    assert out.loc[0, "soil_ph"] == pytest.approx(7.0)                     # 70 (pH*10) -> 7.0
    assert out.loc[0, "soil_bulk_density_gcm3"] == pytest.approx(1.4)      # 140 cg/cm3 -> 1.4 g/cm3
    assert out.loc[2, "soil_organic_carbon_pct"] == pytest.approx(1.0)


def test_soil_features_is_nan_for_a_station_with_no_cached_fetch():
    visits = pd.DataFrame({"site": ["unknown"]})
    lookup = pd.DataFrame({"site": ["a"], "soc_mean": [200.0], "clay_mean": [300.0],
                          "phh2o_mean": [70.0], "bdod_mean": [140.0]})
    out = soil.soil_features(visits, lookup)
    assert out["soil_organic_carbon_pct"].isna().all()
