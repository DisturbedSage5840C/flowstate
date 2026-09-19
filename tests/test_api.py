"""API layer: runs against the real artifacts in the repo (skipped when the real table has not been built)."""

import pytest

from src.api.service import ROOT

pytestmark = pytest.mark.skipif(
    not any((ROOT / "data" / "processed" / f).exists() for f in ("train_real_large.parquet", "train_real.parquet")),
    reason="real training table not built",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from src.api.server import app
    return TestClient(app)


def test_overview_matches_screening_report(client):
    o = client.get("/api/overview").json()
    assert o["stations"] > 100 and o["high"] + o["moderate"] + o["low"] == o["stations"]
    assert 0.5 < o["model"]["roc_auc"] < 1
    assert o["years"] == sorted(o["years"])


def test_stations_and_year_filter(client):
    all_ = client.get("/api/stations").json()
    yr = client.get("/api/stations", params={"year": client.get("/api/overview").json()["years"][0]}).json()
    assert len(all_) > 0 and 0 < len(yr) <= len(all_)
    assert {s["risk"] for s in all_} <= {"low", "mod", "high"}
    assert all(-90 < s["lat"] < 90 for s in all_)


def test_station_detail_roundtrip_and_404(client):
    top = client.get("/api/priorities", params={"n": 1}).json()[0]
    d = client.get(f"/api/stations/{top['id']}").json()
    assert d["id"] == top["id"] and d["risk"] == "high" and d["signals"] and d["recommendations"]
    assert [r["name"] for r in d["readings"]] == ["Turbidity", "BOD", "Dissolved oxygen"]
    assert client.get("/api/stations/nope").status_code == 404


def test_priorities_sorted_by_probability(client):
    ids = [p["id"] for p in client.get("/api/priorities", params={"n": 5}).json()]
    probs = [client.get(f"/api/stations/{i}").json()["prob"] for i in ids]
    assert probs == sorted(probs, reverse=True)


def test_compare_search_cities_alerts_scene(client):
    p = client.get("/api/priorities", params={"n": 2}).json()
    c = client.get("/api/compare", params={"a": p[0]["id"], "b": p[1]["id"]}).json()
    assert c["a"]["name"] and c["distance_km"] >= 0
    assert client.get("/api/search", params={"q": "delhi"}).json()
    assert client.get("/api/cities").json()
    for a in client.get("/api/alerts").json():
        assert a["sev"] in ("mod", "high") and a["evidence"]
    assert client.get("/api/scene", params={"lat": 12.97, "lon": 77.59}).json() is not None
    assert client.get("/api/scene", params={"lat": 0, "lon": 0}).json() is None


def test_ask_is_honest_about_unsupported_data(client):
    r = client.post("/api/ask", json={"query": "hotspots within 2 km of sewage outlets"}).json()
    assert r["understood"] and any("sewage-outlet" in n for n in r["notes"])
    bad = client.post("/api/ask", json={"query": "what is the meaning of life"}).json()
    assert not bad["understood"] and bad["ids"] == []
