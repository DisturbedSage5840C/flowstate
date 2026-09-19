"""tests/test_weather.py — unit tests for src/data/weather.py: Open-Meteo rate-limit handling and the
.asof() antecedent-rainfall lookup (no real network calls; requests.get is monkeypatched)."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest
import requests

from src.data import weather


class FakeResponse:
    def __init__(self, status_code, body=None):
        self.status_code = status_code
        self._body = body if body is not None else {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


# ---------------------------------------------------------------------------
# _seconds_until_next_utc_hour / _retry_reason
# ---------------------------------------------------------------------------

def test_seconds_until_next_utc_hour_is_exact_with_a_fixed_jitter():
    now = dt.datetime(2026, 1, 1, 10, 15, 30, tzinfo=dt.timezone.utc)
    secs = weather._seconds_until_next_utc_hour(now, jitter_s=5.0)
    assert secs == pytest.approx((44 * 60 + 30) + 5.0, abs=0.01)


def test_seconds_until_next_utc_hour_handles_the_boundary():
    now = dt.datetime(2026, 1, 1, 10, 0, 0, tzinfo=dt.timezone.utc)
    assert weather._seconds_until_next_utc_hour(now, jitter_s=0.0) == pytest.approx(3600.0)


def test_retry_reason_parses_the_open_meteo_body():
    resp = FakeResponse(429, {"error": True, "reason": "Hourly API request limit exceeded. Try again later."})
    assert "Hourly" in weather._retry_reason(resp)
    assert weather._retry_reason(FakeResponse(429, {})) == ""
    assert weather._retry_reason(FakeResponse(429, "not json body")) == ""


# ---------------------------------------------------------------------------
# fetch_precip_batch: hourly / daily / unrecognised 429 handling
# ---------------------------------------------------------------------------

def test_hourly_limit_sleeps_to_the_next_hour_then_succeeds(monkeypatch):
    calls = {"n": 0}
    sleeps = []

    def fake_get(url, params, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResponse(429, {"reason": "Hourly API request limit exceeded"})
        return FakeResponse(200, [{"daily": {"time": ["2020-01-01"], "precipitation_sum": [1.0]}}])

    monkeypatch.setattr(weather.requests, "get", fake_get)
    monkeypatch.setattr(weather.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(weather, "_seconds_until_next_utc_hour", lambda: 42.0)

    result = weather.fetch_precip_batch([10.0], [80.0], "2020-01-01", "2020-01-01")
    assert result[0]["daily"]["precipitation_sum"] == [1.0]
    assert sleeps == [42.0]            # slept to the hour boundary, not a short exponential backoff
    assert calls["n"] == 2


def test_daily_limit_raises_immediately_without_sleeping(monkeypatch):
    def fake_get(url, params, timeout):
        return FakeResponse(429, {"reason": "Daily API request limit exceeded"})

    monkeypatch.setattr(weather.requests, "get", fake_get)
    monkeypatch.setattr(weather.time, "sleep", lambda s: pytest.fail("must not sleep on a daily limit"))
    with pytest.raises(weather.DailyQuotaExceeded, match="Daily"):
        weather.fetch_precip_batch([10.0], [80.0], "2020-01-01", "2020-01-01")


def test_unrecognised_429_falls_back_to_exponential_backoff(monkeypatch):
    calls = {"n": 0}
    waits = []

    def fake_get(url, params, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            return FakeResponse(429, {})
        return FakeResponse(200, [{"daily": {"time": [], "precipitation_sum": []}}])

    monkeypatch.setattr(weather.requests, "get", fake_get)
    monkeypatch.setattr(weather.time, "sleep", lambda s: waits.append(s))
    weather.fetch_precip_batch([10.0], [80.0], "2020-01-01", "2020-01-01")
    assert waits == [2.0, 4.0]          # exponential backoff (2 * 2**attempt), not an hourly-length sleep
    assert calls["n"] == 3


def test_hourly_limit_gives_up_after_max_waits(monkeypatch):
    def fake_get(url, params, timeout):
        return FakeResponse(429, {"reason": "Hourly API request limit exceeded"})

    monkeypatch.setattr(weather.requests, "get", fake_get)
    monkeypatch.setattr(weather.time, "sleep", lambda s: None)
    monkeypatch.setattr(weather, "_seconds_until_next_utc_hour", lambda: 0.0)
    monkeypatch.setattr(weather, "MAX_HOURLY_WAITS", 2)
    with pytest.raises(weather.DailyQuotaExceeded):
        weather.fetch_precip_batch([10.0], [80.0], "2020-01-01", "2020-01-01")


# ---------------------------------------------------------------------------
# antecedent_rainfall_features: .asof() lookup across a gap in the cache
# ---------------------------------------------------------------------------

def test_asof_lookup_forward_fills_across_a_cache_gap_instead_of_returning_nan():
    # Cache has only two cumulative-precip anchor days (01-01, 01-10), nothing in between -- a real gap.
    # The old `cum.reindex([end]).ffill()` could never forward-fill (a single-row reindex has nothing
    # earlier to look back at), so it silently returned NaN whenever `end` itself wasn't a cached day.
    rainfall = pd.DataFrame({
        "site": ["s1", "s1"],
        "date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-10")],
        "precip_mm": [5.0, 3.0],
    })
    visits = pd.DataFrame({"site": ["s1"], "date": [pd.Timestamp("2020-01-06")]})   # falls inside the gap
    feats = weather.antecedent_rainfall_features(visits, rainfall)

    # rain_30d_mm: window start (2019-12-07) is before any cached day -> asof is NaN -> treated as 0.
    # window end (2020-01-05) forward-fills from the 01-01 anchor (cumsum 5.0) instead of returning NaN.
    assert feats["rain_30d_mm"].iloc[0] == pytest.approx(5.0)
    # rain_3d_mm: both the window start (01-03) and end (01-05) forward-fill from the same 01-01 anchor,
    # so the 3-day sum is genuinely 0 (not NaN) -- there is no evidence of rain in that sub-window.
    assert feats["rain_3d_mm"].iloc[0] == pytest.approx(0.0)
    assert not feats[["rain_3d_mm", "rain_30d_mm"]].isna().any().any()


def test_asof_lookup_returns_nan_when_visit_predates_all_cached_rainfall():
    rainfall = pd.DataFrame({"site": ["s1"], "date": [pd.Timestamp("2020-06-01")], "precip_mm": [4.0]})
    visits = pd.DataFrame({"site": ["s1"], "date": [pd.Timestamp("2020-01-06")]})   # long before any cached day
    feats = weather.antecedent_rainfall_features(visits, rainfall)
    assert feats["rain_3d_mm"].isna().all()          # genuinely unknown, not fabricated as 0


def test_asof_lookup_unknown_site_returns_nan():
    rainfall = pd.DataFrame({"site": ["s1"], "date": [pd.Timestamp("2020-01-01")], "precip_mm": [4.0]})
    visits = pd.DataFrame({"site": ["s2"], "date": [pd.Timestamp("2020-01-06")]})
    feats = weather.antecedent_rainfall_features(visits, rainfall)
    assert feats["rain_3d_mm"].isna().all()
