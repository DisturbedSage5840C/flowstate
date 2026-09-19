"""Daily rainfall at station locations, from Open-Meteo's historical reanalysis (ERA5-based).

DO and BOD are not optically active (satellite reflectance can't see them directly), but they are
driven in part by runoff: rain washes sewage, fertiliser and sediment into rivers and lakes, which
depresses DO and raises BOD/turbidity a few days later. Antecedent rainfall is therefore a real,
non-satellite feature that can plausibly help predict them, unlike more reflectance-derived indices.

Source: Open-Meteo Historical Weather API (https://open-meteo.com), no key required, ERA5/ERA5-Land
reanalysis at ~9-25 km resolution -- coarser than the station location, and a model reanalysis rather
than a rain gauge, so treat this as regional rainfall context, not a precise local measurement.

Open-Meteo's free/open-access tier is capped at 10,000 "calls"/day, and a multi-location, multi-year
request costs roughly (locations x days-in-range / 14) calls (see their pricing FAQ) -- so fetching
every site's *entire* multi-year span, as an earlier version of this module did, burns through that
budget in a handful of batches. Only ~window_days of rainfall before each visit is ever consumed by
antecedent_rainfall_features, so compute_rainfall_windows narrows each fetch to just that, cutting
total requested days by roughly (span_of_data / window_days) -- an ~8x reduction for this project's
~4-visits-per-site, ~3-year span.
"""

from __future__ import annotations

import datetime as dt
import logging
import random
import time

import pandas as pd
import requests

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
BATCH_SIZE = 50
REQUEST_DELAY_S = 2.0
MAX_RETRIES = 5
MAX_HOURLY_WAITS = 24     # ~a day's worth of hourly windows before giving up, not an infinite loop
DEFAULT_WINDOW_DAYS = 35   # >= the longest antecedent window (30d) plus slack


class DailyQuotaExceeded(RuntimeError):
    """Open-Meteo's free-tier *daily* call quota is exhausted; no reset-time is confirmed, so the
    caller should stop and retry later rather than spin (see fetch_daily_rainfall's offline= param)."""


def _retry_reason(resp: requests.Response) -> str:
    """Open-Meteo's 429 body is JSON like {"error": true, "reason": "Hourly API request limit exceeded..."}."""
    try:
        body = resp.json()
    except ValueError:
        return ""
    return str(body.get("reason", "")) if isinstance(body, dict) else ""


def _seconds_until_next_utc_hour(now: "dt.datetime | None" = None, jitter_s: float | None = None) -> float:
    """Seconds from ``now`` (default: actual UTC now) to the next UTC hour boundary, plus a small random
    jitter so multiple callers hitting the hourly limit together don't all retry in the same instant."""
    now = now or dt.datetime.now(dt.timezone.utc)
    next_hour = now.replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=1)
    jitter_s = random.uniform(1.0, 15.0) if jitter_s is None else jitter_s
    return (next_hour - now).total_seconds() + jitter_s


def compute_rainfall_windows(visits: pd.DataFrame, window_days: int = DEFAULT_WINDOW_DAYS) -> pd.DataFrame:
    """Per site, the minimal set of [start, end] date ranges covering every visit's antecedent window.

    ``visits`` needs columns site/date. For each visit, the raw window is [date - window_days, date - 1]
    (see antecedent_rainfall_features for why it stops the day before, not on, the visit date); raw
    windows for the same site are merged where they overlap or touch, so a site sampled repeatedly in
    a short span gets one fetch instead of one per visit, without ever requesting a day nothing needs.
    """
    rows = []
    for site, g in visits.groupby("site"):
        intervals = sorted(
            (d - pd.Timedelta(days=window_days), d - pd.Timedelta(days=1))
            for d in pd.to_datetime(g["date"]).unique()
        )
        merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for start, end in intervals:
            if merged and start <= merged[-1][1] + pd.Timedelta(days=1):
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        for start, end in merged:
            rows.append({"site": site, "start_date": start, "end_date": end})
    return pd.DataFrame(rows, columns=["site", "start_date", "end_date"])


def fetch_precip_batch(lats: list[float], lons: list[float], start_date: str, end_date: str,
                       timeout: float = 30.0) -> list[dict]:
    """One API call for up to BATCH_SIZE locations sharing a date range; returns a list of
    {time: [...], precipitation_sum: [...]}.

    On HTTP 429 the response body names which limit was hit (confirmed live: the message is
    "Hourly API request limit exceeded", not the daily cap an earlier version of this function
    assumed -- a fixed exponential backoff topping out around a minute can never clear an hourly
    window, so it looped uselessly). Handling per limit:
      * "Hourly" -- sleep to the next UTC-hour boundary (+ jitter), then retry; this limit clears
        on its own, so it does not count against MAX_RETRIES (bounded instead by MAX_HOURLY_WAITS).
      * "Daily" -- raise DailyQuotaExceeded immediately: retrying cannot help until the quota resets.
      * anything else (unrecognised body, transient 429 with no reason) -- exponential backoff,
        MAX_RETRIES attempts, same as before.
    """
    params = {
        "latitude": ",".join(f"{v:.5f}" for v in lats),
        "longitude": ",".join(f"{v:.5f}" for v in lons),
        "start_date": start_date,
        "end_date": end_date,
        "daily": "precipitation_sum",
        "timezone": "Asia/Kolkata",
    }
    backoff_attempt = 0
    hourly_waits = 0
    while True:
        resp = requests.get(ARCHIVE_URL, params=params, timeout=timeout)
        if resp.status_code == 429:
            reason = _retry_reason(resp)
            low = reason.lower()
            if "daily" in low:
                raise DailyQuotaExceeded(reason or "Open-Meteo daily call quota exceeded")
            if "hourly" in low:
                hourly_waits += 1
                if hourly_waits > MAX_HOURLY_WAITS:
                    raise DailyQuotaExceeded(
                        f"still hourly-rate-limited after {MAX_HOURLY_WAITS} UTC-hour windows: {reason}")
                wait = _seconds_until_next_utc_hour()
                log.warning("hourly rate limit (%s); sleeping %.0fs to the next UTC hour", reason, wait)
                time.sleep(wait)
                continue
            if backoff_attempt >= MAX_RETRIES:
                resp.raise_for_status()
            wait = 2.0 * (2 ** backoff_attempt)
            backoff_attempt += 1
            log.warning("rate limited (%s), retrying in %.0fs (attempt %d/%d)",
                       reason or "no reason given", wait, backoff_attempt, MAX_RETRIES)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else [data]


def fetch_daily_rainfall(sites: pd.DataFrame, windows: pd.DataFrame, cache_path=None,
                          offline: bool = False) -> pd.DataFrame:
    """Daily precipitation (mm) for each (site, start_date, end_date) window in ``windows``.

    ``sites`` needs columns site/lat/lon (one row per unique site). ``windows`` comes from
    compute_rainfall_windows -- fetching only what each site's visits actually need, not its whole
    multi-year span. Returns long-format (site, date, precip_mm). Resumable: a window whose every
    day is already present in ``cache_path`` for that site is skipped; requests for windows that
    share an identical (start_date, end_date) are batched together.

    ``offline=True`` returns whatever is already in ``cache_path`` without attempting any network
    call -- for proceeding with partial rainfall coverage while the free tier's daily quota is
    exhausted, instead of idling until it resets. Rerun with offline=False later to fill the rest.
    """
    site_coords = sites.drop_duplicates(subset="site").set_index("site")[["lat", "lon"]]

    cached = pd.DataFrame(columns=["site", "date", "precip_mm"])
    if cache_path is not None and cache_path.exists():
        cached = pd.read_parquet(cache_path)
    if offline:
        return cached
    cached_days = {site: set(g["date"]) for site, g in cached.groupby("site")} if len(cached) else {}

    todo_rows = []
    for _, w in windows.iterrows():
        needed = pd.date_range(w["start_date"], w["end_date"], freq="D")
        have = cached_days.get(w["site"], set())
        if not all(d in have for d in needed):
            todo_rows.append(w)
    if not todo_rows:
        return cached
    todo = pd.DataFrame(todo_rows)

    rows = []
    groups = list(todo.groupby(["start_date", "end_date"]))
    for gi, ((start, end), g) in enumerate(groups):
        site_names = [s for s in g["site"].tolist() if s in site_coords.index]
        for b in range(0, len(site_names), BATCH_SIZE):
            batch_sites = site_names[b:b + BATCH_SIZE]
            lats = site_coords.loc[batch_sites, "lat"].tolist()
            lons = site_coords.loc[batch_sites, "lon"].tolist()
            try:
                results = fetch_precip_batch(lats, lons, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
            except requests.RequestException as e:
                log.warning("rainfall window %s..%s failed: %s", start.date(), end.date(), e)
                continue
            for site_name, result in zip(batch_sites, results):
                daily = result.get("daily", {})
                for d, p in zip(daily.get("time", []), daily.get("precipitation_sum", [])):
                    rows.append({"site": site_name, "date": pd.Timestamp(d), "precip_mm": p})
            time.sleep(REQUEST_DELAY_S)
        log.info("rainfall: window %d/%d done (%s..%s, %d sites)",
                 gi + 1, len(groups), start.date(), end.date(), len(site_names))
        if cache_path is not None:
            partial = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(["site", "date"])
            partial.to_parquet(cache_path)

    fetched = pd.DataFrame(rows)
    return pd.concat([cached, fetched], ignore_index=True).drop_duplicates(["site", "date"])


def antecedent_rainfall_features(visits: pd.DataFrame, rainfall: pd.DataFrame) -> pd.DataFrame:
    """For each (site, date) visit, sum rainfall over the 3/7/14/30 days strictly before the visit.

    Antecedent sums (not same-day) because runoff reaching a station lags the rain event by hours
    to a few days; same-day precipitation_sum would partly describe weather *during* sampling, not
    the pollution-loading history the target reflects.
    """
    rainfall = rainfall.sort_values(["site", "date"]).copy()
    windows = (3, 7, 14, 30)
    cum_by_site = {}
    for site, g in rainfall.groupby("site"):
        g = g.set_index("date")["precip_mm"]
        cum_by_site[site] = g.cumsum()

    def lookup(site, date, window):
        cum = cum_by_site.get(site)
        if cum is None or len(cum) == 0:
            return float("nan")
        end = date - pd.Timedelta(days=1)
        start = date - pd.Timedelta(days=window)
        # .asof() forward-fills from the last cached day at or before the target date; the previous
        # `cum.reindex([end]).ffill()` looked for `end` in a single-element frame, so it could never
        # actually forward-fill and silently returned NaN whenever `end` itself was missing from the
        # cache (verified directly against data/interim/rainfall_daily.parquet).
        end_val = cum.asof(end)
        start_val = cum.asof(start)
        if pd.isna(start_val):
            start_val = 0.0          # nothing cached before the window start: treat prior rainfall as 0
        if pd.isna(end_val):
            return float("nan")      # no cached day at or before the visit: genuinely unknown, not 0
        return end_val - start_val

    feats: dict[str, list[float]] = {f"rain_{w}d_mm": [] for w in windows}
    for site, date in zip(visits["site"], visits["date"]):
        for w in windows:
            feats[f"rain_{w}d_mm"].append(lookup(site, date, w))
    return pd.DataFrame(feats, index=visits.index)
