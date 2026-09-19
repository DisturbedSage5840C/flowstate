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

import logging
import time

import pandas as pd
import requests

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
BATCH_SIZE = 50
REQUEST_DELAY_S = 2.0
MAX_RETRIES = 5
DEFAULT_WINDOW_DAYS = 35   # >= the longest antecedent window (30d) plus slack


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

    Retries with exponential backoff on HTTP 429. Note this only smooths over short bursts -- if
    the day's call quota is actually exhausted, every retry (and every later call) will also 429
    until the free tier's quota resets; compute_rainfall_windows is what keeps total volume low
    enough that this project's fetch can plausibly finish inside a day's quota at all.
    """
    params = {
        "latitude": ",".join(f"{v:.5f}" for v in lats),
        "longitude": ",".join(f"{v:.5f}" for v in lons),
        "start_date": start_date,
        "end_date": end_date,
        "daily": "precipitation_sum",
        "timezone": "Asia/Kolkata",
    }
    for attempt in range(MAX_RETRIES):
        resp = requests.get(ARCHIVE_URL, params=params, timeout=timeout)
        if resp.status_code == 429:
            wait = 2.0 * (2 ** attempt)
            log.warning("rate limited, retrying in %.0fs (attempt %d/%d)", wait, attempt + 1, MAX_RETRIES)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else [data]
    resp.raise_for_status()
    return []


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
        if cum is None:
            return float("nan")
        end = date - pd.Timedelta(days=1)
        start = date - pd.Timedelta(days=window)
        end_val = cum.reindex([end]).ffill().iloc[0] if len(cum) else float("nan")
        before_start = cum[cum.index <= start]
        start_val = before_start.iloc[-1] if len(before_start) else 0.0
        return end_val - start_val

    feats: dict[str, list[float]] = {f"rain_{w}d_mm": [] for w in windows}
    for site, date in zip(visits["site"], visits["date"]):
        for w in windows:
            feats[f"rain_{w}d_mm"].append(lookup(site, date, w))
    return pd.DataFrame(feats, index=visits.index)
