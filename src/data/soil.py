"""Soil composition at station locations, from ISRIC SoilGrids v2.0 (REST API, no key required).

Sewage/nutrient runoff into a water body depends partly on what the surrounding soil is: clay-heavy
soil sheds more surface runoff (higher turbidity after rain) than sandy/well-drained soil, and organic
carbon / pH relate to nutrient retention that can feed BOD. Static (soil composition doesn't change per
visit the way rainfall does), so one point query per STATION, not per visit.

Source: https://rest.isric.org/soilgrids/v2.0/docs, 250 m resolution global machine-learning predictions
over WoSIS soil profile observations, no key/login required. Properties fetched at 0-5cm depth (the layer
most relevant to surface runoff, not deep soil chemistry): soc (organic carbon), clay content, pH in
water, bulk density.

**NOT executed in this environment**: verified directly (curl through this sandbox's egress proxy) that
there is no network route to rest.isric.org here -- the same policy-blocked 403 as Open-Meteo and
Planetary Computer, and Earth Engine has no credentials configured either. This module is code-complete
and unit-tested against mocked HTTP responses (tests/test_soil.py), following the same retry/rate-limit
pattern src.data.weather learned the hard way (parse what the response actually says, never assume the
limit type). Someone with real network access needs to run scripts.backfill_soil_land_cover to populate
the columns and then ablate them before adding to any src.models.schema.FEATURE_SETS entry -- consistent
with this project's "tested before trusted" rule (see how rainfall was fetched, validated, and still
excluded from every feature set because it didn't help).
"""

from __future__ import annotations

import logging
import random
import time

import pandas as pd
import requests

log = logging.getLogger(__name__)

SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
# SoilGrids property codes -> what they are (raw units noted; soil_features() converts to human units).
PROPERTIES = ("soc", "clay", "phh2o", "bdod")
DEPTH_LABEL = "0-5cm"
STAT = "mean"
MAX_RETRIES = 5


class SoilGridsRateLimited(RuntimeError):
    """SoilGrids' rate limit was hit and retries were exhausted -- stop and retry later, don't spin."""


def _retry_reason(resp: requests.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return ""
    if isinstance(body, dict):
        return str(body.get("detail") or body.get("message") or body.get("error") or "")
    return ""


def fetch_soil_point(lat: float, lon: float, timeout: float = 30.0) -> dict:
    """One SoilGrids point query -> {soc_mean, clay_mean, phh2o_mean, bdod_mean} (raw SoilGrids units;
    see soil_features() for the conversion to human units).

    Retries with exponential backoff on HTTP 429, logging whatever the response body actually says (the
    lesson learned in src.data.weather: never assume which limit was hit without checking).
    """
    params = {"lon": lon, "lat": lat, "property": list(PROPERTIES), "depth": DEPTH_LABEL, "value": STAT}
    attempt = 0
    while True:
        resp = requests.get(SOILGRIDS_URL, params=params, timeout=timeout)
        if resp.status_code == 429:
            if attempt >= MAX_RETRIES:
                raise SoilGridsRateLimited(f"rate limited after {MAX_RETRIES} attempts: {_retry_reason(resp)}")
            wait = 2.0 * (2 ** attempt) + random.uniform(0, 1.0)
            log.warning("SoilGrids rate limited (%s), retrying in %.0fs (attempt %d/%d)",
                       _retry_reason(resp) or "no reason given", wait, attempt + 1, MAX_RETRIES)
            time.sleep(wait)
            attempt += 1
            continue
        resp.raise_for_status()
        return _parse_soilgrids_response(resp.json())


def _parse_soilgrids_response(data: dict) -> dict:
    layers = {layer["name"]: layer for layer in data.get("properties", {}).get("layers", [])}
    out = {}
    for prop in PROPERTIES:
        layer = layers.get(prop)
        value = None
        if layer:
            depth_entry = next((d for d in layer.get("depths", []) if d.get("label") == DEPTH_LABEL), None)
            if depth_entry:
                value = depth_entry.get("values", {}).get(STAT)
        out[f"{prop}_mean"] = float(value) if value is not None else float("nan")
    return out


def fetch_soil_for_sites(sites: pd.DataFrame, cache_path=None, offline: bool = False,
                         request_delay_s: float = 1.0) -> pd.DataFrame:
    """Soil properties for each unique station in ``sites`` (columns: site/lat/lon).

    Resumable: a station already present in ``cache_path`` is skipped. ``offline=True`` returns whatever
    is already cached with zero network calls (for proceeding without the fetch having run at all, same
    convention as src.data.weather.fetch_daily_rainfall).
    """
    cols = ["site", *[f"{p}_mean" for p in PROPERTIES]]
    cached = pd.DataFrame(columns=cols)
    if cache_path is not None and cache_path.exists():
        cached = pd.read_parquet(cache_path)
    if offline:
        return cached

    todo = sites.drop_duplicates(subset="site")
    todo = todo[~todo["site"].isin(cached["site"])]
    if todo.empty:
        return cached

    rows = []
    for i, row in enumerate(todo.itertuples(), start=1):
        try:
            values = fetch_soil_point(row.lat, row.lon)
        except requests.RequestException as e:
            log.warning("soil fetch failed for %s: %s", row.site, e)
            continue
        rows.append({"site": row.site, **values})
        time.sleep(request_delay_s)
        if cache_path is not None and i % 50 == 0:
            pd.concat([cached, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("site").to_parquet(cache_path)

    result = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("site")
    if cache_path is not None:
        result.to_parquet(cache_path)
    return result


def soil_features(visits: pd.DataFrame, soil: pd.DataFrame) -> pd.DataFrame:
    """Per-visit soil columns in human units, joined from the per-station ``soil`` table (from
    fetch_soil_for_sites): soil_organic_carbon_pct (soc dg/kg -> %), soil_clay_pct (g/kg -> %),
    soil_ph (pH*10 -> pH), soil_bulk_density_gcm3 (cg/cm3 -> g/cm3). NaN where a station has no
    cached soil fetch yet -- never imputed.
    """
    merged = visits[["site"]].merge(soil.drop_duplicates("site"), on="site", how="left")
    return pd.DataFrame({
        "soil_organic_carbon_pct": merged.get("soc_mean", pd.Series(dtype=float)) / 100.0,
        "soil_clay_pct": merged.get("clay_mean", pd.Series(dtype=float)) / 10.0,
        "soil_ph": merged.get("phh2o_mean", pd.Series(dtype=float)) / 10.0,
        "soil_bulk_density_gcm3": merged.get("bdod_mean", pd.Series(dtype=float)) / 100.0,
    }, index=visits.index)
