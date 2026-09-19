"""Land-use composition near station locations, from ESA WorldCover (10 m global land-cover map, via the
Planetary Computer STAC catalog -- the same catalog and no-login pattern src.data.satellite_extract
already uses for Sentinel-2, extended here to a different collection).

Cropland near a water body means fertiliser/pesticide runoff; built-up land means sewage; both are real,
non-satellite-reflectance drivers of BOD/turbidity that a static land-use percentage can proxy for, more
precisely than src.data.city_proximity's coarse "distance to one of ~85 major city centroids" (which
misses smaller towns and doesn't capture the land use immediately upstream of a station).

Source: `esa-worldcover` collection on the Planetary Computer STAC API (same STAC_URL as
src.data.satellite_extract), one static classification per year (2020/2021 releases), 10 m resolution,
11 classes (10 Tree cover, 20 Shrubland, 30 Grassland, 40 Cropland, 50 Built-up, 60 Bare/sparse
vegetation, 70 Snow/ice, 80 Permanent water bodies, 90 Herbaceous wetland, 95 Mangroves, 100 Moss/lichen).

**NOT executed in this environment**: verified directly that this sandbox has no network route to
planetarycomputer.microsoft.com (the same policy-blocked 403 that blocks the Sentinel-2 fetch here too).
Code-complete and unit-tested against a mocked STAC search + a synthetic classification array
(tests/test_land_cover.py); someone with real network access needs to run
scripts.backfill_soil_land_cover to populate these columns and then ablate them before adding to any
src.models.schema.FEATURE_SETS entry.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.data.satellite_extract import STAC_URL

log = logging.getLogger(__name__)

WORLDCOVER_COLLECTION = "esa-worldcover"
CLASS_LABELS = {10: "tree", 20: "shrub", 30: "grass", 40: "cropland", 50: "built", 60: "bare",
               70: "snow_ice", 80: "water", 90: "wetland", 95: "mangrove", 100: "moss_lichen"}
REPORTED_CLASSES = ("cropland", "built", "tree")   # the fractions actually surfaced as features
PIXEL_M = 10.0


def search_worldcover_item(lat: float, lon: float, catalog=None):
    """The ESA WorldCover STAC item (there is at most one per release year) covering (lat, lon). Pass a
    pre-opened ``catalog`` (pystac_client.Client) in tests to avoid a real network call."""
    from pystac_client import Client

    catalog = catalog or Client.open(STAC_URL)
    res = catalog.search(collections=[WORLDCOVER_COLLECTION], intersects={"type": "Point", "coordinates": [lon, lat]},
                         limit=10)
    items = list(res.items())
    if not items:
        return None
    # Prefer the newest release year when more than one covers the point.
    return sorted(items, key=lambda it: it.properties.get("start_datetime", ""), reverse=True)[0]


def read_classification_window(item, lat: float, lon: float, radius_m: float = 500.0) -> np.ndarray:
    """Read the WorldCover classification band in a square window around (lat, lon)."""
    import planetary_computer as pc
    import rasterio
    from rasterio.warp import transform as warp_transform
    from rasterio.windows import from_bounds

    href = pc.sign(item.assets["map"].href)
    with rasterio.open(href) as src:
        x, y = (v[0] for v in warp_transform("EPSG:4326", src.crs.to_string(), [lon], [lat]))
        n = max(1, int(round(2 * radius_m / PIXEL_M)))
        win = from_bounds(x - radius_m, y - radius_m, x + radius_m, y + radius_m, src.transform)
        return src.read(1, window=win, out_shape=(n, n), boundless=True, fill_value=0)


def landcover_fractions(classification: np.ndarray) -> dict:
    """Per-class pixel fraction (0-1) within a classification window, for the classes this project
    actually uses as a feature (REPORTED_CLASSES); 0-value (nodata/outside-coverage) pixels excluded
    from the denominator."""
    valid = classification[classification != 0]
    if valid.size == 0:
        return {f"landcover_{c}_pct": float("nan") for c in REPORTED_CLASSES}
    out = {}
    for code, label in CLASS_LABELS.items():
        if label in REPORTED_CLASSES:
            out[f"landcover_{label}_pct"] = float((valid == code).mean())
    return out


def fetch_land_cover_for_sites(sites: pd.DataFrame, radius_m: float = 500.0, catalog=None,
                               cache_path=None, offline: bool = False) -> pd.DataFrame:
    """Land-cover fractions for each unique station in ``sites`` (columns: site/lat/lon). Resumable and
    ``offline=True`` mirror src.data.weather.fetch_daily_rainfall's convention exactly."""
    cols = ["site", *[f"landcover_{c}_pct" for c in REPORTED_CLASSES]]
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
        item = search_worldcover_item(row.lat, row.lon, catalog=catalog)
        if item is None:
            log.warning("no WorldCover coverage for %s (%.4f, %.4f)", row.site, row.lat, row.lon)
            continue
        classification = read_classification_window(item, row.lat, row.lon, radius_m)
        rows.append({"site": row.site, **landcover_fractions(classification)})
        if cache_path is not None and i % 50 == 0:
            pd.concat([cached, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("site").to_parquet(cache_path)

    result = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("site")
    if cache_path is not None:
        result.to_parquet(cache_path)
    return result


def land_cover_features(visits: pd.DataFrame, land_cover: pd.DataFrame) -> pd.DataFrame:
    """Per-visit land-cover columns, joined from the per-station ``land_cover`` table. NaN where a
    station has no cached fetch yet -- never imputed."""
    merged = visits[["site"]].merge(land_cover.drop_duplicates("site"), on="site", how="left")
    out = merged[[f"landcover_{c}_pct" for c in REPORTED_CLASSES]]
    out.index = visits.index
    return out
