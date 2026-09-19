"""Sentinel-2 L2A reflectance at in-situ station locations, from the Planetary Computer STAC API.

No credentials are needed. For every (station, sampling date) it finds the closest clear
Sentinel-2 scene within +/- ``day_tolerance`` days, reads a small window around the station,
masks clouds/shadows and non-water pixels, and returns the median reflectance of the water
pixels inside a circular buffer.

Water pixels: MNDWI > threshold  AND  NIR < nir_max  AND  not cloud/shadow/cirrus/snow in the
Scene Classification Layer (SCL).  (SCL == "water" is deliberately NOT required: Sen2Cor often
labels turbid or eutrophic Indian water bodies as vegetation / bare soil.)
"""

from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
# asset key -> canonical band name used everywhere else in the repo
BAND_ASSETS = {"B02": "B2", "B03": "B3", "B04": "B4", "B05": "B5", "B06": "B6", "B08": "B8", "B11": "B11"}
SCL_BAD = (0, 1, 3, 8, 9, 10, 11)        # no data, defective, shadow, cloud (med/high), cirrus, snow
SCL_CLOUDY = (3, 8, 9, 10)
PIXEL_M = 10.0


@dataclass
class ExtractionConfig:
    radius_m: float = 500.0            # matches the plan's 500 m station radius
    day_tolerance: int = 3             # plan: +/-3 days for Sentinel-2
    min_water_px: int = 8
    max_cloud_frac: float = 0.25       # of the buffer
    max_scene_cloud: float = 70.0      # scene-level filter used only to shrink the STAC search
    mndwi_threshold: float = 0.0
    nir_max: float = 0.20
    candidates: int = 3                # try up to N nearest-in-time scenes


# ---------------------------------------------------------------------------
# Pure array logic (unit-tested without any network access)
# ---------------------------------------------------------------------------

def circle_mask(n: int, radius_px: float) -> np.ndarray:
    yy, xx = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2.0
    return (xx - c) ** 2 + (yy - c) ** 2 <= radius_px ** 2


def water_mask(bands: dict[str, np.ndarray], scl: np.ndarray, cfg: ExtractionConfig) -> np.ndarray:
    """Water pixels: MNDWI > threshold AND NIR < nir_max AND SCL not cloud/shadow/cirrus/snow/no-data.

    (SCL "water" is deliberately not required: Sen2Cor often labels turbid or eutrophic Indian water as
    vegetation or bare soil.)
    """
    b3, b8, b11 = bands["B3"], bands["B8"], bands["B11"]
    with np.errstate(invalid="ignore", divide="ignore"):
        mndwi = (b3 - b11) / (b3 + b11)
    valid = ~np.isin(scl, SCL_BAD) & np.isfinite(mndwi) & (b3 > 0)
    return valid & (mndwi > cfg.mndwi_threshold) & (b8 < cfg.nir_max)


def summarize_window(bands: dict[str, np.ndarray], scl: np.ndarray, cfg: ExtractionConfig) -> dict:
    """Median reflectance over water pixels in a square window (circular buffer applied inside).

    ``bands`` maps canonical names (B2..B11) to reflectance arrays (0-1); ``scl`` is the SCL array.
    Returns ``status`` 'ok' or the reason the window is unusable, plus diagnostics.
    """
    n = scl.shape[0]
    circle = circle_mask(n, n / 2.0)
    n_circle = int(circle.sum())
    cloudy = np.isin(scl, SCL_CLOUDY) & circle
    cloud_frac = float(cloudy.sum() / n_circle)
    out = {"cloud_frac": cloud_frac, "n_circle_px": n_circle}
    if cloud_frac > cfg.max_cloud_frac:
        return {**out, "status": "cloudy", "n_water_px": 0}

    water = water_mask(bands, scl, cfg) & circle
    n_water = int(water.sum())
    out.update(n_water_px=n_water, water_frac=float(n_water / n_circle))
    if n_water < cfg.min_water_px:
        return {**out, "status": "no_water"}
    for name, arr in bands.items():
        out[name] = float(np.median(arr[water]))
    out["B4_std"] = float(np.std(bands["B4"][water]))
    return {**out, "status": "ok"}


def rank_items(items: list, target: dt.date, tolerance_days: int, limit: int) -> list:
    """Items within tolerance, nearest date first (ties: lower scene cloud cover first)."""
    scored = []
    for it in items:
        d = pd.Timestamp(it.datetime).tz_localize(None).date() if it.datetime.tzinfo else it.datetime.date()
        diff = abs((d - target).days)
        if diff <= tolerance_days:
            scored.append((diff, it.properties.get("eo:cloud_cover", 100.0), d, it))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [(diff, d, it) for diff, _, d, it in scored[:limit]]


def item_crs(item) -> str | None:
    """CRS string from STAC projection metadata (extension v2 'proj:code' or older 'proj:epsg')."""
    props = item.properties
    if props.get("proj:code"):
        return str(props["proj:code"])
    if props.get("proj:epsg"):
        return f"EPSG:{props['proj:epsg']}"
    return None


def reflectance_from_dn(dn: np.ndarray, processing_baseline: str | None) -> np.ndarray:
    """DN -> BOA reflectance. Baseline >= 04.00 (scenes from 2022-01-25) carries a +1000 DN offset."""
    offset = 0.0
    try:
        if processing_baseline and int(str(processing_baseline).split(".")[0]) >= 4:
            offset = -0.1
    except ValueError:
        pass
    out = dn.astype("float64") * 1e-4 + offset
    out[dn == 0] = np.nan
    return out


# ---------------------------------------------------------------------------
# STAC access
# ---------------------------------------------------------------------------

class StacExtractor:
    """Extract per-visit water reflectance. Thread-safe for reads; searches are cached per station."""

    collection = COLLECTION        # subclasses (e.g. the Landsat temperature extractor) override these
    extra_query: dict = {}

    def __init__(self, cfg: ExtractionConfig | None = None, workers: int = 6):
        from pystac_client import Client
        self.cfg = cfg or ExtractionConfig()
        self.workers = workers
        self.catalog = Client.open(STAC_URL)
        self._item_cache: dict[tuple, list] = {}

    # ---- search ---------------------------------------------------------
    def search(self, lat: float, lon: float, start: dt.date, end: dt.date) -> list:
        key = (round(lat, 4), round(lon, 4), start, end)
        if key not in self._item_cache:
            pad = dt.timedelta(days=self.cfg.day_tolerance)
            for attempt in range(4):
                try:
                    res = self.catalog.search(
                        collections=[self.collection],
                        intersects={"type": "Point", "coordinates": [lon, lat]},
                        datetime=f"{(start - pad).isoformat()}/{(end + pad).isoformat()}",
                        query={"eo:cloud_cover": {"lt": self.cfg.max_scene_cloud}, **self.extra_query},
                        limit=200,
                    )
                    self._item_cache[key] = list(res.items())
                    break
                except Exception as e:            # transient API errors
                    if attempt == 3:
                        raise
                    log.warning("STAC search retry (%s)", e)
                    time.sleep(2 * (attempt + 1))
        return self._item_cache[key]

    # ---- windowed reads -------------------------------------------------
    def _read_window(self, item, lat: float, lon: float, radius_m: float | None = None) -> dict:
        import planetary_computer as pc
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import transform as warp_transform
        from rasterio.transform import from_bounds as transform_from_bounds
        from rasterio.windows import from_bounds

        crs = item_crs(item)
        if crs is None:                          # fall back to the raster's own CRS
            with rasterio.open(pc.sign(item.assets["B04"].href)) as src:
                crs = src.crs.to_string()
        x, y = (v[0] for v in warp_transform("EPSG:4326", crs, [lon], [lat]))
        r = float(radius_m if radius_m is not None else self.cfg.radius_m)
        n = int(round(2 * r / PIXEL_M))
        baseline = item.properties.get("s2:processing_baseline")

        def read(asset_key: str, resampling) -> np.ndarray:
            href = pc.sign(item.assets[asset_key].href)
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="3",
                              GDAL_HTTP_RETRY_DELAY="1", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
                with rasterio.open(href) as src:
                    win = from_bounds(x - r, y - r, x + r, y + r, src.transform)
                    return src.read(1, window=win, out_shape=(n, n), boundless=True, fill_value=0,
                                    resampling=resampling)

        bands = {name: reflectance_from_dn(read(key, Resampling.bilinear), baseline)
                 for key, name in BAND_ASSETS.items()}
        scl = read("SCL", Resampling.nearest)
        return {"bands": bands, "scl": scl, "crs": crs,
                "transform": transform_from_bounds(x - r, y - r, x + r, y + r, n, n)}

    def extract_visit(self, station: str, lat: float, lon: float, date: dt.date, items: list) -> dict:
        base = {"station": station, "date": pd.Timestamp(date), "status": "no_scene"}
        for diff, scene_date, item in rank_items(items, date, self.cfg.day_tolerance, self.cfg.candidates):
            try:
                win = self._read_window(item, lat, lon)
            except Exception as e:               # unreadable tile/asset: try the next scene
                log.warning("read failed %s: %s", item.id, e)
                base = {**base, "status": "read_error"}
                continue
            summary = summarize_window(win["bands"], win["scl"], self.cfg)
            result = {**base, **summary, "scene_id": item.id, "scene_date": pd.Timestamp(scene_date),
                      "day_diff": int(diff), "scene_cloud": float(item.properties.get("eo:cloud_cover", np.nan)),
                      "sensor": item.properties.get("platform", "sentinel-2")}
            if summary["status"] == "ok":
                return result
            base = result                        # keep the most informative failure
        return base

    # ---- driver ---------------------------------------------------------
    def _extract_station(self, station: str, lat: float, lon: float, dates: list[dt.date]) -> list[dict]:
        try:
            items = self.search(lat, lon, dates[0], dates[-1])
        except Exception as e:
            log.error("search failed for %s: %s", station, e)
            return [{"station": station, "date": pd.Timestamp(d), "status": "search_error"} for d in dates]
        return [self.extract_visit(station, lat, lon, d, items) for d in dates]

    def extract(self, visits: pd.DataFrame, cache_path: Path | str | None = None,
                checkpoint_every: int = 20) -> pd.DataFrame:
        """``visits``: columns station, lat, lon, date. Resumable when ``cache_path`` is given.

        Work is parallel across stations (each does one catalog search, then reads its visits).
        """
        visits = visits.drop_duplicates(["station", "date"]).reset_index(drop=True)
        cache_path = Path(cache_path) if cache_path else None
        done: list[dict] = []
        done_keys: set = set()
        if cache_path and cache_path.exists():
            prev = pd.read_parquet(cache_path)
            done = prev.to_dict("records")
            done_keys = set(zip(prev["station"], pd.to_datetime(prev["date"])))
        keep = [(s, pd.Timestamp(d)) not in done_keys for s, d in zip(visits["station"], visits["date"])]
        todo = visits[keep]
        log.info("extracting %d visits at %d stations (%d visits already cached)",
                 len(todo), todo["station"].nunique(), len(done_keys))

        t0 = time.time()
        stations_done = 0
        groups = list(todo.groupby("station"))
        with cf.ThreadPoolExecutor(self.workers) as ex:
            futures = {}
            for station, g in groups:
                dates = sorted(pd.to_datetime(g["date"]).dt.date)
                futures[ex.submit(self._extract_station, station, float(g["lat"].iloc[0]),
                                  float(g["lon"].iloc[0]), dates)] = station
            for fut in cf.as_completed(futures):
                done.extend(fut.result())
                stations_done += 1
                if stations_done % checkpoint_every == 0:
                    ok = sum(1 for r in done if r.get("status") == "ok")
                    log.info("  %d/%d stations | %d visits, %d ok | %.0fs", stations_done, len(groups),
                             len(done), ok, time.time() - t0)
                    if cache_path:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        pd.DataFrame(done).to_parquet(cache_path, index=False)
        result = pd.DataFrame(done)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            result.to_parquet(cache_path, index=False)
        return result
