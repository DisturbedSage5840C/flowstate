"""Water surface temperature at station-visits from Landsat 8/9 Collection-2 Level-2 (ST_B10 / 'lwir11').

CPCB's temperature columns are empty, so this is the only temperature source: a real thermal-infrared
retrieval, available only when a Landsat overpass falls within +/-5 days of the visit (plan tolerance for
Landsat) and the water pixels near the station are clear. Everywhere else ``temp_surface`` stays NaN — it is
never filled with a constant.
"""

from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

from src.data.satellite_extract import ExtractionConfig, StacExtractor, item_crs, rank_items

log = logging.getLogger(__name__)

LANDSAT_COLLECTION = "landsat-c2-l2"
ST_SCALE, ST_OFFSET = 0.00341802, 149.0        # DN -> Kelvin (Collection 2 Level 2 surface temperature)
PIXEL_M = 30.0
QA_BAD_MASK = 0b0111110                        # bits 1-5: dilated cloud, cirrus, cloud, cloud shadow, snow
QA_WATER_BIT = 7


def kelvin_from_dn(dn: np.ndarray) -> np.ndarray:
    out = dn.astype("float64") * ST_SCALE + ST_OFFSET
    out[dn == 0] = np.nan
    return out


def summarize_temperature(st_k: np.ndarray, qa: np.ndarray, min_pixels: int = 4,
                          radius_px: float | None = None) -> dict:
    """Median water temperature (deg C) over clear water pixels of a square window (circular buffer inside)."""
    from src.data.satellite_extract import circle_mask

    n = qa.shape[0]
    circle = circle_mask(n, radius_px if radius_px is not None else n / 2.0)
    clear = (qa & QA_BAD_MASK) == 0
    water = ((qa >> QA_WATER_BIT) & 1) == 1
    plausible = np.isfinite(st_k) & (st_k > 273.15) & (st_k < 313.15)     # liquid water: 0-40 C
    use = circle & clear & water & plausible
    n_px = int(use.sum())
    out = {"temp_n_px": n_px}
    if n_px < min_pixels:
        return {**out, "temp_status": "no_clear_water"}
    return {**out, "temp_c": float(np.median(st_k[use]) - 273.15), "temp_status": "ok"}


class LandsatTempExtractor(StacExtractor):
    """Same parallel/resumable driver as the Sentinel-2 extractor, but reading Landsat thermal windows."""

    collection = LANDSAT_COLLECTION
    extra_query = {"platform": {"in": ["landsat-8", "landsat-9"]}}      # Landsat 7 has no 'lwir11' asset

    def __init__(self, cfg: ExtractionConfig | None = None, workers: int = 6):
        cfg = cfg or ExtractionConfig(day_tolerance=5, candidates=3, max_scene_cloud=80.0)
        super().__init__(cfg, workers)

    def _read_window(self, item, lat: float, lon: float, radius_m: float | None = None) -> dict:
        import planetary_computer as pc
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import transform as warp_transform
        from rasterio.windows import from_bounds

        crs = item_crs(item)
        if crs is None:
            with rasterio.open(pc.sign(item.assets["lwir11"].href)) as src:
                crs = src.crs.to_string()
        x, y = (v[0] for v in warp_transform("EPSG:4326", crs, [lon], [lat]))
        r = float(radius_m if radius_m is not None else self.cfg.radius_m)
        n = int(round(2 * r / PIXEL_M))

        def read(key: str, resampling) -> np.ndarray:
            href = pc.sign(item.assets[key].href)
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="3",
                              GDAL_HTTP_RETRY_DELAY="1", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
                with rasterio.open(href) as src:
                    win = from_bounds(x - r, y - r, x + r, y + r, src.transform)
                    return src.read(1, window=win, out_shape=(n, n), boundless=True, fill_value=0,
                                    resampling=resampling)

        return {"st_k": kelvin_from_dn(read("lwir11", Resampling.nearest)),
                "qa": read("qa_pixel", Resampling.nearest).astype("uint16")}

    def extract_visit(self, station: str, lat: float, lon: float, date: dt.date, items: list) -> dict:
        base = {"station": station, "date": pd.Timestamp(date), "temp_status": "no_scene"}
        for diff, scene_date, item in rank_items(items, date, self.cfg.day_tolerance, self.cfg.candidates):
            try:
                win = self._read_window(item, lat, lon)
            except Exception as e:
                log.warning("landsat read failed %s: %s", item.id, e)
                base = {**base, "temp_status": "read_error"}
                continue
            summary = summarize_temperature(win["st_k"], win["qa"])
            result = {**base, **summary, "temp_scene_id": item.id, "temp_scene_date": pd.Timestamp(scene_date),
                      "temp_day_diff": int(diff)}
            if summary["temp_status"] == "ok":
                return result
            base = result
        return base
