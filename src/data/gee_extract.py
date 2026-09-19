"""Station-visit Sentinel-2 extraction on Google Earth Engine (same output schema as ``StacExtractor``).

Uses ``COPERNICUS/S2_SR_HARMONIZED`` (the same Sen2Cor L2A product as Planetary Computer, with the post-2022 offset
already harmonised) and the same rules: nearest clear scene within +/-3 days, water = MNDWI > 0 & NIR < 0.20 & SCL not
cloud/shadow/cirrus/snow, median reflectance of water pixels within 500 m of the station.

Differences from the STAC implementation (quantified by ``scripts/compare_backends.py``):
  * 20 m bands are sampled at 10 m with Earth Engine's default nearest-neighbour resampling (STAC: bilinear);
  * pixel counts are coverage-weighted (may be fractional at the buffer edge);
  * ``B4_std`` is not computed;
  * scene ids are Earth Engine ``system:index`` values.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.satellite_extract import ExtractionConfig

log = logging.getLogger(__name__)

COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
BANDS = ["B2", "B3", "B4", "B5", "B6", "B8", "B11"]
SCL_BAD = [0, 1, 3, 8, 9, 10, 11]
SCL_CLOUDY = [3, 8, 9, 10]


# ---------------------------------------------------------------------------
# Pure logic (unit-tested without Earth Engine)
# ---------------------------------------------------------------------------

def interpret_candidates(cands: list[dict] | None, cfg: ExtractionConfig) -> dict:
    """Pick the first usable candidate scene (already ordered nearest-in-time first) and shape it like the STAC output."""
    result = {"status": "no_scene"}
    for c in cands or []:
        n_all = float(c.get("n_all") or 0.0)
        n_water = float(c.get("n_water") or 0.0)
        n_cloudy = float(c.get("n_cloudy") or 0.0)
        cloud_frac = n_cloudy / n_all if n_all > 0 else 1.0
        base = {"scene_id": c.get("id"), "day_diff": int(c.get("dd", 0)), "scene_cloud": c.get("scene_cloud"),
                "scene_date": pd.Timestamp(int(c["t"]), unit="ms").normalize() if c.get("t") is not None else pd.NaT,
                "cloud_frac": cloud_frac, "n_circle_px": n_all, "n_water_px": n_water,
                "water_frac": n_water / n_all if n_all > 0 else 0.0, "sensor": "Sentinel-2"}
        if cloud_frac > cfg.max_cloud_frac:
            result = {**base, "status": "cloudy", "n_water_px": 0.0}
            continue
        if n_water < cfg.min_water_px or any(c.get(b) is None for b in BANDS):
            result = {**base, "status": "no_water"}
            continue
        return {**base, "status": "ok", **{b: float(c[b]) for b in BANDS}}
    return result


# ---------------------------------------------------------------------------
# Earth Engine access
# ---------------------------------------------------------------------------

class GeeExtractor:
    def __init__(self, cfg: ExtractionConfig | None = None, project: str = "prayashack", batch_size: int = 40):
        import ee

        self.ee = ee
        self.cfg = cfg or ExtractionConfig()
        self.batch_size = batch_size
        ee.Initialize(project=project)

    # ---- server-side pieces ---------------------------------------------
    def _candidate_stats(self, img, region):
        ee, cfg = self.ee, self.cfg
        proj = ee.Image(img).select("B2").projection()
        scaled = ee.Image(img).select(BANDS).multiply(1e-4)
        scl = ee.Image(img).select("SCL")
        bad = scl.remap(SCL_BAD, [1] * len(SCL_BAD), 0)
        cloudy = scl.remap(SCL_CLOUDY, [1] * len(SCL_CLOUDY), 0)
        b3, b8, b11 = scaled.select("B3"), scaled.select("B8"), scaled.select("B11")
        mndwi = b3.subtract(b11).divide(b3.add(b11))
        water = mndwi.gt(cfg.mndwi_threshold).And(b8.lt(cfg.nir_max)).And(bad.Not()).And(b3.gt(0))
        med = scaled.updateMask(water).reduceRegion(ee.Reducer.median(), region, 10, proj, None, False, 1e8)
        counts = ee.Image.cat([water.rename("n_water"), cloudy.rename("n_cloudy"),
                               ee.Image.constant(1).rename("n_all")]).reduceRegion(
            ee.Reducer.sum(), region, 10, proj, None, False, 1e8)
        meta = ee.Dictionary({"id": ee.Image(img).get("system:index"), "t": ee.Image(img).get("system:time_start"),
                              "dd": ee.Image(img).get("dd"), "scene_cloud": ee.Image(img).get("CLOUDY_PIXEL_PERCENTAGE")})
        return meta.combine(med, True).combine(counts, True)

    def _per_feature(self, feature):
        ee, cfg = self.ee, self.cfg
        pt = feature.geometry()
        region = pt.buffer(cfg.radius_m)
        t = ee.Date(ee.Number(feature.get("t")))
        col = (ee.ImageCollection(COLLECTION).filterBounds(pt)
               .filterDate(t.advance(-cfg.day_tolerance, "day"), t.advance(cfg.day_tolerance + 1, "day"))
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cfg.max_scene_cloud)))

        def add_rank(im):
            im = ee.Image(im)
            dd = ee.Number(im.date().difference(t, "day")).round().abs()
            return im.set({"dd": dd, "rank": dd.multiply(1000).add(ee.Number(im.get("CLOUDY_PIXEL_PERCENTAGE")))})

        cands = col.map(add_rank).sort("rank").toList(cfg.candidates)
        return feature.set("cands", cands.map(lambda im: self._candidate_stats(im, region)))

    # ---- client side ------------------------------------------------------
    def _run_batch(self, batch: pd.DataFrame, retries: int = 3) -> dict[int, list[dict]]:
        ee = self.ee
        feats = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]),
                            {"idx": int(i), "t": int(pd.Timestamp(r.date).value // 1_000_000)})
                 for i, r in batch.iterrows()]
        fc = ee.FeatureCollection(feats).map(self._per_feature)
        for attempt in range(retries):
            try:
                info = fc.getInfo()
                return {f["properties"]["idx"]: f["properties"].get("cands", []) for f in info["features"]}
            except Exception as e:                                   # quota / timeout: back off, then split
                log.warning("Earth Engine batch failed (%s)", str(e)[:120])
                time.sleep(3 * (attempt + 1))
        if len(batch) > 1:
            half = len(batch) // 2
            out = self._run_batch(batch.iloc[:half], retries=1)
            out.update(self._run_batch(batch.iloc[half:], retries=1))
            return out
        return {int(batch.index[0]): []}

    def extract(self, visits: pd.DataFrame, cache_path: Path | str | None = None) -> pd.DataFrame:
        """``visits``: columns station, lat, lon, date. Resumable when ``cache_path`` is given."""
        visits = visits.drop_duplicates(["station", "date"]).reset_index(drop=True)
        cache_path = Path(cache_path) if cache_path else None
        done: list[dict] = []
        done_keys: set = set()
        if cache_path and cache_path.exists():
            prev = pd.read_parquet(cache_path)
            done, done_keys = prev.to_dict("records"), set(zip(prev["station"], pd.to_datetime(prev["date"])))
        todo = visits[[(s, pd.Timestamp(d)) not in done_keys for s, d in zip(visits["station"], visits["date"])]]
        t0 = time.time()
        for start in range(0, len(todo), self.batch_size):
            batch = todo.iloc[start:start + self.batch_size]
            results = self._run_batch(batch)
            for i, row in batch.iterrows():
                rec = interpret_candidates(results.get(int(i), []), self.cfg)
                done.append({"station": row.station, "date": pd.Timestamp(row.date), **rec})
            log.info("  %d/%d visits (%.0fs)", min(start + self.batch_size, len(todo)), len(todo), time.time() - t0)
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(done).to_parquet(cache_path, index=False)
        return pd.DataFrame(done)
