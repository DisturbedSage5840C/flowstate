"""Earth Engine ingestion. Produces masked, water-only surface-reflectance scenes per site."""
import os
from pathlib import Path

import ee

from src.acquisition.sites import ROOT, Site
from src.data import sensors
from src.preprocessing import masking

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
LANDSAT_COLLECTIONS = ["LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC09/C02/T1_L2"]

# Band contract (order of exported rasters incl. the trailing 'water' mask band): src/data/sensors.py
OUT_BANDS = sensors.OUT_BANDS
# Rivers keep the full band set (B5/B6/B11 are 20 m and are resampled to the 10 m export grid) so that
# NDCI / MNDWI can still be computed; the water mask is eroded by 1 px to avoid mixed bank pixels.
# Chl-a over narrow, turbid rivers is low-confidence and must be reported as such.


def init_ee(project: str | None = None, service_account_json: str | None = None) -> None:
    """Initialise Earth Engine. Uses EE_SERVICE_ACCOUNT_JSON / EE_PROJECT env vars if args omitted."""
    project = project or os.environ.get("EE_PROJECT")
    key = service_account_json or os.environ.get("EE_SERVICE_ACCOUNT_JSON")
    if key:
        import json
        email = json.loads(Path(key).read_text())["client_email"]
        ee.Initialize(ee.ServiceAccountCredentials(email, key), project=project)
    else:
        ee.Initialize(project=project)


def site_region(site: Site) -> ee.Geometry:
    return ee.Geometry(site.geometry())


def _s2_collection(region, start, end, max_cloud):
    return (ee.ImageCollection(S2_COLLECTION)
            .filterBounds(region).filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
            .map(masking.mask_s2_clouds).map(masking.scale_s2))


def _landsat_collection(region, start, end, max_cloud):
    cols = [ee.ImageCollection(c) for c in LANDSAT_COLLECTIONS]
    merged = cols[0].merge(cols[1])
    return (merged.filterBounds(region).filterDate(start, end)
            .filter(ee.Filter.lt("CLOUD_COVER", max_cloud))
            .map(masking.mask_landsat_clouds).map(masking.scale_landsat))


def resolve_mndwi_threshold(site: Site, override: float | None = None) -> float:
    """Explicit override > per-site tuned value > global default."""
    if override is not None:
        return override
    if site.mndwi_threshold is not None:
        return site.mndwi_threshold
    return masking.DEFAULT_MNDWI_THRESHOLD


def fetch_scenes(site: Site, start: str, end: str, sensor: str = "S2", max_cloud: float = 30,
                 mndwi_threshold: float | None = None) -> ee.ImageCollection:
    """Cloud-masked, MNDWI water-masked scenes clipped to `site`, with canonical band names.

    sensor: 'S2' (Sentinel-2 SR) or 'LS' (Landsat 8/9 C2 L2, gap filler).
    Bands follow OUT_BANDS; each image carries `site`, `sensor`, `date` (YYYYMMDD) properties.
    """
    if sensor not in OUT_BANDS:
        raise ValueError("sensor must be 'S2' or 'LS'")
    region = site_region(site)
    mndwi_threshold = resolve_mndwi_threshold(site, mndwi_threshold)
    erode = 1 if site.type == "river" else 0
    is_s2 = sensor == "S2"
    col = _s2_collection(region, start, end, max_cloud) if is_s2 else _landsat_collection(region, start, end, max_cloud)
    keep = OUT_BANDS[sensor]
    if is_s2:
        sel_in, sel_out = [b for b in keep if b != "water"], None
    else:
        sel_in = ["SR_" + b for b in keep if b != "water"]
        sel_out = [b for b in keep if b != "water"]

    def prep(img):
        out = masking.apply_water_mask(img, sensor, mndwi_threshold, erode)
        bands = out.select(sel_in)
        if sel_out:
            bands = bands.rename(sel_out)
        bands = bands.addBands(out.select("water"))
        date = ee.Date(img.get("system:time_start")).format("YYYYMMdd")
        return (bands.clip(region).set({"site": site.name, "sensor": sensor, "date": date,
                                        "system:time_start": img.get("system:time_start")}))

    return col.map(prep)


def list_dates(col: ee.ImageCollection) -> list[str]:
    return sorted(set(col.aggregate_array("date").getInfo()))


def scene_name(site: Site, sensor: str, date: str) -> str:
    return f"{site.name}_{sensor}_{date}"


# Earth Engine's direct download (getDownloadURL) rejects requests above 50331648 bytes.
MAX_DOWNLOAD_BYTES = 50_331_648


def estimate_export_bytes(bounds: tuple[float, float, float, float], scale_m: float, n_bands: int,
                          bytes_per_value: int = 4) -> int:
    """Approximate size of a float32 export of ``bounds`` (west, south, east, north) at ``scale_m``."""
    import math

    w, s, e, n = bounds
    mid_lat = math.radians((s + n) / 2)
    width_px = math.ceil((e - w) * 111_320 * math.cos(mid_lat) / scale_m)
    height_px = math.ceil((n - s) * 110_540 / scale_m)
    return width_px * height_px * n_bands * bytes_per_value


def tile_bounds(bounds: tuple[float, float, float, float], scale_m: float, n_bands: int,
                max_bytes: int = int(MAX_DOWNLOAD_BYTES * 0.8)) -> list[tuple[float, float, float, float]]:
    """Split ``bounds`` into a k x k grid so every tile stays below ``max_bytes`` (0.8 safety margin)."""
    import math

    total = estimate_export_bytes(bounds, scale_m, n_bands)
    k = max(1, math.ceil(math.sqrt(total / max_bytes)))
    w, s, e, n = bounds
    xs = [w + (e - w) * i / k for i in range(k + 1)]
    ys = [s + (n - s) * j / k for j in range(k + 1)]
    return [(xs[i], ys[j], xs[i + 1], ys[j + 1]) for j in range(k) for i in range(k)]


def merge_tiles(paths: list[Path | str], dest: Path | str) -> Path:
    """Mosaic downloaded tile GeoTIFFs into one file (nodata -9999)."""
    import rasterio
    from rasterio.merge import merge

    srcs = [rasterio.open(p) for p in paths]
    try:
        data, transform = merge(srcs, nodata=-9999)
        profile = srcs[0].profile.copy()
        profile.update(height=data.shape[1], width=data.shape[2], transform=transform, nodata=-9999, compress="lzw")
        with rasterio.open(dest, "w", **profile) as out:
            out.write(data)
    finally:
        for s in srcs:
            s.close()
    return Path(dest)


def export_scene(site: Site, col: ee.ImageCollection, date: str, sensor: str = "S2",
                 out_dir: Path | str = ROOT / "data" / "interim", scale: int | None = None) -> Path:
    """Download one day's mosaic as {site}_{sensor}_{YYYYMMDD}.tif in the band contract order.

    Sites whose export would exceed Earth Engine's direct-download limit are split into tiles that are
    downloaded separately and merged. UNTESTED against a live Earth Engine account (only the size/tiling/merge
    logic is unit-tested).
    """
    import geemap
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = col.filter(ee.Filter.eq("date", date)).mosaic().unmask(-9999)
    dest = out_dir / f"{scene_name(site, sensor, date)}.tif"
    scale = scale or (10 if sensor == "S2" else 30)
    tiles = tile_bounds(site.bounds(), scale, len(OUT_BANDS[sensor]))
    if len(tiles) == 1:
        geemap.ee_export_image(img, filename=str(dest), scale=scale, region=site_region(site),
                               file_per_band=False, crs="EPSG:4326")
        return dest
    parts = []
    for i, (w, s, e, n) in enumerate(tiles):
        part = out_dir / f"{dest.stem}_tile{i:02d}.tif"
        geemap.ee_export_image(img, filename=str(part), scale=scale, region=ee.Geometry.Rectangle([w, s, e, n]),
                               file_per_band=False, crs="EPSG:4326")
        parts.append(part)
    merge_tiles(parts, dest)
    for part in parts:
        part.unlink(missing_ok=True)
    return dest
