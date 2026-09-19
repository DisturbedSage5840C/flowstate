"""Earth Engine ingestion. Produces masked, water-only surface-reflectance scenes per site."""
import os
from pathlib import Path

import ee

from src.acquisition.sites import ROOT, Site
from src.preprocessing import masking

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
LANDSAT_COLLECTIONS = ["LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC09/C02/T1_L2"]

# Handoff contract with Marutey: band order of exported rasters (+ trailing 'water' mask band).
OUT_BANDS = {
    "S2": ["B2", "B3", "B4", "B5", "B8", "B11", "water"],
    "LS": ["B3", "B4", "B5", "B6", "water"],
}
# Rivers: 10 m S2 bands only, and erode the mask to avoid mixed bank pixels.
RIVER_S2_BANDS = ["B2", "B3", "B4", "B8", "water"]


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


def fetch_scenes(site: Site, start: str, end: str, sensor: str = "S2", max_cloud: float = 30,
                 mndwi_threshold: float = masking.DEFAULT_MNDWI_THRESHOLD) -> ee.ImageCollection:
    """Cloud-masked, MNDWI water-masked scenes clipped to `site`, with canonical band names.

    sensor: 'S2' (Sentinel-2 SR) or 'LS' (Landsat 8/9 C2 L2, gap filler).
    Bands follow OUT_BANDS; each image carries `site`, `sensor`, `date` (YYYYMMDD) properties.
    """
    if sensor not in OUT_BANDS:
        raise ValueError("sensor must be 'S2' or 'LS'")
    region = site_region(site)
    erode = 1 if site.type == "river" else 0
    is_s2 = sensor == "S2"
    col = _s2_collection(region, start, end, max_cloud) if is_s2 else _landsat_collection(region, start, end, max_cloud)
    keep = (RIVER_S2_BANDS if (site.type == "river" and is_s2) else OUT_BANDS[sensor])
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


def export_scene(site: Site, col: ee.ImageCollection, date: str, sensor: str = "S2",
                 out_dir: Path | str = ROOT / "data" / "interim", scale: int | None = None) -> Path:
    """Download one day's mosaic as GeoTIFF named {site}_{sensor}_{YYYYMMDD}.tif (small AOIs only;
    for large sites like Chilika use ee.batch.Export.image.toDrive instead)."""
    import geemap
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = col.filter(ee.Filter.eq("date", date)).mosaic().unmask(-9999)
    dest = out_dir / f"{scene_name(site, sensor, date)}.tif"
    scale = scale or (10 if sensor == "S2" else 30)
    geemap.ee_export_image(img, filename=str(dest), scale=scale, region=site_region(site),
                           file_per_band=False, crs="EPSG:4326")
    return dest
