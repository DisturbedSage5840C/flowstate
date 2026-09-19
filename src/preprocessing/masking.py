try:
    import ee
except ImportError:
    ee = None
import numpy as np

# Band maps: canonical name -> collection band
S2_BANDS = {"B2": "B2", "B3": "B3", "B4": "B4", "B5": "B5", "B6": "B6", "B8": "B8", "B11": "B11"}
LANDSAT_BANDS = {"B3": "SR_B3", "B4": "SR_B4", "B5": "SR_B5", "B6": "SR_B6"}

# Valid-water thresholds are scene-dependent; these are only starting points.
DEFAULT_MNDWI_THRESHOLD = 0.0


def mask_s2_clouds(img: ee.Image) -> ee.Image:
    """QA60 opaque cloud (bit 10) + cirrus (bit 11), plus SCL cloud/shadow classes."""
    qa = img.select("QA60")
    qa_ok = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    scl = img.select("SCL")
    scl_ok = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))  # shadow, cloud med/high, cirrus
    return img.updateMask(qa_ok.And(scl_ok))


def mask_landsat_clouds(img: ee.Image) -> ee.Image:
    """QA_PIXEL bits: 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow."""
    qa = img.select("QA_PIXEL")
    ok = qa.bitwiseAnd(0b11110).eq(0)
    return img.updateMask(ok)


def scale_s2(img: ee.Image) -> ee.Image:
    refl = img.select(list(S2_BANDS.values())).multiply(1e-4)
    return img.addBands(refl, overwrite=True)


def scale_landsat(img: ee.Image) -> ee.Image:
    refl = img.select(list(LANDSAT_BANDS.values())).multiply(2.75e-5).add(-0.2)
    return img.addBands(refl, overwrite=True)


def mndwi(img: ee.Image, sensor: str) -> ee.Image:
    """MNDWI = (Green - SWIR1) / (Green + SWIR1) on scaled reflectance."""
    green, swir = ("B3", "B11") if sensor == "S2" else ("SR_B3", "SR_B6")
    return img.normalizedDifference([green, swir]).rename("MNDWI")


def water_mask(img: ee.Image, sensor: str, threshold: float = DEFAULT_MNDWI_THRESHOLD,
               erode_px: int = 0) -> ee.Image:
    """Binary water mask. For rivers pass erode_px=1 to drop mixed bank pixels."""
    mask = mndwi(img, sensor).gt(threshold)
    if erode_px > 0:
        mask = mask.focalMin(radius=erode_px, kernelType="square", units="pixels")
    return mask.rename("water")


def apply_water_mask(img: ee.Image, sensor: str, threshold: float = DEFAULT_MNDWI_THRESHOLD,
                     erode_px: int = 0) -> ee.Image:
    m = water_mask(img, sensor, threshold, erode_px)
    return img.addBands(mndwi(img, sensor)).addBands(m).updateMask(m)


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu split of an MNDWI sample; use to seed the per-site threshold, then eyeball the histogram."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    hist, edges = np.histogram(v, bins=bins, range=(-1, 1))
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = ((hist * centers).sum() - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[np.argmax(between)])


def suggest_threshold(img: ee.Image, sensor: str, region: ee.Geometry, scale: int = 20,
                      n: int = 5000) -> tuple[float, np.ndarray]:
    """Sample MNDWI over the site, return (otsu_threshold, samples) for plotting."""
    samples = mndwi(img, sensor).sample(region=region, scale=scale, numPixels=n, seed=0)
    vals = np.array(samples.aggregate_array("MNDWI").getInfo(), dtype=float)
    if vals.size < 50:
        raise RuntimeError("too few valid pixels to pick a threshold (cloudy scene or wrong AOI)")
    return otsu_threshold(vals), vals
