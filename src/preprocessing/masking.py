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


# Water is dark in the NIR; anything brighter is foam, sand, buildings or cloud edge. Same cap as the station
# extraction (src/data/satellite_extract.ExtractionConfig.nir_max) so rasters and station medians agree.
DEFAULT_NIR_MAX = 0.20
NIR_BAND = {"S2": "B8", "LS": "SR_B5"}


def water_mask(img: ee.Image, sensor: str, threshold: float = DEFAULT_MNDWI_THRESHOLD,
               erode_px: int = 0, nir_max: float | None = DEFAULT_NIR_MAX) -> ee.Image:
    """Binary water mask: MNDWI > threshold AND NIR < nir_max. For rivers pass erode_px=1 to drop mixed bank pixels."""
    mask = mndwi(img, sensor).gt(threshold)
    if nir_max is not None:
        mask = mask.And(img.select(NIR_BAND[sensor]).lt(nir_max))
    if erode_px > 0:
        mask = mask.focalMin(radius=erode_px, kernelType="square", units="pixels")
    return mask.rename("water")


def apply_water_mask(img: ee.Image, sensor: str, threshold: float = DEFAULT_MNDWI_THRESHOLD,
                     erode_px: int = 0, nir_max: float | None = DEFAULT_NIR_MAX) -> ee.Image:
    m = water_mask(img, sensor, threshold, erode_px, nir_max)
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
    # For well-separated clusters the between-class variance is flat across the gap; argmax would return its first
    # point (hugging one cluster), so take the middle of the plateau.
    best = np.flatnonzero(between >= between.max() * (1.0 - 1e-9))
    return float(centers[best[len(best) // 2]])


def bimodality(values: np.ndarray, threshold: float) -> tuple[float, float]:
    """(eta, minority_share): eta = between-class / total variance for the split at ``threshold`` (1 = perfectly
    two-cluster); minority_share = fraction of samples above the threshold."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    hi, lo = v[v > threshold], v[v <= threshold]
    if len(hi) == 0 or len(lo) == 0 or v.var() == 0:
        return 0.0, float(len(hi) / max(len(v), 1))
    between = (len(hi) * (hi.mean() - v.mean()) ** 2 + len(lo) * (lo.mean() - v.mean()) ** 2) / len(v)
    return float(between / v.var()), float(len(hi) / len(v))


def choose_threshold(values: np.ndarray, default: float = DEFAULT_MNDWI_THRESHOLD, min_eta: float = 0.8,
                     lo: float = -0.2, hi: float = 0.3) -> tuple[float, str]:
    """Otsu threshold only when the MNDWI sample is genuinely bimodal and the split is in a sane range.

    Water is usually a small minority of a site's bounding box, so the histogram is dominated by land and Otsu then
    splits the LAND into two groups (e.g. -0.38 for a city lake, which would call half the city water). In that case
    the standard default is kept. Returns (threshold, method).
    """
    thr = otsu_threshold(values)
    eta, share = bimodality(values, thr)
    if eta >= min_eta and lo <= thr <= hi and 0.02 <= share <= 0.98:
        return thr, "otsu"
    return default, f"default (otsu {thr:+.2f} rejected: eta={eta:.2f}, share={share:.2f})"


def suggest_threshold(img: ee.Image, sensor: str, region: ee.Geometry, scale: int = 20,
                      n: int = 5000) -> tuple[float, np.ndarray]:
    """Sample MNDWI over the site, return (otsu_threshold, samples) for plotting."""
    samples = mndwi(img, sensor).sample(region=region, scale=scale, numPixels=n, seed=0)
    vals = np.array(samples.aggregate_array("MNDWI").getInfo(), dtype=float)
    if vals.size < 50:
        raise RuntimeError("too few valid pixels to pick a threshold (cloudy scene or wrong AOI)")
    return otsu_threshold(vals), vals
