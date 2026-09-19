"""Convert atmospheric-correction output into the pipeline's band-contract GeoTIFF.

``correct_scene`` used to return file paths that nothing consumed, so the downstream pipeline only ever saw the
Earth Engine fallback. ``acolite_to_contract`` turns ACOLITE's per-wavelength GeoTIFFs into one
``{site}_S2_{YYYYMMDD}.tif`` with bands B2,B3,B4,B5,B6,B8,B11,water (src/data/sensors.py) and provenance tags.

ASSUMPTIONS (unverified: ACOLITE is not installed in the development environment)
  * files are named ``*_<PRODUCT>_<wavelength_nm>.tif`` (e.g. ``..._L2W_Rrs_665.tif``);
  * the 1610 nm band is present (needed for MNDWI); if it is not, a clear error is raised instead of guessing;
  * Rrs (sr^-1) is multiplied by pi to obtain the dimensionless water reflectance the models were trained on.
C2RCC GeoTIFFs written by SNAP GPT do not carry band names, so they are NOT converted (read the BEAM-DIMAP
product with SNAP instead).
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import rasterio

from src.data import sensors

S2_BAND_CENTERS_NM = {"B2": 492, "B3": 560, "B4": 665, "B5": 704, "B6": 740, "B8": 833, "B11": 1614}


class ConversionError(RuntimeError):
    pass


def date_from_safe(safe_path: str | Path) -> str:
    """YYYYMMDD from a Sentinel-2 SAFE / product name such as S2A_MSIL1C_20200129T051041_N0208_..."""
    m = re.search(r"_(\d{8})T\d{6}", Path(safe_path).name)
    if not m:
        raise ConversionError(f"cannot read the acquisition date from {Path(safe_path).name!r}")
    return m.group(1)


def _wavelength_files(directory: Path, product: str) -> dict[float, Path]:
    pat = re.compile(rf"_{re.escape(product)}_(\d+(?:\.\d+)?)\.tif$")
    found = {}
    for p in directory.glob("*.tif"):
        m = pat.search(p.name)
        if m:
            found[float(m.group(1))] = p
    return found


def acolite_to_contract(acolite_dir: str | Path, dest: str | Path, product: str = "Rrs", max_offset_nm: float = 25.0,
                        mndwi_threshold: float = 0.0, nir_max: float = 0.20) -> Path:
    """Stack ACOLITE ``<product>_<nm>`` GeoTIFFs into a contract raster (+ water mask band) with provenance tags."""
    acolite_dir, dest = Path(acolite_dir), Path(dest)
    files = _wavelength_files(acolite_dir, product)
    if not files:
        raise ConversionError(f"no '*_{product}_<nm>.tif' files in {acolite_dir}")

    chosen: dict[str, Path] = {}
    missing = []
    for band, center in S2_BAND_CENTERS_NM.items():
        wl = min(files, key=lambda w: abs(w - center))
        if abs(wl - center) <= max_offset_nm:
            chosen[band] = files[wl]
        else:
            missing.append(f"{band} (~{center} nm)")
    if missing:
        raise ConversionError(f"ACOLITE output lacks bands {missing}; export them (e.g. include SWIR) or use the "
                              "Earth Engine surface-reflectance fallback")

    factor = math.pi if product == "Rrs" else 1.0
    arrays, profile = {}, None
    for band in sensors.BANDS["S2"]:
        with rasterio.open(chosen[band]) as src:
            arr = src.read(1).astype("float64")
            if profile is None:
                profile = src.profile.copy()
            elif arr.shape != (profile["height"], profile["width"]):
                raise ConversionError(f"{chosen[band].name} is on a different grid than the other bands")
            nodata = src.nodata
        arr = np.where(np.isfinite(arr), arr, np.nan)
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        arrays[band] = arr * factor

    with np.errstate(invalid="ignore", divide="ignore"):
        mndwi = (arrays["B3"] - arrays["B11"]) / (arrays["B3"] + arrays["B11"])
    water = np.isfinite(mndwi) & (mndwi > mndwi_threshold) & (arrays["B8"] < nir_max)

    stack = [np.where(np.isfinite(arrays[b]), arrays[b], -9999.0) for b in sensors.BANDS["S2"]]
    stack.append(water.astype("float64"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    profile.update(count=len(stack), dtype="float32", nodata=-9999.0, compress="lzw")
    with rasterio.open(dest, "w", **profile) as out:
        out.write(np.stack(stack).astype("float32"))
        out.update_tags(CORRECTION_METHOD="acolite_dsf", SOURCE_PRODUCT=product,
                        UNITS="water reflectance (pi * Rrs)" if product == "Rrs" else "surface reflectance",
                        SOURCE_DIR=str(acolite_dir), BANDS=",".join(sensors.OUT_BANDS["S2"]))
    return dest


def read_provenance(tif: str | Path) -> dict:
    with rasterio.open(tif) as src:
        return dict(src.tags())
