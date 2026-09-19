"""Apply a model over reflectance rasters / arrays -> per-parameter maps.

Works for any sensor in the band contract (src/data/sensors.py). Bands are looked up BY NAME, so a river
export or a Landsat raster can never be mistaken for another layout (Landsat B5 is NIR, not red-edge).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import rasterio

from src.data import sensors
from src.features.feature_engineering import compute_all_features

NODATA = -9999.0


def _smooth(band: np.ndarray, valid: np.ndarray, size: int) -> np.ndarray:
    """Mean over the valid neighbours only (a NaN-safe box filter)."""
    from scipy.ndimage import uniform_filter

    v = valid.astype(float)
    num = uniform_filter(np.where(valid, band, 0.0), size=size, mode="nearest")
    den = uniform_filter(v, size=size, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def predict_arrays(bands: Dict[str, np.ndarray], water: np.ndarray, predict_fn: Callable[[pd.DataFrame], pd.DataFrame],
                   sensor: str = "S2", feature_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
                   smooth_px: int = 0, extra_layers: Sequence[str] = ("ndci",)) -> Dict[str, np.ndarray]:
    """Per-pixel model output as 2-D float32 arrays (NaN outside valid water pixels).

    ``predict_fn`` receives the feature frame and returns a frame of predicted parameters. ``extra_layers`` are
    feature columns (e.g. the NDCI index) returned as additional maps, prefixed ``idx_``.
    ``smooth_px`` > 1 averages each band over its valid neighbours first: models are trained on the median of many
    pixels, so single-pixel noise would otherwise produce speckle.
    """
    sensors.validate_bands(sensor, bands.keys())
    water = np.asarray(water, dtype=bool)
    shape = water.shape
    valid = water.copy()
    for name, arr in bands.items():
        valid &= np.isfinite(arr) & (arr > NODATA)

    frame = {}
    for name, arr in bands.items():
        a = np.asarray(arr, dtype=np.float64)
        frame[name] = (_smooth(a, valid, smooth_px) if smooth_px and smooth_px > 1 else a)[valid]
    df = pd.DataFrame(frame)

    out: Dict[str, np.ndarray] = {}
    if len(df) == 0:
        return out
    feats = feature_fn(df) if feature_fn else compute_all_features(df, sensor)
    preds = predict_fn(feats)
    for col in preds.columns:
        layer = np.full(shape, np.nan, dtype=np.float32)
        layer[valid] = preds[col].to_numpy(dtype=np.float32)
        out[col] = layer
    for col in extra_layers:
        if col in feats.columns:
            layer = np.full(shape, np.nan, dtype=np.float32)
            layer[valid] = feats[col].to_numpy(dtype=np.float32)
            out[f"idx_{col}"] = layer
    return out


def predict_raster(tif: Path | str, predict_fn: Callable[[pd.DataFrame], pd.DataFrame], sensor: str = "S2",
                   band_names: Optional[Sequence[str]] = None,
                   feature_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
                   out_dir: Path | str | None = None, smooth_px: int = 0) -> Dict[str, Path]:
    """Read an exported reflectance GeoTIFF (band order = the sensor's contract + trailing 'water') and write one
    single-band GeoTIFF per predicted parameter next to it (nodata = -9999)."""
    code = sensors.normalize_sensor(sensor)
    band_names = list(band_names or sensors.OUT_BANDS[code])
    tif = Path(tif)
    out_dir = Path(out_dir or tif.parent.parent / "predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(tif) as src:
        arr = src.read().astype("float64")
        profile = src.profile
    if arr.shape[0] != len(band_names):
        raise ValueError(f"{tif.name} has {arr.shape[0]} bands but the {code} contract expects {len(band_names)}: "
                         f"{band_names}")
    layers = {name: arr[i] for i, name in enumerate(band_names)}
    water = layers.pop("water") == 1
    maps = predict_arrays(layers, water, predict_fn, code, feature_fn, smooth_px)

    profile.update(count=1, dtype="float32", nodata=NODATA, compress="lzw")
    outputs: Dict[str, Path] = {}
    for name, layer in maps.items():
        dest = out_dir / f"{tif.stem}_{name}.tif"
        with rasterio.open(dest, "w", **profile) as dst:
            dst.write(np.where(np.isfinite(layer), layer, NODATA).astype("float32"), 1)
        outputs[name] = dest
    return outputs
