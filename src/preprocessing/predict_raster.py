"""Apply a model's predict(df) over an exported reflectance raster -> per-parameter GeoTIFFs."""
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import rasterio

NODATA = -9999.0
PARAMS = ["chl_a", "turbidity", "do"]


def predict_raster(tif: Path | str, band_names: list[str], predict_fn: Callable[[pd.DataFrame], pd.DataFrame],
                   feature_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
                   out_dir: Path | str | None = None) -> dict[str, Path]:
    """band_names must match the raster's band order (see gee.OUT_BANDS).

    feature_fn (Marutey's features module) turns reflectance columns into model inputs;
    predict_fn (Navya/Jashan) returns a DataFrame with chl_a, turbidity, do columns.
    """
    tif = Path(tif)
    out_dir = Path(out_dir or tif.parent.parent / "predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(tif) as src:
        arr = src.read().astype("float64")
        profile = src.profile
    if arr.shape[0] != len(band_names):
        raise ValueError(f"raster has {arr.shape[0]} bands, expected {len(band_names)}")
    h, w = arr.shape[1:]
    df = pd.DataFrame({b: arr[i].ravel() for i, b in enumerate(band_names)})
    valid = (df["water"] == 1) & (df.drop(columns="water") > NODATA).all(axis=1)
    feats = df.loc[valid].drop(columns="water")
    if feature_fn:
        feats = feature_fn(feats)
    preds = predict_fn(feats) if len(feats) else pd.DataFrame(columns=PARAMS)
    profile.update(count=1, dtype="float32", nodata=NODATA, compress="lzw")
    outputs = {}
    for p in PARAMS:
        if p not in preds:
            continue
        band = np.full(h * w, NODATA, dtype="float32")
        band[valid.to_numpy()] = preds[p].to_numpy(dtype="float32")
        dest = out_dir / f"{tif.stem}_{p}.tif"
        with rasterio.open(dest, "w", **profile) as dst:
            dst.write(band.reshape(h, w), 1)
        outputs[p] = dest
    return outputs
