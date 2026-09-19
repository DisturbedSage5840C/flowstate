"""Any-AOI prediction maps without Earth Engine: real Sentinel-2 window -> water mask -> per-pixel model output.

    result = predict_aoi(12.9359, 77.6701, "2020-01-29", half_size_m=2500)   # Bellandur Lake, needs network only

Outputs are GeoTIFFs in EPSG:4326 (nodata -9999) that ``src.app.raster_layers.add_raster_overlay`` can draw.
The maps are only as good as the models: see the out-of-fold skill in reports/real/metrics_table.csv.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from src.data.satellite_extract import SCL_CLOUDY, ExtractionConfig, StacExtractor, rank_items, water_mask
from src.preprocessing.predict_raster import NODATA, predict_arrays
from src.wqi.wqi_engine import compute_wqi_dataframe

ROOT = Path(__file__).resolve().parents[2]
REAL_MODELS = ROOT / "reports" / "real" / "models"
AOI_DIR = ROOT / "data" / "interim" / "aoi"


class AOIError(RuntimeError):
    """No usable scene / no water / no model for the requested AOI."""


@dataclass
class AOIScene:
    bands: Dict[str, np.ndarray]
    scl: np.ndarray
    transform: object                 # rasterio Affine (in ``crs``)
    crs: str
    item_id: str
    scene_date: dt.date
    scene_cloud: float
    lat: float
    lon: float
    half_size_m: float
    extra: dict = field(default_factory=dict)


def fetch_aoi_scene(lat: float, lon: float, date: dt.date | str, half_size_m: float = 2000.0,
                    tolerance_days: int = 5, max_window_cloud: float = 0.4, cfg: ExtractionConfig | None = None,
                    extractor: StacExtractor | None = None) -> Optional[AOIScene]:
    """Nearest-in-time Sentinel-2 scene with an acceptably clear window around the point, or None."""
    date = pd.Timestamp(date).date()
    cfg = cfg or ExtractionConfig(day_tolerance=tolerance_days, candidates=4)
    extractor = extractor or StacExtractor(cfg, workers=1)
    items = extractor.search(lat, lon, date, date)
    for diff, scene_date, item in rank_items(items, date, tolerance_days, limit=cfg.candidates):
        win = extractor._read_window(item, lat, lon, radius_m=half_size_m)
        cloud_frac = float(np.isin(win["scl"], SCL_CLOUDY).mean())
        if cloud_frac > max_window_cloud:
            continue
        return AOIScene(win["bands"], win["scl"], win["transform"], win["crs"], item.id, scene_date,
                        float(item.properties.get("eo:cloud_cover", np.nan)), lat, lon, half_size_m,
                        {"day_diff": int(diff), "window_cloud_frac": cloud_frac})
    return None


def write_wgs84(array: np.ndarray, transform, crs: str, dest: Path | str, nodata: float = NODATA) -> Path:
    """Write a 2-D array as a single-band GeoTIFF reprojected to EPSG:4326 (NaN -> nodata)."""
    import rasterio
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = np.where(np.isfinite(array), array, nodata).astype("float32")
    h, w = src.shape
    left, top = transform * (0, 0)
    right, bottom = transform * (w, h)
    dst_transform, dw, dh = calculate_default_transform(crs, "EPSG:4326", w, h, left=left, bottom=bottom,
                                                         right=right, top=top)
    out = np.full((dh, dw), nodata, dtype="float32")
    reproject(src, out, src_transform=transform, src_crs=crs, dst_transform=dst_transform, dst_crs="EPSG:4326",
              src_nodata=nodata, dst_nodata=nodata, resampling=Resampling.nearest)
    with rasterio.open(dest, "w", driver="GTiff", height=dh, width=dw, count=1, dtype="float32", crs="EPSG:4326",
                       transform=dst_transform, nodata=nodata, compress="lzw") as dst:
        dst.write(out, 1)
    return dest


def make_predict_fn(models_dir: Path | str = REAL_MODELS) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """XGBoost predictions (DO, BOD, turbidity) plus a WQI computed from those predictions only.

    Checks for and loads the model eagerly (raises AOIError with no network call if it is missing) so
    ``predict_aoi`` can fail fast before fetching a scene. The returned function only has spectral
    features to work with; wrap it with ``with_context`` once a scene (and its lat/lon/date) is known,
    since the BOD/turbidity models also need water-body-type/season/urban-proximity columns
    (src.models.schema.FEATURE_SETS) that a per-pixel spectral frame alone cannot supply.
    """
    from src.models.xgboost_pipeline import WaterQualityXGB

    models_dir = Path(models_dir)
    if not any(models_dir.glob("*_xgb.json")):
        raise AOIError(f"no trained models in {models_dir}; run python -m scripts.train_real_models")
    model = WaterQualityXGB.load(models_dir)

    def predict(feats: pd.DataFrame) -> pd.DataFrame:
        preds = model.predict(feats)
        preds["wqi"] = compute_wqi_dataframe(preds[[c for c in ("do", "bod", "turbidity") if c in preds]])["wqi"]
        return preds

    return predict


def with_context(predict_fn: Callable[[pd.DataFrame], pd.DataFrame], lat: float, lon: float,
                 date: dt.date | str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Wrap a predict_fn so every pixel also carries the non-spectral context features a BOD/turbidity
    model needs: per-pixel urban proximity and season, both computed once from the scene's own
    lat/lon/date (one Sentinel-2 window is small enough, and covers one date, so every pixel in it
    shares the same value) via the same helpers the training table uses
    (src.data.city_proximity.urban_proxy_features, src.models.schema.add_season_onehot). Water-body
    type is not knowable for an arbitrary AOI point and is left at the all-zero "unknown" baseline,
    the same convention src.models.schema.add_water_body_onehot uses for the training table.
    """
    from src.data.city_proximity import urban_proxy_features
    from src.models.schema import TYPE_COLS, add_season_onehot

    urban = urban_proxy_features(pd.Series([lat]), pd.Series([lon])).iloc[0].to_dict()
    season_row = add_season_onehot(pd.DataFrame({"date": [pd.Timestamp(date)]})).iloc[0]
    context = {**urban, **{c: int(season_row[c]) for c in ("is_winter", "is_summer", "is_monsoon")},
              **{c: 0 for c in TYPE_COLS}}

    def predict(feats: pd.DataFrame) -> pd.DataFrame:
        feats = feats.copy()
        for col, val in context.items():
            feats[col] = val
        return predict_fn(feats)

    return predict


def predict_aoi_from_scene(scene: AOIScene, predict_fn: Callable[[pd.DataFrame], pd.DataFrame], out_dir: Path | str,
                           name: str, smooth_px: int = 3, cfg: ExtractionConfig | None = None) -> dict:
    cfg = cfg or ExtractionConfig()
    water = water_mask(scene.bands, scene.scl, cfg)
    n_water = int(water.sum())
    if n_water < cfg.min_water_px:
        raise AOIError(f"only {n_water} water pixels in this window: pick a point over water or a larger area")
    maps = predict_arrays(scene.bands, water, predict_fn, "S2", smooth_px=smooth_px,
                          extra_layers=("ndci", "turbidity_empirical"))
    out_dir = Path(out_dir)
    layers = {key: write_wgs84(arr, scene.transform, scene.crs, out_dir / f"{name}_{key}.tif")
              for key, arr in maps.items()}
    return {
        "layers": layers, "n_water_px": n_water, "water_frac": float(water.mean()),
        "scene_id": scene.item_id, "scene_date": str(scene.scene_date), "scene_cloud": scene.scene_cloud,
        "day_diff": scene.extra.get("day_diff"), "window_cloud_frac": scene.extra.get("window_cloud_frac"),
        "half_size_m": scene.half_size_m, "lat": scene.lat, "lon": scene.lon,
        "note": "Model maps are experimental: check the out-of-fold skill (R2) in reports/real/metrics_table.csv.",
    }


def predict_aoi(lat: float, lon: float, date: dt.date | str, models_dir: Path | str = REAL_MODELS,
                out_dir: Path | str = AOI_DIR, half_size_m: float = 2000.0, tolerance_days: int = 5,
                smooth_px: int = 3) -> dict:
    """End to end: find a scene, mask water, predict, write GeoTIFFs. Raises AOIError with a readable reason."""
    predict_fn = make_predict_fn(models_dir)                    # fails fast (no network) if models are missing
    scene = fetch_aoi_scene(lat, lon, date, half_size_m, tolerance_days)
    if scene is None:
        raise AOIError(f"no clear Sentinel-2 scene within ±{tolerance_days} days of {date} at ({lat:.4f}, {lon:.4f})")
    predict_fn = with_context(predict_fn, scene.lat, scene.lon, scene.scene_date)
    name = f"aoi_{lat:.4f}_{lon:.4f}_{scene.scene_date}"
    return predict_aoi_from_scene(scene, predict_fn, out_dir, name, smooth_px)
