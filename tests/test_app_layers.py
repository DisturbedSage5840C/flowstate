"""Map/tier consistency and raster overlay rendering."""

import numpy as np
import pandas as pd
import pytest

from src.wqi.wqi_engine import WQI_TIERS, wqi_tier


def test_single_tier_definition_across_modules():
    from src.app import map_utils, raster_layers
    engine = [(t.label, t.lower, t.upper, t.color) for t in WQI_TIERS]
    assert [(k, *v["range"], v["color"]) for k, v in map_utils.WQI_CLASSES.items()] == engine
    assert [(lab, up, col) for up, lab, col in raster_layers.WQI_TIERS] == [(t.label, t.upper, t.color) for t in WQI_TIERS]
    for score in (0, 19.9, 20, 45, 79.9, 80, 100):
        assert raster_layers.wqi_tier(score) == (wqi_tier(score).label, wqi_tier(score).color)
        assert map_utils._wqi_class(score) == wqi_tier(score).label


def test_map_handles_nan_wqi_and_missing_values():
    from src.app.map_utils import build_wqi_map
    sites = [{"name": "a", "type": "lake", "bbox": [77, 12, 77.1, 12.1]},
             {"name": "b", "type": "river", "bbox": [78, 13, 78.1, 13.1]}]
    preds = pd.DataFrame({"site": ["a", "b"], "wqi": [np.nan, 55.0],
                          "chl_a": [np.nan, 12.0], "turbidity": [5.0, np.nan], "do": [6.0, 4.0]})
    fmap = build_wqi_map(preds, sites)
    html = fmap.get_root().render()
    assert "Very Poor" in html and "Excellent" in html          # legend built from the tiers


def test_raster_overlay_renders_on_current_matplotlib(tmp_path):
    import folium
    import rasterio
    from rasterio.transform import from_bounds
    from src.app.raster_layers import add_raster_overlay

    path = tmp_path / "pred.tif"
    arr = np.random.default_rng(0).random((1, 20, 20)).astype("float32")
    arr[0, :5, :5] = -9999.0                                     # nodata corner
    with rasterio.open(path, "w", driver="GTiff", height=20, width=20, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_bounds(77, 12, 77.1, 12.1, 20, 20), nodata=-9999.0) as dst:
        dst.write(arr)
    m = add_raster_overlay(folium.Map(), path, "chl_a")
    assert "data:image/png;base64" in m.get_root().render()


def test_all_nodata_raster_is_skipped(tmp_path):
    import folium
    import rasterio
    from rasterio.transform import from_bounds
    from src.app.raster_layers import add_raster_overlay

    path = tmp_path / "empty.tif"
    with rasterio.open(path, "w", driver="GTiff", height=4, width=4, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_bounds(0, 0, 1, 1, 4, 4), nodata=-9999.0) as dst:
        dst.write(np.full((1, 4, 4), -9999.0, dtype="float32"))
    assert "data:image/png" not in add_raster_overlay(folium.Map(), path, "do").get_root().render()
