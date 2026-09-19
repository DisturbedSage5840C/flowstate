"""folium layers for predicted parameters and WQI tiers."""
import base64
import io
import sys
from pathlib import Path

import folium
import matplotlib
import numpy as np
import rasterio

matplotlib.use("Agg")
import matplotlib.image  # noqa: E402
from matplotlib import colors  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.app.map_utils import WQI_CLASSES  # noqa: E402 — single source of truth for WQI tiers

NODATA = -9999.0
# WQI tiers mirror src/wqi/wqi_engine.py's pollution-index convention (0 = pure, higher = worse),
# via the same WQI_CLASSES the dashboard map legend uses (src/app/map_utils.py).
WQI_TIERS = [
    (info["range"][1], info["label"], info["color"]) for info in WQI_CLASSES.values()
]
PARAM_CMAPS = {"chl_a": "YlGn", "turbidity": "YlOrBr", "do": "RdYlBu", "wqi": "RdYlGn_r"}


def wqi_tier(value: float) -> tuple[str, str]:
    for upper, label, color in WQI_TIERS:
        if value < upper:
            return label, color
    return WQI_TIERS[-1][1:]


def base_map(lat: float, lon: float, zoom: int = 13) -> folium.Map:
    m = folium.Map(location=[lat, lon], zoom_start=zoom, tiles="OpenStreetMap")
    folium.TileLayer("Esri.WorldImagery", name="Satellite").add_to(m)
    return m


def add_raster_overlay(m: folium.Map, tif: Path | str, param: str, name: str | None = None,
                       vmin: float | None = None, vmax: float | None = None, opacity: float = 0.8) -> folium.Map:
    """Render a single-band prediction GeoTIFF (EPSG:4326) as a colour-mapped image overlay."""
    with rasterio.open(tif) as src:
        data = src.read(1).astype("float64")
        b = src.bounds
    valid = data != NODATA
    if not valid.any():
        return m
    vmin = float(np.nanmin(data[valid])) if vmin is None else vmin
    vmax = float(np.nanmax(data[valid])) if vmax is None else vmax
    norm = colors.Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-9))
    rgba = matplotlib.colormaps[PARAM_CMAPS.get(param, "viridis")](norm(np.where(valid, data, vmin)))
    rgba[..., 3] = np.where(valid, 1.0, 0.0)
    buf = io.BytesIO()
    matplotlib.image.imsave(buf, rgba, format="png")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    folium.raster_layers.ImageOverlay(image=url, bounds=[[b.bottom, b.left], [b.top, b.right]],
                                      opacity=opacity, name=name or param).add_to(m)
    return m


def add_wqi_legend(m: folium.Map) -> folium.Map:
    rows = "".join(f'<div><span style="background:{c};display:inline-block;width:12px;height:12px;'
                   f'margin-right:6px"></span>{label}</div>' for _, label, c in WQI_TIERS)
    html = ('<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;'
            f'padding:8px 10px;border:1px solid #999;font-size:12px"><b>WQI</b>{rows}</div>')
    m.get_root().html.add_child(folium.Element(html))
    return m


def build_map(site, overlays: dict[str, Path | str] | None = None) -> folium.Map:
    """site: acquisition.sites.Site; overlays: {param: prediction tif}."""
    lat, lon = site.centroid()
    m = base_map(lat, lon)
    folium.GeoJson(site.geometry(), name=site.name, style_function=lambda _: {"fill": False, "color": "#444"}).add_to(m)
    for param, tif in (overlays or {}).items():
        add_raster_overlay(m, tif, param)
    add_wqi_legend(m)
    folium.LayerControl().add_to(m)
    return m
