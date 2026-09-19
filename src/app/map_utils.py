"""
map_utils.py — Folium/deck.gl layer builders for the Aqua-Sense dashboard (Jashan / P4)

Functions
---------
build_wqi_map(predictions_df, sites_cfg)
    → folium.Map with WQI markers, colour-coded by WQI tier

add_site_markers(fmap, sites_cfg)
    → add circle markers for each preset site

build_parameter_layer(fmap, predictions_df, parameter, sites_cfg)
    → coloured markers for a chosen parameter (Chl-a / Turbidity / DO / WQI)

geojson_preview_layer(fmap, geojson_dict)
    → overlay an uploaded AOI GeoJSON onto the map
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import folium
import numpy as np
import pandas as pd
from folium.plugins import HeatMap, MarkerCluster

from src.wqi.wqi_engine import WQI_TIERS, wqi_tier

# ─────────────────────────────────────────────────────────
# WQI colour tiers — defined once, in src/wqi/wqi_engine.py
# ─────────────────────────────────────────────────────────
WQI_CLASSES = {
    t.label: {"range": (t.lower, t.upper), "label": t.label, "color": t.color} for t in WQI_TIERS
}
NO_DATA_COLOR = "#888888"

# Visual ranges for each parameter (for colour normalisation)
PARAM_RANGES = {
    "chl_a":    (0,   100),   # µg/L
    "turbidity":(0,   200),   # NTU / FNU
    "do":       (0,   14),    # mg/L (inverted: low DO = bad)
    "bod":      (0,   30),    # mg/L
    "wqi":      (0,   100),
}


def _wqi_class(wqi_value: float):
    """Tier label ('Excellent' … 'Very Poor') for a 0-100 WQI score, or None for NaN."""
    tier = wqi_tier(wqi_value)
    return tier.label if tier else None


def _wqi_color(wqi_value: float) -> str:
    tier = wqi_tier(wqi_value)
    return tier.color if tier else NO_DATA_COLOR


def _fmt(value, digits: int) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "–"
    return "–" if not np.isfinite(v) else f"{v:.{digits}f}"


def _param_color(value: float, param: str) -> str:
    """Interpolate between green and red for a given parameter value."""
    lo, hi = PARAM_RANGES.get(param, (0, 100))
    # For DO, low is bad (invert)
    if param == "do":
        norm = 1.0 - np.clip((value - lo) / (hi - lo + 1e-9), 0, 1)
    else:
        norm = np.clip((value - lo) / (hi - lo + 1e-9), 0, 1)

    # Green (good) → Yellow → Red (bad)
    if norm < 0.5:
        r = int(255 * norm * 2)
        g = 200
    else:
        r = 200
        g = int(200 * (1 - (norm - 0.5) * 2))
    return f"#{r:02x}{g:02x}50"


def build_wqi_map(
    predictions_df: pd.DataFrame,
    sites_cfg: List[Dict],
    center: Optional[List[float]] = None,
    zoom: int = 5,
) -> folium.Map:
    """
    Build a Folium map with WQI choropleth markers for each site.

    Parameters
    ----------
    predictions_df : DataFrame with columns [site, chl_a, turbidity, do, wqi]
                     One row per site (latest prediction).
    sites_cfg      : list of site dicts from config/sites.yaml.
    center         : [lat, lon] map centre. Defaults to centre of India.
    zoom           : initial zoom level.

    Returns
    -------
    folium.Map
    """
    center = center or [22.5, 80.0]
    fmap = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="OpenStreetMap",
        prefer_canvas=True,
    )

    _add_legend(fmap)
    if predictions_df is not None and len(predictions_df) > 0:
        add_site_markers(fmap, sites_cfg, predictions_df)
    else:
        add_site_markers(fmap, sites_cfg, None)

    return fmap


def add_site_markers(
    fmap: folium.Map,
    sites_cfg: List[Dict],
    predictions_df: Optional[pd.DataFrame] = None,
    parameter: str = "wqi",
) -> folium.Map:
    """
    Add circle markers for each preset site.
    Colour is driven by WQI class (or any chosen parameter).

    Parameters
    ----------
    fmap           : existing folium map.
    sites_cfg      : site list from YAML.
    predictions_df : if provided, colour is derived from prediction values.
    parameter      : one of 'wqi', 'chl_a', 'turbidity', 'do'.
    """
    for site in sites_cfg:
        name = site.get("name")
        bbox = site.get("bbox", [])
        lat  = site.get("lat") or ((bbox[1] + bbox[3]) / 2 if len(bbox) == 4 else 20.0)
        lon  = site.get("lon") or ((bbox[0] + bbox[2]) / 2 if len(bbox) == 4 else 78.0)
        stype = site.get("type", "lake")
        display = site.get("display_name", name)

        # Determine colour
        color   = "#888888"
        tooltip  = f"<b>{display}</b><br>Type: {stype}"

        if (
            predictions_df is not None
            and len(predictions_df) > 0
            and "site" in predictions_df.columns
            and name in predictions_df["site"].values
        ):
            row = predictions_df[predictions_df["site"] == name].iloc[-1]
            if parameter == "wqi" and "wqi" in row:
                val   = float(row["wqi"])
                color = _wqi_color(val)
                cls   = _wqi_class(val)
                label = WQI_CLASSES[cls]["label"] if cls else "n/a"
                tooltip += (
                    f"<br>WQI: {val:.1f} ({label})"
                    f"<br>Chl-a: {_fmt(row.get('chl_a'), 2)} µg/L"
                    f"<br>Turbidity: {_fmt(row.get('turbidity'), 1)} NTU"
                    f"<br>DO: {_fmt(row.get('do'), 2)} mg/L"
                )
            elif parameter in row:
                val   = float(row[parameter])
                color = _param_color(val, parameter)
                tooltip += f"<br>{parameter}: {val:.2f}"

        icon_map = {"lake": "tint", "river": "water", "reservoir": "database"}
        folium.CircleMarker(
            location=[lat, lon],
            radius=12,
            color="white",
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            tooltip=folium.Tooltip(tooltip, sticky=True),
            popup=folium.Popup(tooltip, max_width=250),
        ).add_to(fmap)

    return fmap


def build_parameter_layer(
    fmap: folium.Map,
    predictions_df: pd.DataFrame,
    parameter: str,
    sites_cfg: List[Dict],
) -> folium.Map:
    """
    Update/re-colour site markers for a specific parameter.
    Returns the map with updated markers (creates a new FeatureGroup).
    """
    fg = folium.FeatureGroup(name=parameter.upper())
    for site in sites_cfg:
        name    = site.get("name")
        _bbox   = site.get("bbox", [])
        lat     = site.get("lat") or ((_bbox[1] + _bbox[3]) / 2 if len(_bbox) == 4 else 20.0)
        lon     = site.get("lon") or ((_bbox[0] + _bbox[2]) / 2 if len(_bbox) == 4 else 78.0)
        display = site.get("display_name", name)

        if predictions_df is not None and name in predictions_df["site"].values:
            row = predictions_df[predictions_df["site"] == name].iloc[-1]
            val   = float(row.get(parameter, 0) or 0)
            color = _param_color(val, parameter) if parameter != "wqi" else _wqi_color(val)

            units = {"chl_a": "µg/L", "turbidity": "NTU", "do": "mg/L", "wqi": ""}
            folium.CircleMarker(
                location=[lat, lon],
                radius=12,
                color="white",
                weight=1.5,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                tooltip=folium.Tooltip(
                    f"<b>{display}</b><br>{parameter}: {val:.2f} {units.get(parameter,'')}",
                    sticky=True,
                ),
            ).add_to(fg)

    fg.add_to(fmap)
    folium.LayerControl().add_to(fmap)
    return fmap


def geojson_preview_layer(
    fmap: folium.Map,
    geojson_data: dict | str,
    name: str = "Uploaded AOI",
) -> folium.Map:
    """
    Overlay a GeoJSON AOI onto the map and zoom to it.

    Parameters
    ----------
    fmap         : existing folium map.
    geojson_data : dict or JSON string.
    name         : layer name shown in LayerControl.
    """
    if isinstance(geojson_data, str):
        geojson_data = json.loads(geojson_data)

    folium.GeoJson(
        geojson_data,
        name=name,
        style_function=lambda _: {
            "fillColor": "#3388ff",
            "color":     "#0044ff",
            "weight":    2,
            "fillOpacity": 0.2,
        },
        tooltip=folium.GeoJsonTooltip(fields=[], aliases=[]),
    ).add_to(fmap)

    # Try to fit the map to the AOI bounds
    try:
        from shapely.geometry import shape
        geometries = [
            shape(feat["geometry"])
            for feat in geojson_data.get("features", [geojson_data])
            if feat.get("geometry")
        ]
        if geometries:
            from shapely.ops import unary_union
            union  = unary_union(geometries)
            bounds = union.bounds           # (minx, miny, maxx, maxy)
            fmap.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    except Exception:
        pass  # shapely unavailable — skip auto-zoom

    return fmap


def build_heatmap_layer(
    fmap: folium.Map,
    predictions_df: pd.DataFrame,
    sites_cfg: List[Dict],
    parameter: str = "wqi",
) -> folium.Map:
    """
    Add a HeatMap layer weighted by parameter value.

    Parameters
    ----------
    fmap           : folium map.
    predictions_df : predictions with site and parameter columns.
    sites_cfg      : site coordinates.
    parameter      : column to weight heat intensity.
    """
    def _site_latlon(s):
        bb = s.get("bbox", [])
        lat = s.get("lat") or ((bb[1] + bb[3]) / 2 if len(bb) == 4 else 0)
        lon = s.get("lon") or ((bb[0] + bb[2]) / 2 if len(bb) == 4 else 0)
        return lat, lon
    site_coords = {s["name"]: _site_latlon(s) for s in sites_cfg}
    heat_data = []
    lo, hi = PARAM_RANGES.get(parameter, (0, 100))

    for _, row in predictions_df.iterrows():
        name = row.get("site")
        if name in site_coords and parameter in row:
            lat, lon = site_coords[name]
            val = float(row[parameter] or 0)
            weight = np.clip((val - lo) / (hi - lo + 1e-9), 0, 1)
            heat_data.append([lat, lon, weight])

    if heat_data:
        HeatMap(
            heat_data,
            name=f"{parameter} heatmap",
            radius=25,
            blur=15,
            gradient={"0.0": "blue", "0.5": "lime", "1.0": "red"},
        ).add_to(fmap)

    return fmap


def _add_legend(fmap: folium.Map) -> None:
    """Inject the WQI tier legend into the map HTML."""
    rows = "<br>".join(
        f'<span style="color:{t.color}">&#9679;</span> {t.label} ({t.lower:g}–{t.upper:g})' for t in WQI_TIERS
    )
    legend_html = (
        '<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white; '
        'padding: 12px 16px; border-radius: 8px; box-shadow: 2px 2px 8px rgba(0,0,0,0.3); font-size:13px;">'
        f"<b>WQI (0 = pristine, 100 = worst)</b><br>{rows}</div>"
    )
    fmap.get_root().html.add_child(folium.Element(legend_html))
