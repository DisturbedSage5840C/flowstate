"""Sensor codes, band contract and matching tolerances shared by acquisition, features and the join.

Band contract (what every exported / extracted table or raster carries, in this order):

  S2 (Sentinel-2 L2A)  B2 490 nm, B3 560, B4 665, B5 705 (red-edge), B6 740 (red-edge 2), B8 842 (NIR),
                       B11 1610 (SWIR)  [+ 'water' mask band in rasters]
                       Rivers keep all bands: the 20 m bands are resampled to the 10 m grid, and Chl-a over narrow
                       turbid rivers is low-confidence.
  LS (Landsat 8/9 L2)  B3 green, B4 red, B5 NIR (NOT red-edge), B6 SWIR1  [+ 'water']
                       No red-edge band exists, so NDCI / Chl-a cannot be derived from Landsat; it is used for
                       turbidity, the water mask and (thermal band, separately) surface temperature.
"""

from __future__ import annotations

S2 = "S2"
LS = "LS"

_ALIASES = {
    "S2": S2, "S2A": S2, "S2B": S2, "SENTINEL2": S2, "SENTINEL-2": S2, "SENTINEL_2": S2, "MSI": S2,
    "LS": LS, "L8": LS, "L9": LS, "LC08": LS, "LC09": LS, "LANDSAT": LS, "LANDSAT8": LS, "LANDSAT9": LS,
    "LANDSAT-8": LS, "LANDSAT-9": LS, "LANDSAT_8": LS, "LANDSAT_9": LS,
}

BANDS = {
    S2: ["B2", "B3", "B4", "B5", "B6", "B8", "B11"],
    LS: ["B3", "B4", "B5", "B6"],
}
# Bands a feature computation needs at minimum (B6 is optional for S2: without it bdm3 is NaN).
REQUIRED_FOR_FEATURES = {
    S2: ("B3", "B4", "B5", "B8", "B11"),
    LS: ("B3", "B4", "B5", "B6"),
}
OUT_BANDS = {sensor: bands + ["water"] for sensor, bands in BANDS.items()}

# Plan: Sentinel-2 revisit ~5 d -> +/-3 d; Landsat ~16 d -> +/-5 d
DAY_TOLERANCE = {S2: 3, LS: 5}
SUPPORTS_CHL_A = {S2: True, LS: False}


def normalize_sensor(name: str) -> str:
    """Canonical code ('S2' or 'LS') for any common spelling; ValueError for anything unknown."""
    key = str(name).strip().upper().replace(" ", "")
    if key in _ALIASES:
        return _ALIASES[key]
    squashed = key.replace("-", "").replace("_", "")
    for alias, code in _ALIASES.items():
        if squashed == alias.replace("-", "").replace("_", ""):
            return code
    raise ValueError(f"unknown sensor {name!r}; expected one of {sorted(set(_ALIASES.values()))}")


def day_tolerance(name: str, default: int = DAY_TOLERANCE[S2]) -> int:
    try:
        return DAY_TOLERANCE[normalize_sensor(name)]
    except ValueError:
        return default


def validate_bands(sensor: str, columns) -> None:
    """Raise KeyError naming missing bands (never silently substitute another band)."""
    code = normalize_sensor(sensor)
    missing = [b for b in REQUIRED_FOR_FEATURES[code] if b not in set(columns)]
    if missing:
        raise KeyError(
            f"sensor {code!r} needs bands {list(REQUIRED_FOR_FEATURES[code])}; missing {missing}. "
            "River rasters must keep B5/B6/B11 (resampled to 10 m) - see src.data.sensors.")
