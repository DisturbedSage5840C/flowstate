import json
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "sites.yaml"
DEFAULT_THRESHOLDS = ROOT / "config" / "mndwi_thresholds.yaml"
WATER_BODY_TYPES = {"lake", "river", "reservoir", "lagoon"}


@dataclass(frozen=True)
class Site:
    name: str
    type: str
    state: str
    bbox: tuple | None = None
    geojson: str | None = None
    mndwi_threshold: float | None = None   # per-site water threshold; None -> masking.DEFAULT_MNDWI_THRESHOLD
    altitude_m: float = 0.0                # approximate surface elevation, used for DO saturation

    def geometry(self) -> dict:
        """GeoJSON geometry dict (EPSG:4326)."""
        if self.geojson:
            gj = json.loads((ROOT / self.geojson).read_text())
            if gj.get("type") == "FeatureCollection":
                gj = gj["features"][0]
            return gj.get("geometry", gj)
        w, s, e, n = self.bbox
        return {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}

    def bounds(self) -> tuple[float, float, float, float]:
        """(west, south, east, north) of the bbox or of the GeoJSON geometry."""
        if self.bbox:
            return tuple(self.bbox)

        def walk(coords):
            if coords and isinstance(coords[0], (int, float)):
                yield coords
            else:
                for c in coords:
                    yield from walk(c)

        pts = list(walk(self.geometry()["coordinates"]))
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)

    def centroid(self) -> tuple[float, float]:
        """(lat, lon)"""
        if self.bbox:
            w, s, e, n = self.bbox
            return (s + n) / 2, (w + e) / 2
        ring = self.geometry()["coordinates"][0]
        return (sum(p[1] for p in ring) / len(ring), sum(p[0] for p in ring) / len(ring))


def _validate(raw: dict) -> Site:
    if raw.get("type") not in WATER_BODY_TYPES:
        raise ValueError(f"site {raw.get('name')!r}: type must be one of {sorted(WATER_BODY_TYPES)}")
    if not raw.get("bbox") and not raw.get("geojson"):
        raise ValueError(f"site {raw['name']!r}: needs bbox or geojson")
    bbox = tuple(raw["bbox"]) if raw.get("bbox") else None
    if bbox and not (len(bbox) == 4 and bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        raise ValueError(f"site {raw['name']!r}: bbox must be [west, south, east, north]")
    thr = raw.get("mndwi_threshold")
    if thr is not None and not -1.0 <= float(thr) <= 1.0:
        raise ValueError(f"site {raw['name']!r}: mndwi_threshold must be within [-1, 1]")
    return Site(raw["name"], raw["type"], raw.get("state", ""), bbox, raw.get("geojson"),
                None if thr is None else float(thr), float(raw.get("altitude_m", 0.0)))


def _load_thresholds(path: Path | str | None) -> dict:
    """Tuned per-site thresholds written by scripts/tune_mndwi.py (kept out of sites.yaml)."""
    if path is None or not Path(path).exists():
        return {}
    return (yaml.safe_load(Path(path).read_text()) or {}).get("thresholds", {})


def load_sites(path: Path | str = DEFAULT_CONFIG,
               thresholds_path: Path | str | None = DEFAULT_THRESHOLDS) -> dict[str, Site]:
    data = yaml.safe_load(Path(path).read_text())
    tuned = _load_thresholds(thresholds_path)
    raws = []
    for raw in data["sites"]:
        raw = dict(raw)
        if raw.get("mndwi_threshold") is None and raw["name"] in tuned:
            raw["mndwi_threshold"] = tuned[raw["name"]].get("threshold")
        raws.append(raw)
    sites = [_validate(s) for s in raws]
    names = [s.name for s in sites]
    if len(names) != len(set(names)):
        raise ValueError("duplicate site names in config")
    return {s.name: s for s in sites}


def get_site(name: str, path: Path | str = DEFAULT_CONFIG) -> Site:
    return load_sites(path)[name]
