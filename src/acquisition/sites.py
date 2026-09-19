import json
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "sites.yaml"
WATER_BODY_TYPES = {"lake", "river", "reservoir", "lagoon"}


@dataclass(frozen=True)
class Site:
    name: str
    type: str
    state: str
    bbox: tuple | None = None
    geojson: str | None = None

    def geometry(self) -> dict:
        """GeoJSON geometry dict (EPSG:4326)."""
        if self.geojson:
            gj = json.loads((ROOT / self.geojson).read_text())
            if gj.get("type") == "FeatureCollection":
                gj = gj["features"][0]
            return gj.get("geometry", gj)
        w, s, e, n = self.bbox
        return {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}

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
    return Site(raw["name"], raw["type"], raw.get("state", ""), bbox, raw.get("geojson"))


def load_sites(path: Path | str = DEFAULT_CONFIG) -> dict[str, Site]:
    data = yaml.safe_load(Path(path).read_text())
    sites = [_validate(s) for s in data["sites"]]
    names = [s.name for s in sites]
    if len(names) != len(set(names)):
        raise ValueError("duplicate site names in config")
    return {s.name: s for s in sites}


def get_site(name: str, path: Path | str = DEFAULT_CONFIG) -> Site:
    return load_sites(path)[name]
