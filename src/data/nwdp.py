"""CPCB surface-water-quality data from India's National Water Data Portal (NWDP).

Open CKAN API, no login: https://nwdp.nwic.gov.in/api/3/action/package_show?id=<dataset>
Three datasets hold *manual* (grab-sample) CPCB measurements with station name, latitude,
longitude and timestamp:

  chemical   DO, pH, ammonia-N, nitrate, phosphorus, TDS, SAR, boron, ...
  biological BOD, COD, fecal / total coliform
  physical   turbidity, conductivity, temperature, ...

Findings that shaped this loader (checked against the live API):
  * files are one CSV per state per vintage, e.g. "(1961-2020)" and "(2021-2025)";
    the "(2026-2030)" files exist but are empty templates;
  * although labelled up to 2025, the rows currently end in Dec 2021;
  * chlorophyll-a is not measured; temperature is empty; turbidity is mostly present
    for 2020 only; some numeric cells are corrupted ("3.16.2") and are set to NaN;
  * in the 2020 rows of the older files, secondary chemistry columns (ammonia, SAR, nitrate, metals)
    hold colour words ("Clear", "Light Green"). Core values (DO, pH, BOD, turbidity, conductivity)
    in those rows are fine and are kept; the secondary group is nulled for those rows;
  * timestamps are day-first ("11-01-2021 08:30"), local time; there are duplicate rows.
"""

from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
API = "https://nwdp.nwic.gov.in/api/3/action"
CACHE_DIR = ROOT / "data" / "raw" / "nwdp"
DATASETS = {
    "chemical": "surface-water-quality-manual-chemical-parameters-cpcb",
    "biological": "surface-water-quality-manual-biological-parameters-cpcb",
    "physical": "surface-water-quality-manual-physical-parameters-cpcb",
}
USER_AGENT = "aqua-sense-hackathon/0.1 (research; contact via repo)"
INDIA_BOX = (6.0, 37.6, 68.0, 97.5)   # lat_min, lat_max, lon_min, lon_max

# (canonical name, matcher on the lower-cased source column name)
_COLUMN_RULES: dict[str, list[tuple[str, str]]] = {
    "common": [
        ("station", r"^station$"), ("agency", r"^agency$"), ("state", r"^state$"),
        ("district", r"^district$"), ("river", r"^river$"), ("basin", r"^basin$"),
        ("lat", r"^latitude"), ("lon", r"^longitude"), ("timestamp", r"^data acquisition time"),
    ],
    "chemical": [
        ("do", r"^dissolved oxygen"), ("ph", r"^potential of hydrogen"),
        ("ammonia_n", r"^amonia n|^ammonia n"), ("nitrate_n", r"^nitrate n"),
        ("nitrite_nitrate_n", r"^nitrite n"), ("total_p", r"^total phosphorus"),
        ("tds", r"^total dissolved solids"), ("sar", r"^sodium adsorption"),
        ("boron", r"^boron"), ("chloride", r"^chloride"), ("sulphate", r"^sulphate"),
        ("fluoride", r"^fluoride"), ("total_hardness", r"^total hardness"),
    ],
    "biological": [
        ("bod", r"^biochemical oxygen demand"), ("cod", r"^chemical oxygen demand"),
        ("fecal_coliform", r"^fecal coliform"), ("total_coliform", r"^total coliform"),
    ],
    "physical": [
        ("conductivity", r"^electric conductivity"), ("turbidity", r"^turbidity"),
        ("temp_c", r"^temperature"), ("total_solids", r"^total solids"),
    ],
}

_NULL_TOKENS = {"", "-", "--", "nan", "na", "n/a", "nil", "nd", "bdl", "<dl"}

# Core measurements per dataset: text here => the row is corrupt. Everything else is "secondary".
_CORE_COLUMNS = {"chemical": ["do", "ph"], "biological": ["bod"], "physical": ["turbidity", "conductivity"]}

# Physically plausible bounds; values outside become NaN (and are counted).
QC_BOUNDS = {
    "do": (0.0, 20.0), "bod": (0.0, 1000.0), "cod": (0.0, 5000.0), "ph": (2.0, 13.0),
    "turbidity": (0.0, 5000.0), "conductivity": (0.0, 100000.0), "temp_c": (0.0, 45.0),
    "ammonia_n": (0.0, 500.0), "total_coliform": (0.0, 1e9), "fecal_coliform": (0.0, 1e9),
    "sar": (0.0, 100.0), "boron": (0.0, 50.0), "tds": (0.0, 100000.0),
}

_GROUNDWATER_WORDS = r"(?<![a-z])(bore ?wells?|bore ?holes?|tube ?wells?|hand ?pumps?|open wells?|dug wells?|wells?|hpm|springs?|ground ?water|b/w)(?![a-z])"
_LAKE_WORDS = r"\b(lake|tank|pond|kere|talab|talao|sagar|sarovar|jheel|jhil|beel|wetland|lagoon|kunta|cheruvu)\b"
_RESERVOIR_WORDS = r"\b(reservoir|dam|barrage|weir|anicut)\b"
_RIVER_WORDS = (r"\b(river|nadi|nala|nalla|nallah|nullah|drain|canal|stream|creek|tributary|"
                r"d/s|u/s|downstream|upstream|bridge|ghat|confluence)\b")


# Some resources are named by file ("SWQ_Manual_Chemical_Parameters_CPCB_OD_1961_2020.csv") with a state code.
STATE_CODES = {
    "OD": "Odisha", "PB": "Punjab", "PY": "Puducherry", "RJ": "Rajasthan", "SK": "Sikkim",
    "TN": "Tamil Nadu", "TR": "Tripura", "TS": "Telangana", "UK": "Uttarakhand",
    "UP": "Uttar Pradesh", "WB": "West Bengal",
}


def parse_resource_name(name: str) -> tuple[str, str] | None:
    """(state, 'YYYY-YYYY') from either naming style; None if unrecognised."""
    m = re.search(r"CPCB (.+?)\s*\((\d{4})\s*-\s*(\d{4})\)", name)
    if m:
        return m.group(1).strip(), f"{m.group(2)}-{m.group(3)}"
    m = re.search(r"CPCB_([A-Z]{2})_(\d{4})_(\d{4})", name)
    if m and m.group(1) in STATE_CODES:
        return STATE_CODES[m.group(1)], f"{m.group(2)}-{m.group(3)}"
    return None


@dataclass(frozen=True)
class Resource:
    kind: str
    name: str
    state: str
    period: str
    url: str
    resource_id: str

    @property
    def filename(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", self.state.lower()).strip("_")
        return f"{self.kind}__{slug}__{self.period}__{self.resource_id[:8]}.csv"


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def list_resources(kind: str, session: requests.Session | None = None) -> list[Resource]:
    """All CSV resources of one NWDP dataset, parsed into (state, period, url)."""
    session = session or _session()
    r = session.get(f"{API}/package_show", params={"id": DATASETS[kind]}, timeout=60)
    r.raise_for_status()
    out = []
    for res in r.json()["result"]["resources"]:
        parsed = parse_resource_name(res["name"])
        if not parsed:
            log.warning("unparsed resource name: %s", res["name"])
            continue
        out.append(Resource(kind, res["name"], parsed[0], parsed[1], res["url"], res["id"]))
    return out


def fetch_resource(res: Resource, cache_dir: Path | str = CACHE_DIR, force: bool = False,
                   session: requests.Session | None = None, retries: int = 3) -> Path:
    """Download one CSV into the cache (idempotent)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / res.filename
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest
    session = session or _session()
    last = None
    for attempt in range(retries):
        try:
            r = session.get(res.url, timeout=120)
            r.raise_for_status()
            dest.write_bytes(r.content)
            return dest
        except requests.RequestException as e:      # network flake: back off and retry
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not download {res.url}: {last}")


def fetch_all(kinds=("chemical", "biological", "physical"), periods=("2021-2025", "1961-2020"),
              states: list[str] | None = None, cache_dir: Path | str = CACHE_DIR,
              workers: int = 4) -> list[Resource]:
    """Download every matching resource; returns the resources whose files are now cached."""
    sess = _session()
    wanted: list[Resource] = []
    for kind in kinds:
        for res in list_resources(kind, sess):
            if res.period not in periods:
                continue
            if states and res.state.lower() not in {s.lower() for s in states}:
                continue
            wanted.append(res)
    with cf.ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda r: fetch_resource(r, cache_dir), wanted))
    return wanted


# ---------------------------------------------------------------------------
# Parsing and cleaning
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def normalize_frame(raw: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Rename to canonical columns, parse types, apply physical bounds. Records QC counts in .attrs."""
    if raw.empty:
        return pd.DataFrame()
    rules = _COLUMN_RULES["common"] + _COLUMN_RULES[kind]
    rename = {}
    for col in raw.columns:
        low = str(col).strip().lower()
        for canon, pat in rules:
            if re.search(pat, low) and canon not in rename.values():
                rename[col] = canon
                break
    df = raw.rename(columns=rename)[list(rename.values())].copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M", errors="coerce")
    fallback = df["timestamp"].isna()
    if fallback.any():   # tolerate other day-first formats
        df.loc[fallback, "timestamp"] = pd.to_datetime(raw.loc[fallback, [c for c, v in rename.items() if v == "timestamp"][0]],
                                                       dayfirst=True, errors="coerce")
    for col in ("lat", "lon"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    qc = {"corrupt_numeric": 0, "out_of_bounds": 0, "rows_dropped_core_text": 0, "secondary_groups_nulled": 0}
    value_cols = [c for c, _ in _COLUMN_RULES[kind] if c in df.columns]
    core = [c for c in _CORE_COLUMNS[kind] if c in df.columns]
    secondary = [c for c in value_cols if c not in core]

    def has_text(col: str) -> pd.Series:
        as_text = df[col].astype(str).str.strip().str.lower()
        return df[col].notna() & as_text.str.contains(r"[a-z]", regex=True) & ~as_text.isin(_NULL_TOKENS)

    # The 2020 vintage of the older files stores colour descriptors ("Clear", "Light Green") in secondary
    # columns such as ammonia / SAR / metals. The core measurements in those same rows (DO, pH, BOD,
    # turbidity, conductivity) are valid, so: text inside a CORE column means the row itself is corrupt
    # (drop it); text inside a SECONDARY column means only that column group is unreliable for the row
    # (null the whole group, keep the row).
    if core:
        core_bad = pd.concat([has_text(c) for c in core], axis=1).any(axis=1)
        qc["rows_dropped_core_text"] = int(core_bad.sum())
        df = df[~core_bad].copy()
    if secondary:
        sec_bad = pd.concat([has_text(c) for c in secondary], axis=1).any(axis=1)
        qc["secondary_groups_nulled"] = int(sec_bad.sum())
        df.loc[sec_bad, secondary] = np.nan

    for canon in value_cols:
        num = pd.to_numeric(df[canon], errors="coerce")
        # remaining un-parseable cells are things like "3.16.2" (two numbers glued together)
        qc["corrupt_numeric"] += int((num.isna() & df[canon].notna()
                                      & ~df[canon].astype(str).str.strip().str.lower().isin(_NULL_TOKENS)).sum())
        lo, hi = QC_BOUNDS.get(canon, (-np.inf, np.inf))
        bad = num.notna() & ((num < lo) | (num > hi))
        qc["out_of_bounds"] += int(bad.sum())
        df[canon] = num.mask(bad)
    df.attrs["qc"] = qc
    df["kind"] = kind
    return df


def fix_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Swap lat/lon when swapped, drop rows outside India / null island. Adds coord_ok flag."""
    df = df.copy()
    lat_min, lat_max, lon_min, lon_max = INDIA_BOX
    lat, lon = df["lat"], df["lon"]
    swapped = (~lat.between(lat_min, lat_max)) & (~lon.between(lon_min, lon_max)) & \
              lon.between(lat_min, lat_max) & lat.between(lon_min, lon_max)
    df.loc[swapped, ["lat", "lon"]] = df.loc[swapped, ["lon", "lat"]].to_numpy()
    df["coord_swapped_fixed"] = swapped
    df["coord_ok"] = df["lat"].between(lat_min, lat_max) & df["lon"].between(lon_min, lon_max)
    return df


def classify_water_body(station: str, river: str | None = None) -> str:
    """Heuristic groundwater / reservoir / lake / river from the station name (a documented guess).

    "groundwater" (bore wells, hand pumps, springs...) shows up inside the surface-water datasets and can
    never be observed from space, so downstream satellite matching excludes it.
    """
    name = str(station).lower()
    if re.search(_GROUNDWATER_WORDS, name):      # groundwater points appear in the "surface water" files
        return "groundwater"
    if re.search(_RESERVOIR_WORDS, name):
        return "reservoir"
    if re.search(_LAKE_WORDS, name):
        return "lake"
    if re.search(_RIVER_WORDS, name) or (river and str(river).strip() not in ("", "-", "nan")):
        return "river"
    if re.search(r"at|d/s|u/s", name):    # "GANGA AT PATNA", "KRISHNA AT VEDADRI": CPCB river naming
        return "river"
    return "unknown"


def load_kind(kind: str, cache_dir: Path | str = CACHE_DIR, states: list[str] | None = None,
              periods=("2021-2025", "1961-2020"), since: str | None = "2019-01-01") -> pd.DataFrame:
    """Read and normalise all cached files of one kind; drops rows before ``since`` and duplicates."""
    cache_dir = Path(cache_dir)
    frames, qc_total = [], {"corrupt_numeric": 0, "out_of_bounds": 0, "rows_dropped_core_text": 0,
                            "secondary_groups_nulled": 0}
    for path in sorted(cache_dir.glob(f"{kind}__*.csv")):
        _, state_slug, period, _ = path.stem.split("__")
        if period not in periods:
            continue
        if states and state_slug not in {re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") for s in states}:
            continue
        df = normalize_frame(_read_csv(path), kind)
        if df.empty:
            continue
        for k in qc_total:
            qc_total[k] += df.attrs["qc"][k]
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if since:
        out = out[out["timestamp"] >= pd.Timestamp(since)]
    out = fix_coordinates(out)
    keys = ["station", "lat", "lon", "timestamp"]
    value_cols = [c for c in out.columns if c in QC_BOUNDS or c in ("nitrate_n", "nitrite_nitrate_n", "total_p",
                  "chloride", "sulphate", "fluoride", "total_hardness", "total_solids")]
    n_before = len(out)
    out = out.groupby(keys, as_index=False, dropna=False).agg(
        {**{c: "mean" for c in value_cols}, **{c: "first" for c in ("state", "district", "river", "basin",
                                                                      "coord_ok", "source_file") if c in out.columns}})
    out.attrs["qc"] = {**qc_total, "duplicates_merged": n_before - len(out)}
    return out


def build_insitu_table(cache_dir: Path | str = CACHE_DIR, states: list[str] | None = None,
                       periods=("2021-2025", "1961-2020"), since: str | None = "2019-01-01") -> pd.DataFrame:
    """One row per (station, timestamp) combining chemical + biological + physical measurements."""
    parts = {k: load_kind(k, cache_dir, states, periods, since) for k in DATASETS}
    parts = {k: v for k, v in parts.items() if not v.empty}
    if not parts:
        return pd.DataFrame()
    keys = ["station", "lat", "lon", "timestamp"]
    meta_cols = ["state", "district", "river", "basin", "coord_ok"]

    # metadata (state, river, ...) from whichever kind has the row first
    meta = pd.concat([d[keys + [c for c in meta_cols if c in d.columns]] for d in parts.values()])
    meta = meta.drop_duplicates(keys)

    wide = None
    for kind, df in parts.items():
        values = [c for c in df.columns if c not in keys + meta_cols + ["source_file"]]
        part = df[keys + values].copy()
        part[f"file_{kind}"] = df["source_file"]
        wide = part if wide is None else wide.merge(part, on=keys, how="outer")
    wide = wide.merge(meta, on=keys, how="left")

    wide["date"] = wide["timestamp"].dt.normalize()
    wide["water_body_type"] = [classify_water_body(st, rv) for st, rv in zip(wide["station"], wide["river"])]
    wide["source"] = "CPCB NWDP manual monitoring"
    wide["is_proxy"] = False
    wide["retrieved_on"] = dt.date.today().isoformat()
    wide = wide[wide["coord_ok"].fillna(False).astype(bool)].reset_index(drop=True)
    wide.attrs["qc"] = {k: p.attrs.get("qc", {}) for k, p in parts.items()}
    return wide
