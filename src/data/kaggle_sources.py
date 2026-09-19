"""Kaggle IoT dataset "River Ganga and Sangam, India" (dal4206) - loader with quality control.

Verdict from checking it against CPCB's own measurements at the same reach and dates
(see data/ground_truth/data_source_log.md):
  * temperature   plausible (clean seasonal cycle 17 -> 29 C)            -> USED
  * DO            plausible except impossible values (> 14 mg/L)          -> used with bounds
  * pH            NOT reliable: 83-85 % of readings exceed 8.5 (CPCB 6.8-8.5), monthly medians 9-13   -> excluded
  * conductivity  NOT reliable: scale jumps between ~1 and ~900 uS/cm     -> excluded
  * ORP, WQI, Status: not used (WQI/Status were computed by the dataset author from the unreliable columns)
Coverage is only ~55 days per file (Jan 2019 - Feb 2020) in one-reading-per-minute bursts.

The dataset's own location is not available through the API; the coordinates below are those of the nearest CPCB
stations (assumption, +/- a few km). A commonly copied value of 78 deg 54' E for the confluence is a typo: Prayagraj
lies near 81.9 deg E.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASET = "dal4206/dataset-of-river-ganga-and-sangam-india"
RAW_DIR = ROOT / "data" / "raw" / "kaggle" / "ganga_sangam"

# name -> (lat, lon, basis of the coordinates)
LOCATIONS = {
    "sangam": (25.4192, 81.9005, "CPCB station 'GANGA AT ALLAHABAD D/S (SANGAM) U.P.'"),
    "ganga": (25.4431, 81.8871, "CPCB station 'GANGA AT KADAGHAT ALLAHABAD' (assumed reach; +/- a few km)"),
}
DO_BOUNDS = (0.0, 14.0)         # mg/L; > 14 is physically impossible for these temperatures
TEMP_BOUNDS = (0.0, 40.0)       # deg C liquid water


def download(dest: Path | str = RAW_DIR) -> Path:
    """Download the dataset with the Kaggle API (needs ~/.kaggle/access_token or KAGGLE_API_TOKEN)."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(DATASET, path=str(dest), unzip=True, quiet=True)
    return dest


def load_iot(name: str, raw_dir: Path | str = RAW_DIR) -> pd.DataFrame:
    """Raw readings with a parsed timestamp and per-column plausibility flags."""
    if name not in LOCATIONS:
        raise ValueError(f"unknown series {name!r}; expected one of {sorted(LOCATIONS)}")
    path = Path(raw_dir) / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run src.data.kaggle_sources.download() (needs a Kaggle token)")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df["do_ok"] = df["DO"].between(*DO_BOUNDS) & (df["DO"] > 0)
    df["temp_ok"] = df["Temp"].between(*TEMP_BOUNDS)
    return df


def reliability_report(df: pd.DataFrame) -> dict:
    """Numbers that justify which columns are used."""
    return {
        "rows": int(len(df)),
        "days_with_data": int(df["Date"].dt.date.nunique()),
        "first": str(df["Date"].min()), "last": str(df["Date"].max()),
        "share_ph_above_8_5": round(float((df["pH"] > 8.5).mean()), 3),
        "median_ph": round(float(df["pH"].median()), 2),
        "share_do_impossible": round(float((df["DO"] > DO_BOUNDS[1]).mean()), 4),
        "conductivity_p5_p95": [round(float(df["Cond"].quantile(q)), 2) for q in (0.05, 0.95)],
        "temp_p5_p95": [round(float(df["Temp"].quantile(q)), 1) for q in (0.05, 0.95)],
    }


def daily_summary(df: pd.DataFrame, hours: tuple[int, int] = (9, 14), min_readings: int = 5) -> pd.DataFrame:
    """One row per day: median temperature and DO of the plausible readings taken between ``hours`` (local clock).

    The window brackets the ~10:30 local overpass time of Landsat / Sentinel-2. Only temperature and DO are
    exported; pH, conductivity and ORP are deliberately dropped (unreliable).
    """
    d = df[(df["Date"].dt.hour >= hours[0]) & (df["Date"].dt.hour < hours[1])].copy()
    d["day"] = d["Date"].dt.normalize()
    rows = []
    for day, g in d.groupby("day"):
        t, o = g.loc[g["temp_ok"], "Temp"], g.loc[g["do_ok"], "DO"]
        rows.append({"date": day,
                     "temp_c": float(t.median()) if len(t) >= min_readings else np.nan, "n_temp": int(len(t)),
                     "do": float(o.median()) if len(o) >= min_readings else np.nan, "n_do": int(len(o))})
    return pd.DataFrame(rows)


def station_frame(name: str, raw_dir: Path | str = RAW_DIR) -> pd.DataFrame:
    """Daily summaries with station id and coordinates, ready for the satellite extractors."""
    lat, lon, _ = LOCATIONS[name]
    out = daily_summary(load_iot(name, raw_dir))
    out.insert(0, "station", f"KAGGLE_{name.upper()}")
    out["lat"], out["lon"] = lat, lon
    out["source"] = f"Kaggle {DATASET} (IoT sensor)"
    out["is_proxy"] = False
    return out
