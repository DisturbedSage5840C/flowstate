"""
scripts/generate_synthetic_data.py
====================================
Generates a synthetic train.parquet with the exact schema agreed between
Marutey (P2) and Navya/Jashan (P3/P4).

Navya and Jashan can import and run the pipeline against this file from
Hour 1, before real ground-truth data arrives.

Schema
------
site, water_body_type, lat, lon, date, sensor,
B2, B3, B4, B5, B8, B11,
ndci, bdm2, bdm3, red_green, nir, temp_surface,
chl_a, turbidity, do, wqi

Realistic ranges (references: CPCB NWMP reports, Nechad 2010, NDCI literature)
-------------------------------------------------------------------------------
  chl_a:      5–200  µg/L  (lake eutrophic); 2–40 µg/L (river)
  turbidity:  2–500  FNU   (lake); 5–2000 FNU (monsoon river)
  do:         2–12   mg/L  (low in eutrophic; higher in cold/fast rivers)
  wqi:        0–100  (CPCB weighted arithmetic; A≥90, B 75-90, C 50-75, D 25-50, E<25)

Run
---
    python scripts/generate_synthetic_data.py
Outputs data/processed/train.parquet  (~300 rows, one per site×date×sensor)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

# ── Reproducibility ───────────────────────────────────────────────────────────
RNG = np.random.default_rng(seed=42)

# ── Sites mirroring config/sites.yaml ────────────────────────────────────────
SITES = [
    ("Bellandur",        "lake",      12.925,  77.655, "Karnataka"),
    ("Varthur",          "lake",      12.945,  77.740, "Karnataka"),
    ("Ulsoor",           "lake",      12.990,  77.625, "Karnataka"),
    ("Sutlej_Ludhiana",  "river",     30.925,  75.900, "Punjab"),
    ("Beas_Amritsar",    "river",     31.625,  74.950, "Punjab"),
    ("Buddha_Nullah",    "river",     30.915,  75.860, "Punjab"),
    ("Yamuna_Delhi",     "river",     28.590,  77.260, "Delhi"),
    ("Ganga_Varanasi",   "river",     25.315,  83.015, "Uttar Pradesh"),
    ("Hussain_Sagar",    "lake",      17.445,  78.465, "Telangana"),
    ("Dal_Lake",         "lake",      34.105,  74.855, "Jammu and Kashmir"),
    ("Chilika",          "lagoon",    19.780,  85.325, "Odisha"),
]

SENSORS = ["sentinel2", "landsat"]

# Dates: 2023-01 → 2024-06, roughly bi-monthly per site (simulates sparse coverage)
DATE_RANGE = pd.date_range("2023-01-01", "2024-06-30", freq="15D")


def _band_values(water_body_type: str, chl_a: float, turbidity: float) -> dict:
    """Generate plausible surface reflectance values correlated with WQ params."""
    base_blue   = RNG.uniform(0.04, 0.12)
    base_green  = base_blue + turbidity / 5000 + RNG.normal(0, 0.005)
    base_red    = base_green - chl_a / 4000 + RNG.normal(0, 0.003)
    base_rededge= base_red  + chl_a / 3000  + RNG.normal(0, 0.004)  # B5 key feature
    base_nir    = RNG.uniform(0.01, 0.05) + turbidity / 8000
    base_swir   = RNG.uniform(0.005, 0.03)

    noise = lambda s=0.003: RNG.normal(0, s)
    return {
        "B2":  float(np.clip(base_blue   + noise(), 0, 0.5)),
        "B3":  float(np.clip(base_green  + noise(), 0, 0.5)),
        "B4":  float(np.clip(base_red    + noise(), 0, 0.5)),
        "B5":  float(np.clip(base_rededge + noise(), 0, 0.5)),   # red-edge, Chl-a driver
        "B8":  float(np.clip(base_nir    + noise(), 0, 0.5)),
        "B11": float(np.clip(base_swir   + noise(), 0, 0.3)),
    }


def _derived_features(bands: dict) -> dict:
    """Compute spectral indices from band values."""
    B4, B5, B3, B8 = bands["B4"], bands["B5"], bands["B3"], bands["B8"]
    eps = 1e-9
    ndci  = (B5 - B4) / (B5 + B4 + eps)
    bdm2  = B4 / (B3 + eps)             # 2-band turbidity model
    bdm3  = (B4 - B3) / (B4 + B3 + eps) # 3-band alternative
    return {
        "ndci":      float(ndci),
        "bdm2":      float(bdm2),
        "bdm3":      float(bdm3),
        "red_green": float(B4 / (B3 + eps)),
        "nir":       float(B8),
    }


def _wq_params(water_body_type: str) -> tuple[float, float, float]:
    """Sample realistic chl_a, turbidity, do for the water body type."""
    if water_body_type == "lake":
        chl_a     = float(RNG.lognormal(mean=3.5, sigma=0.8))   # ~33 µg/L median
        chl_a     = np.clip(chl_a, 5, 350)
        turbidity = float(RNG.lognormal(mean=2.5, sigma=0.9))
        turbidity = np.clip(turbidity, 2, 500)
        do        = float(RNG.normal(6.0, 1.5))
    elif water_body_type == "river":
        chl_a     = float(RNG.lognormal(mean=2.5, sigma=0.7))
        chl_a     = np.clip(chl_a, 2, 80)
        turbidity = float(RNG.lognormal(mean=3.5, sigma=1.0))
        turbidity = np.clip(turbidity, 5, 2000)
        do        = float(RNG.normal(7.5, 2.0))
    else:  # lagoon, reservoir
        chl_a     = float(RNG.lognormal(mean=3.0, sigma=0.7))
        chl_a     = np.clip(chl_a, 3, 150)
        turbidity = float(RNG.lognormal(mean=2.8, sigma=0.9))
        turbidity = np.clip(turbidity, 3, 400)
        do        = float(RNG.normal(7.0, 1.8))
    do = float(np.clip(do, 1.0, 14.0))
    return chl_a, turbidity, do


def _wqi(chl_a: float, turbidity: float, do: float) -> float:
    """Simplified CPCB weighted-arithmetic WQI proxy for synthetic labels.

    Real engine lives in src/wqi/wqi_engine.py (Marutey's deliverable).
    This is for label generation only.
    """
    # DO sub-index: ideal 7 mg/L; penalise deviation
    do_si      = max(0.0, min(100.0, 100 * (do / 9.0)))
    # Turbidity sub-index: CPCB permissible = 5 NTU; 500+ → 0
    turb_si    = max(0.0, min(100.0, 100 * (1 - turbidity / 500)))
    # Chl-a sub-index: permissible ~20 µg/L; 200+ → 0
    chla_si    = max(0.0, min(100.0, 100 * (1 - chl_a / 200)))
    wqi = (0.35 * do_si + 0.35 * turb_si + 0.30 * chla_si)
    return float(np.clip(wqi, 0, 100))


def generate(n_per_site: int = 30) -> pd.DataFrame:
    """Generate synthetic training data."""
    rows = []
    for site, wbt, lat, lon, state in SITES:
        # Sample n_per_site dates per site
        dates = RNG.choice(DATE_RANGE, size=n_per_site, replace=False)
        for date in dates:
            sensor = RNG.choice(SENSORS)
            chl_a, turbidity, do = _wq_params(wbt)
            bands = _band_values(wbt, chl_a, turbidity)
            derived = _derived_features(bands)
            temp_surface = float(RNG.normal(28.0, 5.0) if "river" in wbt else RNG.normal(26.0, 4.0))
            wqi = _wqi(chl_a, turbidity, do)

            # Add small spatial jitter so each obs has a slightly different lat/lon
            lat_jit = lat + RNG.uniform(-0.01, 0.01)
            lon_jit = lon + RNG.uniform(-0.01, 0.01)

            row = {
                "site":             site,
                "water_body_type":  wbt,
                "lat":              round(float(lat_jit), 5),
                "lon":              round(float(lon_jit), 5),
                "date":             pd.Timestamp(date),
                "sensor":           sensor,
                **{k: round(v, 6) for k, v in bands.items()},
                **{k: round(v, 6) for k, v in derived.items()},
                "temp_surface":     round(temp_surface, 2),
                "chl_a":            round(chl_a, 3),
                "turbidity":        round(turbidity, 3),
                "do":               round(do, 3),
                "wqi":              round(wqi, 2),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    # Enforce column order matching the agreed schema
    ordered_cols = [
        "site", "water_body_type", "lat", "lon", "date", "sensor",
        "B2", "B3", "B4", "B5", "B8", "B11",
        "ndci", "bdm2", "bdm3", "red_green", "nir", "temp_surface",
        "chl_a", "turbidity", "do", "wqi",
    ]
    df = df[ordered_cols].sort_values(["site", "date"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate(n_per_site=30)
    out_path = out_dir / "train.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n[OK] Synthetic train.parquet written: {out_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Sites: {df['site'].nunique()}")
    print(f"\nSample statistics:")
    print(df[["chl_a", "turbidity", "do", "wqi"]].describe().round(2).to_string())
