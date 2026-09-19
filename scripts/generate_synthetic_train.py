"""
Synthetic Training Data Generator — Aqua-Sense (Marutey P2)
Hour-2 Deliverable

Generates a synthetic `data/processed/train.parquet` with the exact schema
agreed in the project plan so Navya and Jashan can start building models
before real rasters land.

Schema (from plan Section 4, Handoff 2):
    site, water_body_type, lat, lon, date, sensor,
    B2, B3, B4, B5, B8, B11,
    ndci, bdm2, bdm3, red_green, nir,
    temp_surface,
    chl_a, turbidity, do, wqi

Note: SYNTHETIC demo data — physically plausible but NOT real measurements.
Real in-situ training data comes from scripts/build_real_training_table.py.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make sure we can import from src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.feature_engineering import (
    compute_ndci,
    compute_2bdm,
    compute_3bdm,
    compute_red_green_ratio,
)
from src.wqi.wqi_engine import compute_wqi_dataframe

# ---------------------------------------------------------------------------
# Site definitions (matches config/sites.yaml)
# ---------------------------------------------------------------------------

SITES = [
    # (name, type, state, lat_centre, lon_centre, chl_range, turb_range, do_range, bod_range)
    ("bellandur",             "lake",      "Karnataka",      12.930, 77.665, (60, 180),  (30, 100),  (1.0, 4.0),  (12, 40)),
    ("varthur",               "lake",      "Karnataka",      12.952, 77.750, (70, 200),  (35, 110),  (0.8, 3.5),  (15, 45)),
    ("ulsoor",                "lake",      "Karnataka",      12.987, 77.630, (20, 80),   (10, 40),   (4.0, 7.0),  (5,  18)),
    ("yamuna_delhi",          "river",     "Delhi",          28.610, 77.260, (15, 40),   (150, 450), (0.5, 3.0),  (30, 80)),
    ("ganga_kanpur",          "river",     "Uttar Pradesh",  26.450, 80.325, (10, 30),   (100, 500), (2.0, 5.5),  (15, 60)),
    ("ganga_varanasi",        "river",     "Uttar Pradesh",  25.315, 83.025, (8,  28),   (80,  400), (2.5, 6.0),  (12, 55)),
    ("sutlej_ludhiana",       "river",     "Punjab",         30.875, 75.800, (8,  25),   (80,  320), (1.5, 4.5),  (20, 70)),
    ("buddha_nullah_ludhiana","river",     "Punjab",         30.905, 75.860, (4,  12),   (300, 600), (0.2, 1.5),  (80,180)),
    ("beas_amritsar",         "river",     "Punjab",         31.600, 75.150, (6,  20),   (40,  180), (5.0, 8.0),  (5,  25)),
    ("ghaggar_patiala",       "river",     "Punjab",         30.325, 76.350, (8,  22),   (80,  250), (2.5, 5.5),  (15, 45)),
    ("hussain_sagar",         "lake",      "Telangana",      17.440, 78.460, (30, 120),  (15, 55),   (3.0, 6.5),  (8,  28)),
    ("dal_lake",              "lake",      "Jammu and Kashmir", 34.105, 74.865, (20, 90), (8, 35),   (5.0, 8.5),  (4,  18)),
    ("chilika",               "lagoon",    "Odisha",         19.875, 85.275, (10, 45),   (20, 90),   (4.0, 7.5),  (5,  22)),
]

SENSORS = ["S2", "L8", "L9"]
N_PER_SITE = 30  # synthetic samples per site
SEED = 42


def simulate_bands_for_site(
    chl_range: tuple,
    turb_range: tuple,
    n: int,
    rng: np.random.Generator,
) -> dict:
    """
    Generate physically plausible Sentinel-2 surface reflectance values
    conditioned on Chl-a and turbidity ranges for a given site type.

    Returns dict of band arrays (B2–B11, float32, range 0–0.3) plus the latent
    true Chl-a / turbidity used to draw them.
    """
    chl_vals  = rng.uniform(*chl_range, size=n)
    turb_vals = rng.uniform(*turb_range, size=n)

    # Base reflectance (oligotrophic clean water ~0.02)
    base = 0.02

    # Blue (B2) decreases with turbidity, increases slightly with Chl-a absorption
    B2 = np.clip(base + 0.001 * turb_vals / 50 - 0.0002 * chl_vals + rng.normal(0, 0.002, n), 0.005, 0.25)

    # Green (B3) increases with turbidity and moderate Chl-a
    B3 = np.clip(base + 0.002 * turb_vals / 50 + 0.0003 * chl_vals + rng.normal(0, 0.002, n), 0.01, 0.25)

    # Red (B4) dominated by turbidity
    B4 = np.clip(0.005 + 0.004 * turb_vals / 50 + rng.normal(0, 0.002, n), 0.005, 0.30)

    # Red-edge (B5) responds to Chl-a fluorescence
    B5 = np.clip(0.01 + 0.0008 * chl_vals + 0.001 * turb_vals / 50 + rng.normal(0, 0.002, n), 0.01, 0.20)

    # NIR (B8) dominated by turbidity in water
    B8 = np.clip(0.003 + 0.006 * turb_vals / 100 + rng.normal(0, 0.001, n), 0.002, 0.25)

    # Red-edge 2 (B6, 740 nm) sits slightly below B5 for productive water
    B6 = np.clip(0.008 + 0.0006 * chl_vals + 0.0008 * turb_vals / 50 + rng.normal(0, 0.002, n), 0.005, 0.20)

    # SWIR (B11) — used for MNDWI; very low over water
    B11 = np.clip(0.002 + rng.normal(0, 0.001, n), 0.001, 0.05)

    return {"B2": B2, "B3": B3, "B4": B4, "B5": B5, "B6": B6, "B8": B8, "B11": B11,
            "_chl_true": chl_vals, "_turb_true": turb_vals}


def generate_synthetic_train(output_path: Path) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    all_rows = []

    date_range = pd.date_range("2023-01-01", "2024-12-31", freq="15D")

    for (site, wbt, state, lat_c, lon_c,
         chl_range, turb_range, do_range, bod_range) in SITES:

        bands = simulate_bands_for_site(chl_range, turb_range, N_PER_SITE, rng)

        dates   = rng.choice(date_range, size=N_PER_SITE, replace=True)
        sensors = rng.choice(SENSORS, size=N_PER_SITE)
        lats    = rng.normal(lat_c, 0.003, N_PER_SITE)
        lons    = rng.normal(lon_c, 0.003, N_PER_SITE)
        temp_surface = rng.uniform(22.0, 35.0, N_PER_SITE)

        # Spectral indices (features) — computed only from the noisy simulated
        # bands, never from the true target values directly.
        B3, B4, B5, B6, B8, B11 = (bands[k] for k in ("B3", "B4", "B5", "B6", "B8", "B11"))
        ndci      = compute_ndci(B5, B4)
        bdm2      = compute_2bdm(B5, B4)
        bdm3      = compute_3bdm(B4, B5, B6)
        red_green = compute_red_green_ratio(B4, B3)

        # Targets: independently-sampled "true" values plus their own observation
        # noise — NOT recomputed from any feature column. A label that is an exact
        # function of its own inputs lets a model reconstruct it algebraically
        # (label leakage) instead of learning a band -> water-quality relationship.
        chl_a = np.clip(
            bands["_chl_true"] * rng.normal(1.0, 0.10, N_PER_SITE), 0.5, None
        ).astype(np.float32)
        turbidity = np.clip(
            bands["_turb_true"] * rng.normal(1.0, 0.10, N_PER_SITE), 0.5, None
        ).astype(np.float32)
        # DO is drawn from the site's documented range, independent of every feature.
        do = np.clip(
            rng.uniform(*do_range, size=N_PER_SITE) * rng.normal(1.0, 0.08, N_PER_SITE), 0.05, None
        ).astype(np.float32)

        bod = rng.uniform(*bod_range, size=N_PER_SITE)

        rows = pd.DataFrame({
            "site":            site,
            "water_body_type": wbt,
            "state":           state,
            "lat":             lats,
            "lon":             lons,
            "date":            dates,
            "sensor":          sensors,
            "B2":              bands["B2"],
            "B3":              B3,
            "B4":              B4,
            "B5":              B5,
            "B6":              B6,
            "B8":              B8,
            "B11":             B11,
            "ndci":            ndci,
            "bdm2":            bdm2,
            "bdm3":            bdm3,
            "red_green":       red_green,
            "nir":             B8,
            "temp_surface":    temp_surface,
            "chl_a":           chl_a,
            "turbidity":       turbidity,
            "do":              do,
            "bod":             bod,
        })

        all_rows.append(rows)

    df = pd.concat(all_rows, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["site", "date"]).reset_index(drop=True)

    # WQI from the parameters that exist (no pH here, so none is assumed)
    df = compute_wqi_dataframe(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"\n✅ Synthetic train.parquet written → {output_path}")
    print(f"   Rows: {len(df):,}  |  Sites: {df['site'].nunique()}  |  Columns: {len(df.columns)}")
    print(f"\nSchema:\n{df.dtypes.to_string()}")
    print(f"\nSample (first 3 rows):\n{df[['site','date','chl_a','turbidity','do','wqi','wqi_tier']].head(3).to_string()}")

    return df


if __name__ == "__main__":
    output = ROOT / "data" / "processed" / "train.parquet"
    generate_synthetic_train(output)
