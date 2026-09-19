"""
Spatial-Temporal Join — Aqua-Sense (Marutey P2)

Joins satellite-derived spectral features to ground-truth station data.

Matching rule (from project plan Section 3):
  - Sentinel-2: ±3 days
  - Landsat:    ±5 days
  - Within 500 m radius of the station lat/lon
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.data.sensors import day_tolerance


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Returns great-circle distance in km between two points."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Tolerance lookup by sensor
# ---------------------------------------------------------------------------

def get_day_tolerance(sensor: str) -> int:
    """+/-3 days for Sentinel-2, +/-5 for Landsat (any common spelling); unknown sensors get the S2 default."""
    return day_tolerance(sensor)


# ---------------------------------------------------------------------------
# Core join
# ---------------------------------------------------------------------------

def spatial_temporal_join(
    features_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    radius_km: float = 0.5,
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Join feature pixels to ground-truth station readings.

    Args:
        features_df    : per-pixel DataFrame with columns
                         [site, lat, lon, date, sensor, B2..B11, ndci, bdm2,
                          bdm3, red_green, nir, turbidity, chl_a, do, mndwi]
        ground_truth_df: station DataFrame with columns
                         [site, lat, lon, date, chl_a, turbidity, do, bod, source]
        radius_km      : spatial matching radius (default 0.5 km)
        agg            : aggregation for multiple matched pixels ("mean" | "median")

    Returns:
        Matched DataFrame with both feature columns and GT labels, plus
        a `match_dist_km` column and `day_diff` column for QA.
        Logs the number of matched pairs per site.
    """
    features_df  = features_df.copy()
    ground_truth_df = ground_truth_df.copy()

    features_df["date"] = pd.to_datetime(features_df["date"])
    ground_truth_df["date"] = pd.to_datetime(ground_truth_df["date"])

    matched_rows = []

    for _, gt_row in ground_truth_df.iterrows():
        gt_site = gt_row["site"]
        gt_date = gt_row["date"]
        gt_lat  = gt_row["lat"]
        gt_lon  = gt_row["lon"]

        # Filter to same site
        site_feats = features_df[features_df["site"] == gt_site]
        if site_feats.empty:
            continue

        # Determine temporal tolerance per sensor
        tolerance = site_feats["sensor"].map(get_day_tolerance).fillna(3).astype(int)
        day_diff = (site_feats["date"] - gt_date).dt.days.abs()
        time_mask = day_diff <= tolerance

        # Spatial filter
        dist_km = pd.Series(
            _haversine_km(gt_lat, gt_lon, site_feats["lat"].to_numpy(dtype=float),
                          site_feats["lon"].to_numpy(dtype=float)),
            index=site_feats.index,
        )
        space_mask = dist_km <= radius_km

        candidates = site_feats[time_mask & space_mask].copy()
        if candidates.empty:
            continue

        candidates["match_dist_km"] = dist_km[time_mask & space_mask]
        candidates["day_diff"]      = day_diff[time_mask & space_mask]

        # Aggregate pixels (mean or median)
        num_cols = candidates.select_dtypes(include=np.number).columns.tolist()
        if agg == "median":
            agg_row = candidates[num_cols].median()
        else:
            agg_row = candidates[num_cols].mean()

        # Overwrite with GT labels
        agg_row["chl_a_gt"]     = gt_row["chl_a"]
        agg_row["turbidity_gt"] = gt_row["turbidity"]
        agg_row["do_gt"]        = gt_row["do"]
        agg_row["bod_gt"]       = gt_row["bod"]
        agg_row["gt_source"]    = gt_row["source"]
        agg_row["site"]         = gt_site
        agg_row["date"]         = gt_date

        matched_rows.append(agg_row)

    if not matched_rows:
        print("WARNING: No matched pairs found. Check site names and date ranges.")
        return pd.DataFrame()

    result = pd.DataFrame(matched_rows).reset_index(drop=True)

    # Log per-site pair counts
    pair_counts = result.groupby("site").size()
    print("\n=== Spatial-Temporal Join Results ===")
    print(pair_counts.to_string())
    print(f"Total matched pairs: {len(result)}")

    # Drop sites with < 3 pairs (per plan)
    valid_sites = pair_counts[pair_counts >= 3].index
    dropped = pair_counts[pair_counts < 3].index.tolist()
    if dropped:
        print(f"\nDropping sites with < 3 pairs: {dropped}")
        result = result[result["site"].isin(valid_sites)]

    return result
