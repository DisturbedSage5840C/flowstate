"""Join CPCB in-situ visits with Sentinel-2 reflectance into a real training table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.feature_engineering import compute_all_features
from src.wqi.wqi_engine import classify_cpcb_best_use, compute_wqi_dataframe

from src.models.schema import REAL_FEATURE_COLS as FEATURE_COLS_REAL  # single source of truth
from src.models.schema import REAL_TARGET_COLS as TARGETS               # real labels; Chl-a has no in-situ source

BANDS = ["B2", "B3", "B4", "B5", "B6", "B8", "B11"]


def quality_filter(refl: pd.DataFrame, min_water_px: int = 20, min_b3: float = 0.005,
                   max_b8: float = 0.20) -> pd.DataFrame:
    """Keep extractions that are trustworthy: status ok, enough water pixels, sane reflectance."""
    r = refl[refl["status"] == "ok"].copy()
    ok = (r["n_water_px"] >= min_water_px) & (r["B3"] >= min_b3) & (r["B8"] <= max_b8)
    ok &= r[BANDS].notna().all(axis=1) & (r[BANDS] > -0.01).all(axis=1)
    return r[ok]


def build_training_table(insitu: pd.DataFrame, refl: pd.DataFrame, min_water_px: int = 20,
                         temperature: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (station, date): real labels + median water reflectance + spectral features.

    ``temperature`` (optional, from src.data.landsat_temp) supplies ``temp_surface`` in deg C where a Landsat
    thermal retrieval exists; every other row keeps NaN (never a constant).
    """
    good = quality_filter(refl, min_water_px)
    good = good.assign(date=pd.to_datetime(good["date"]))
    ins = insitu.assign(date=pd.to_datetime(insitu["date"]))
    keep_ins = ["station", "date", "state", "lat", "lon", "water_body_type", "do", "bod", "turbidity", "ph",
                "conductivity", "total_coliform", "ammonia_n", "sar", "boron", "source", "is_proxy", "retrieved_on"]
    keep_ins = [c for c in keep_ins if c in ins.columns]
    ins = ins[keep_ins].groupby(["station", "date"], as_index=False).first()
    df = ins.merge(good[["station", "date", "scene_id", "scene_date", "day_diff", "n_water_px", "cloud_frac",
                         "scene_cloud", *BANDS]], on=["station", "date"], how="inner")
    if temperature is not None and len(temperature):
        t = temperature[temperature["temp_status"] == "ok"][["station", "date", "temp_c", "temp_day_diff"]].copy()
        t = t[t["temp_c"].between(0.0, 40.0)]                      # liquid-water plausibility (drops hot-land / ice mixtures)
        t["date"] = pd.to_datetime(t["date"])
        t = t.rename(columns={"temp_c": "temp_surface"}).drop_duplicates(["station", "date"])
        df = df.merge(t, on=["station", "date"], how="left")
        df["temp_source"] = np.where(df["temp_surface"].notna(), "Landsat 8/9 L2 lwir11 (median of clear water pixels)", None)
    else:
        df["temp_surface"] = np.nan
        df["temp_source"] = None
    df = df.rename(columns={"station": "site"})
    df["sensor"] = "S2"
    df["date"] = pd.to_datetime(df["date"])
    df = compute_all_features(df, sensor="S2")
    df = df.rename(columns={"chl_a_empirical": "chl_a_empirical"})   # kept under its honest name

    df = compute_wqi_dataframe(df)                      # from measured DO/BOD/pH/turbidity/conductivity
    df = classify_cpcb_best_use(df)                     # from measured criteria only
    return df.sort_values(["site", "date"]).reset_index(drop=True)
