"""
Feature Engineering Module — Aqua-Sense (Marutey P2)

Computes water quality spectral indices from atmospherically-corrected
BOA reflectance rasters:
  - NDCI  : Normalized Difference Chlorophyll Index
  - 2BDM  : Two-Band Difference Model (Chl-a)
  - 3BDM  : Three-Band Difference Model (Chl-a)
  - Red-Green ratio (turbidity proxy)
  - NIR turbidity index
  - NDWI / MNDWI (water mask validation)

Input: corrected raster bands as numpy arrays or a pandas DataFrame of
       per-pixel reflectance values.
Output: DataFrame with all indices appended.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Band helpers
# ---------------------------------------------------------------------------

def _safe_divide(num: np.ndarray, den: np.ndarray, fill: float = np.nan) -> np.ndarray:
    """Element-wise division, filling zeros in denominator with `fill`."""
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(den != 0, num / den, fill)
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Spectral indices
# ---------------------------------------------------------------------------

def compute_ndci(b5: np.ndarray, b4: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Chlorophyll Index (Mishra & Mishra 2012).
    NDCI = (B5 - B4) / (B5 + B4)
    S2 bands: B5=red-edge (705 nm), B4=red (665 nm)
    Range: [-1, 1]; higher = more Chl-a
    """
    return _safe_divide(b5 - b4, b5 + b4)


def compute_2bdm(b5: np.ndarray, b4: np.ndarray) -> np.ndarray:
    """
    Two-Band Difference Model for Chl-a (Dall'Olmo & Gitelson 2005).
    2BDM = (1/B4) - (1/B5)
    Linearises the Chl-a relationship via reflectance inverse.
    """
    return _safe_divide(1.0, b4) - _safe_divide(1.0, b5)


def compute_3bdm(b4: np.ndarray, b5: np.ndarray, b6: np.ndarray) -> np.ndarray:
    """
    Three-Band Difference Model for Chl-a (Gitelson et al. 2008).
    3BDM = ((1/B4) - (1/B5)) * B6
    B6 = red-edge 2 (740 nm) on Sentinel-2.
    Reduces CDOM influence compared to 2BDM.
    """
    return (_safe_divide(1.0, b4) - _safe_divide(1.0, b5)) * b6


def compute_red_green_ratio(b4: np.ndarray, b3: np.ndarray) -> np.ndarray:
    """
    Red-to-Green reflectance ratio — turbidity proxy for low-moderate turbidity.
    Higher ratio = more red sediment backscatter.
    """
    return _safe_divide(b4, b3)


def compute_nir_turbidity_index(b8: np.ndarray) -> np.ndarray:
    """
    NIR-based turbidity index for high-turbidity regimes (FNU > 50).
    Uses Sentinel-2 B8 (842 nm). Nechad et al. (2010) NIR branch.
    Returns a dimensionless proxy; actual FNU computed by wqi_engine.
    """
    return b8.astype(np.float32)


def compute_nechad_turbidity(
    rho: np.ndarray,
    branch: str = "red",
    A_T: float = 228.1,
    B_T: float = 0.1641,
    C_T: float = 0.1728,
) -> np.ndarray:
    """
    Nechad et al. (2010) turbidity retrieval.
    T [FNU] = A_T * rho / (1 - rho/C_T) + B_T

    Args:
        rho   : surface reflectance band (B4 for red branch, B8 for NIR branch)
        branch: "red" or "nir" (for logging; coefficients can differ)
        A_T, B_T, C_T: Nechad coefficients (default: red-band calibration)

    Returns:
        Turbidity in FNU (float32 array)
    """
    denominator = 1.0 - rho / C_T
    with np.errstate(invalid="ignore", divide="ignore"):
        T = np.where(denominator > 0, A_T * rho / denominator + B_T, np.nan)
    return T.astype(np.float32)


def compute_chl_a_from_ndci(ndci: np.ndarray) -> np.ndarray:
    """
    Empirical Chl-a from NDCI (Mishra & Mishra 2012, calibrated for Indian inland waters).
    Chl-a [µg/L] = 10^(1.35 * NDCI + 1.58)
    """
    return np.power(10.0, 1.35 * ndci + 1.58).astype(np.float32)


def compute_mndwi(b3: np.ndarray, b11: np.ndarray) -> np.ndarray:
    """
    Modified Normalized Difference Water Index.
    MNDWI = (B3 - B11) / (B3 + B11)
    Water pixels typically MNDWI > 0 (threshold is scene-dependent; plot histogram).
    """
    return _safe_divide(b3 - b11, b3 + b11)


def compute_do_surrogate(
    chl_a: np.ndarray,
    turbidity: np.ndarray,
    month: int,
    t_water_c: float = 28.0,
) -> np.ndarray:
    """
    DO surrogate model — inferred from Chl-a, turbidity and temperature dynamics.
    DO has no direct spectral signature; this is the surrogate as per project plan.

    Baseline: DO_sat = 14.62 - 0.3898 * T_water (at standard salinity ~0)
    Adjustment: 
      - Chl-a > 50 µg/L (heavy bloom) reduces DO via night respiration
      - Turbidity > 100 FNU reduces DO via light attenuation
      - Monsoon months (Jun-Sep) get +1 mg/L from fresh water influx

    Args:
        chl_a     : Chlorophyll-a [µg/L]
        turbidity : Turbidity [FNU]
        month     : Calendar month (1-12)
        t_water_c : Water surface temperature in °C (from Landsat B10 or climatology)

    Returns:
        Estimated DO [mg/L] clipped to [0, 14]
    """
    do_sat = 14.62 - 0.3898 * t_water_c

    # Chl-a depression: 0.015 mg/L DO per µg/L Chl-a above 20
    chl_depression = np.where(chl_a > 20, 0.015 * (chl_a - 20), 0.0)

    # Turbidity depression: 0.005 mg/L DO per FNU above 50
    turb_depression = np.where(turbidity > 50, 0.005 * (turbidity - 50), 0.0)

    # Monsoon bonus (Jun-Sep)
    monsoon_bonus = 1.0 if 6 <= month <= 9 else 0.0

    do = do_sat - chl_depression - turb_depression + monsoon_bonus
    return np.clip(do, 0.0, 14.0).astype(np.float32)


# ---------------------------------------------------------------------------
# High-level: compute all features from a band DataFrame
# ---------------------------------------------------------------------------

def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns [B2, B3, B4, B5, B8, B11] (S2 surface reflectance)
    and optionally B6 (S2) / B5_L (Landsat), computes all spectral indices.

    Adds columns:
        ndci, bdm2, bdm3, red_green, nir, mndwi,
        turbidity_empirical, chl_a_empirical, do_empirical

    The `_empirical` suffix on turbidity/chl_a/do marks these as formula-derived
    proxies (Nechad turbidity, NDCI-Chl-a regression, DO surrogate) — never
    ground truth. They intentionally do NOT reuse the `turbidity`/`chl_a`/`do`
    names used as model TARGET_COLS elsewhere in the repo: a column that shares
    a target's name but is actually a deterministic function of the model's
    own FEATURE_COLS would let a model reconstruct the label algebraically
    instead of learning a genuine band -> water-quality relationship (see the
    label-leakage bug fixed in scripts/generate_synthetic_train.py).

    Args:
        df: DataFrame with band reflectance columns (float, 0-1 scale)

    Returns:
        df with feature columns appended (in-place copy)
    """
    df = df.copy()

    b3 = df["B3"].values.astype(np.float32)
    b4 = df["B4"].values.astype(np.float32)
    b5 = df["B5"].values.astype(np.float32)
    b8 = df["B8"].values.astype(np.float32)
    b11 = df["B11"].values.astype(np.float32)

    # Use B6 if available (S2 red-edge 2), otherwise use B8 as fallback for 3BDM
    b6 = df["B6"].values.astype(np.float32) if "B6" in df.columns else b8

    df["ndci"]      = compute_ndci(b5, b4)
    df["bdm2"]      = compute_2bdm(b5, b4)
    df["bdm3"]      = compute_3bdm(b4, b5, b6)
    df["red_green"] = compute_red_green_ratio(b4, b3)
    df["mndwi"]     = compute_mndwi(b3, b11)
    df["nir"]       = b8

    # Turbidity: switch branch per pixel based on threshold
    turb_red = compute_nechad_turbidity(b4, branch="red")
    turb_nir = compute_nechad_turbidity(b8, branch="nir",
                                         A_T=1528.0, B_T=0.1641, C_T=0.3742)
    df["turbidity_empirical"] = np.where(turb_red > 50, turb_nir, turb_red)

    df["chl_a_empirical"] = compute_chl_a_from_ndci(df["ndci"].values)

    # DO surrogate — use month from date column if present
    month = 6  # default
    if "date" in df.columns:
        try:
            month = pd.to_datetime(df["date"].iloc[0]).month
        except Exception:
            pass

    df["do_empirical"] = compute_do_surrogate(
        df["chl_a_empirical"].values,
        df["turbidity_empirical"].values,
        month=month,
    )

    return df
