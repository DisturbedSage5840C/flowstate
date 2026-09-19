"""
Feature engineering: spectral indices and empirical water-quality proxies.

Everything named ``*_empirical`` is a *formula-derived estimate*, never ground truth.
The constants below come from the remote-sensing literature and, where noted, could
not be independently verified against the original papers; ``scripts/validate_empirical_formulas.py``
checks them against real CPCB turbidity/DO data instead of trusting them.

Sensor conventions (band names as exported by ``src.acquisition.gee``):
  S2 : B2 blue 490, B3 green 560, B4 red 665, B5 red-edge 705, B6 red-edge 740,
       B8 NIR 842, B11 SWIR 1610
  LS : Landsat 8/9 exported as B3 green, B4 red, B5 NIR, B6 SWIR1.
       NOTE Landsat B5 is NIR (865 nm), NOT red-edge. Landsat has no red-edge band, so
       NDCI / Chl-a cannot be derived from it; only turbidity and MNDWI can.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.wqi.water_chemistry import DEFAULT_TEMP_C, do_saturation_mg_l

from src.data.sensors import REQUIRED_FOR_FEATURES  # noqa: E402

S2_REQUIRED = REQUIRED_FOR_FEATURES["S2"]
LS_REQUIRED = REQUIRED_FOR_FEATURES["LS"]

# ---------------------------------------------------------------------------
# Literature constants (see module docstring for verification status)
# ---------------------------------------------------------------------------

# Nechad et al. (2010) single-band algorithm as parameterised for turbidity by
# Dogliotti et al. (2015): T = A * rho / (1 - rho / C)   [FNU].
# UNVERIFIED: values are the ones commonly reproduced for the 645 nm and 859 nm bands;
# the original tables could not be retrieved in this project. Validate on data.
NECHAD_RED = {"A": 228.1, "C": 0.1641}
NECHAD_NIR = {"A": 3078.9, "C": 0.2112}
DOGLIOTTI_BLEND = (0.05, 0.07)  # red reflectance range over which red->NIR blending happens

# Mishra & Mishra (2012) quadratic NDCI -> Chl-a (ug/L). UNVERIFIED coefficients;
# calibrated on simulated data for turbid productive waters (Chl-a 1-60 ug/L),
# NOT calibrated for Indian inland waters. Valid for NDCI in [-0.1, 0.5].
MISHRA_NDCI_COEFFS = (14.039, 86.115, 194.325)
NDCI_VALID_RANGE = (-0.1, 0.5)

# Locally recalibrated turbidity power law: T = a * B4^b [NTU], fit by log-log regression of
# real CPCB NWDP turbidity against Sentinel-2 B4 (red) reflectance (data/processed/train_real_large.parquet,
# n=3223, see scripts/calibrate_empirical_formulas.py). Unlike NECHAD_RED/NECHAD_NIR above, which are
# unmodified literature constants for a different sensor/region, these coefficients are fit directly on the
# real ground truth this project measures against, and fix the ~10x median overestimate documented in
# reports/real/empirical_formula_validation.json. Fit per water-body type where the type has an optically
# distinct sediment/turbidity relationship (rivers carry more suspended sediment per unit reflectance than
# still water); "unknown"/unrecognised types and Landsat (no B4-only fit was done for LS) fall back to GLOBAL.
TURBIDITY_CALIBRATION_GLOBAL = (56.49, 0.68)
TURBIDITY_CALIBRATION_BY_TYPE = {
    "river": (107.49, 0.89),
    "lake": (32.15, 0.42),
    "reservoir": (21.17, 0.47),
    "unknown": (40.12, 0.66),
}


def _safe_divide(num, den, fill: float = np.nan) -> np.ndarray:
    """Element-wise division, filling zeros in denominator with `fill`."""
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(den != 0, num / den, fill)
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Spectral indices
# ---------------------------------------------------------------------------

def compute_ndci(b5, b4) -> np.ndarray:
    """NDCI = (B5 - B4) / (B5 + B4)  (Mishra & Mishra 2012; S2 705 nm vs 665 nm)."""
    b5 = np.asarray(b5, dtype=np.float64)
    b4 = np.asarray(b4, dtype=np.float64)
    return _safe_divide(b5 - b4, b5 + b4)


def compute_2bdm(b5, b4) -> np.ndarray:
    """Two-band model, as defined in the project plan: 2BDM = R(705) / R(665)."""
    return _safe_divide(b5, b4)


def compute_3bdm(b4, b5, b6) -> np.ndarray:
    """Three-band model: 3BDM = (1/R665 - 1/R705) * R740.

    ``b6`` must be Sentinel-2 B6 (740 nm). It is never substituted by another band;
    if it is unavailable pass ``None`` and the result is NaN.
    """
    b4 = np.asarray(b4, dtype=np.float64)
    if b6 is None:
        return np.full(b4.shape, np.nan, dtype=np.float32)
    inv = _safe_divide(1.0, b4).astype(np.float64) - _safe_divide(1.0, b5).astype(np.float64)
    return (inv * np.asarray(b6, dtype=np.float64)).astype(np.float32)


def compute_red_green_ratio(b4, b3) -> np.ndarray:
    """Red/green reflectance ratio (turbidity indicator)."""
    return _safe_divide(b4, b3)


def compute_mndwi(b3, b11) -> np.ndarray:
    """MNDWI = (Green - SWIR1) / (Green + SWIR1). Water threshold is scene-dependent."""
    b3 = np.asarray(b3, dtype=np.float64)
    b11 = np.asarray(b11, dtype=np.float64)
    return _safe_divide(b3 - b11, b3 + b11)


# ---------------------------------------------------------------------------
# Empirical proxies
# ---------------------------------------------------------------------------

def compute_nechad_turbidity(rho, A_T: float, C_T: float) -> np.ndarray:
    """Single-band Nechad/Dogliotti turbidity (FNU): T = A_T * rho / (1 - rho / C_T).

    NaN where rho >= C_T (the model saturates and has no valid inverse).
    """
    rho = np.asarray(rho, dtype=np.float64)
    denom = 1.0 - rho / C_T
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where((denom > 0) & (rho >= 0), A_T * rho / denom, np.nan)
    return t.astype(np.float32)


def compute_turbidity_dogliotti(red, nir) -> np.ndarray:
    """Dogliotti et al. (2015) red->NIR blended turbidity.

    Weight w rises linearly from 0 to 1 as red reflectance goes from 0.05 to 0.07;
    T = (1 - w) * T_red + w * T_nir. Uses the module's (unverified) coefficient sets.
    """
    red = np.asarray(red, dtype=np.float64)
    t_red = compute_nechad_turbidity(red, NECHAD_RED["A"], NECHAD_RED["C"]).astype(np.float64)
    t_nir = compute_nechad_turbidity(nir, NECHAD_NIR["A"], NECHAD_NIR["C"]).astype(np.float64)
    lo, hi = DOGLIOTTI_BLEND
    w = np.clip((red - lo) / (hi - lo), 0.0, 1.0)
    blended = (1.0 - w) * t_red + w * t_nir
    # if one branch is undefined fall back to the other
    blended = np.where(np.isnan(t_nir) & (w > 0), t_red, blended)
    blended = np.where(np.isnan(t_red) & (w < 1), t_nir, blended)
    return blended.astype(np.float32)


def compute_turbidity_calibrated(red, water_body_type=None) -> np.ndarray:
    """Locally recalibrated turbidity (NTU): T = a * red^b, fit on real CPCB data.

    Unlike ``compute_turbidity_dogliotti`` (unmodified literature constants, ~3.5-14x
    overestimate bias vs measured CPCB turbidity), this is refit directly on real ground
    truth. If ``water_body_type`` is given (array-like, aligned with ``red``), each row uses
    its type's coefficients from ``TURBIDITY_CALIBRATION_BY_TYPE`` (falling back to the
    global fit for types not in that table); otherwise the global fit is used for all rows.
    Negative/zero reflectance yields NaN (the power law is undefined there).
    """
    red = np.asarray(red, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        safe_red = np.where(red > 0, red, np.nan)

    if water_body_type is None:
        a, b = TURBIDITY_CALIBRATION_GLOBAL
        return (a * np.power(safe_red, b)).astype(np.float32)

    wbt = np.asarray(water_body_type, dtype=object)
    a_glob, b_glob = TURBIDITY_CALIBRATION_GLOBAL
    a = np.full(wbt.shape, a_glob, dtype=np.float64)
    b = np.full(wbt.shape, b_glob, dtype=np.float64)
    for t, (a_t, b_t) in TURBIDITY_CALIBRATION_BY_TYPE.items():
        mask = wbt == t
        a[mask] = a_t
        b[mask] = b_t
    return (a * np.power(safe_red, b)).astype(np.float32)


def compute_chl_a_from_ndci(ndci) -> np.ndarray:
    """Chl-a (ug/L) from NDCI: c0 + c1*NDCI + c2*NDCI^2 (Mishra & Mishra 2012 form).

    Empirical, calibrated elsewhere; NDCI is clipped to the model's validity range.
    """
    c0, c1, c2 = MISHRA_NDCI_COEFFS
    x = np.clip(np.asarray(ndci, dtype=np.float64), *NDCI_VALID_RANGE)
    return (c0 + c1 * x + c2 * x * x).astype(np.float32)


def compute_do_surrogate(chl_a, turbidity, month, t_water_c=DEFAULT_TEMP_C, altitude_m=0.0) -> np.ndarray:
    """Heuristic dissolved-oxygen estimate (mg/L). DO has no direct optical signature.

        DO = DO_sat(T, altitude) - 0.015*max(Chl-a - 20, 0) - 0.005*max(Turb - 50, 0) + monsoon

    ``DO_sat`` is the Benson-Krause freshwater saturation. The depression and monsoon
    terms (+0.5 mg/L for Jun-Sep re-aeration) are heuristics, not fitted coefficients.
    Result is clipped to [0, 1.3 * DO_sat] (blooms can supersaturate).
    """
    chl_a = np.asarray(chl_a, dtype=np.float64)
    turbidity = np.asarray(turbidity, dtype=np.float64)
    month = np.asarray(month)
    do_sat = do_saturation_mg_l(t_water_c, altitude_m)

    chl_dep = np.where(chl_a > 20, 0.015 * (chl_a - 20), 0.0)
    turb_dep = np.where(turbidity > 50, 0.005 * (turbidity - 50), 0.0)
    monsoon = np.where((month >= 6) & (month <= 9), 0.5, 0.0)

    do = do_sat - chl_dep - turb_dep + monsoon
    return np.clip(do, 0.0, 1.3 * do_sat).astype(np.float32)


# ---------------------------------------------------------------------------
# High-level: all features from a per-pixel / per-sample band frame
# ---------------------------------------------------------------------------

def _require(df: pd.DataFrame, bands: tuple[str, ...], sensor: str) -> None:
    missing = [b for b in bands if b not in df.columns]
    if missing:
        raise KeyError(
            f"sensor {sensor!r} needs bands {list(bands)}; missing {missing}. "
            "River rasters must keep B5/B6/B11 (resampled to 10 m) — see src.acquisition.gee.OUT_BANDS."
        )


def compute_all_features(df: pd.DataFrame, sensor: str = "S2") -> pd.DataFrame:
    """Append spectral indices and ``*_empirical`` proxies to a band DataFrame.

    sensor='S2'  -> ndci, bdm2, bdm3, red_green, mndwi, nir, turbidity_empirical,
                    chl_a_empirical, do_empirical
    sensor='LS'  -> red_green, mndwi, nir, turbidity_empirical  (no red-edge -> no NDCI/Chl-a)

    Optional columns: ``B6`` (S2, needed for bdm3), ``temp_surface`` (deg C, water
    temperature for the DO surrogate; default 25), ``altitude_m``, ``date`` (month).
    Raises ``KeyError`` naming the missing bands instead of silently substituting others.
    """
    sensor = sensor.upper()
    if sensor not in ("S2", "LS"):
        raise ValueError("sensor must be 'S2' or 'LS'")
    df = df.copy()

    if sensor == "LS":
        _require(df, LS_REQUIRED, sensor)
        b3, b4, nir, swir = (df[c].to_numpy(dtype=np.float64) for c in ("B3", "B4", "B5", "B6"))
        df["red_green"] = compute_red_green_ratio(b4, b3)
        df["mndwi"] = compute_mndwi(b3, swir)
        df["nir"] = nir.astype(np.float32)
        df["turbidity_empirical"] = compute_turbidity_dogliotti(b4, nir)
        # Global fit only: TURBIDITY_CALIBRATION was fit on Sentinel-2 B4, not Landsat's
        # differently-centred red band, so per-type coefficients are not applied here.
        df["turbidity_calibrated"] = compute_turbidity_calibrated(b4)
        return df

    _require(df, S2_REQUIRED, sensor)
    b3, b4, b5, b8, b11 = (df[c].to_numpy(dtype=np.float64) for c in S2_REQUIRED)
    b6 = df["B6"].to_numpy(dtype=np.float64) if "B6" in df.columns else None

    df["ndci"] = compute_ndci(b5, b4)
    df["bdm2"] = compute_2bdm(b5, b4)
    df["bdm3"] = compute_3bdm(b4, b5, b6)
    df["red_green"] = compute_red_green_ratio(b4, b3)
    df["mndwi"] = compute_mndwi(b3, b11)
    df["nir"] = b8.astype(np.float32)
    df["turbidity_empirical"] = compute_turbidity_dogliotti(b4, b8)
    wbt = df["water_body_type"].to_numpy() if "water_body_type" in df.columns else None
    df["turbidity_calibrated"] = compute_turbidity_calibrated(b4, water_body_type=wbt)
    df["chl_a_empirical"] = compute_chl_a_from_ndci(df["ndci"].to_numpy())

    if "date" in df.columns:
        month = pd.to_datetime(df["date"], errors="coerce").dt.month.fillna(6).to_numpy()
    else:
        month = np.full(len(df), 6)
    temp = (pd.to_numeric(df["temp_surface"], errors="coerce").fillna(DEFAULT_TEMP_C).to_numpy()
            if "temp_surface" in df.columns else DEFAULT_TEMP_C)
    alt = (pd.to_numeric(df["altitude_m"], errors="coerce").fillna(0.0).to_numpy()
           if "altitude_m" in df.columns else 0.0)
    df["do_empirical"] = compute_do_surrogate(
        df["chl_a_empirical"].to_numpy(), df["turbidity_empirical"].to_numpy(),
        month=month, t_water_c=temp, altitude_m=alt,
    )
    return df
