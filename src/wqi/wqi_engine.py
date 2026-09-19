"""
Water-quality index engine.

Two independent outputs, deliberately kept separate:

1. ``wqi``  - a *satellite-adapted weighted-arithmetic pollution index*, bounded 0-100
   (0 = pristine, 100 = worst). Aqua-Sense's own convention, NOT an official CPCB
   product.  Formula (Brown-type weighted arithmetic mean):

       Qi  = 100 * (Vi - V0) / (Si - V0)      clipped to [0, 100]
             (pH:  Qi = 100 * |Vi - 7| / (Si - 7))
       Wi  = K / Si,   K = 1 / sum(1/Si)      (weights renormalised over the
                                                parameters actually available)
       WQI = sum(Wi * Qi)

   Only parameters that are present (not NaN) contribute; nothing is imputed.
   Deviation from the textbook formula: the ideal DO value V0 is the saturation
   concentration at the measured water temperature (default 25 C) instead of
   14.6 mg/L (saturation at 0 C), which no Indian surface water can reach.
   Chlorophyll-a is not a CPCB parameter; it is included only as an eutrophication
   indicator with a reference value of 10 ug/L.

2. ``cpcb_class`` - the CPCB *designated-best-use* class A-E, obtained from the
   concentration criteria published by CPCB
   (https://cpcb.gov.in/water-quality-criteria/), not from the WQI score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.wqi.water_chemistry import DEFAULT_TEMP_C, do_saturation_mg_l, free_ammonia_n


# ---------------------------------------------------------------------------
# WQI parameter table
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WQIParameter:
    name: str
    unit: str
    ideal: float          # V0
    standard: float       # Si
    basis: str            # where Si comes from
    two_sided: bool = False   # True for pH: deviation from ideal in either direction


PARAMETERS: dict[str, WQIParameter] = {
    "do":  WQIParameter("Dissolved Oxygen", "mg/L", DEFAULT_TEMP_C, 5.0,
                        "CPCB Class B minimum (ideal = saturation at water temperature)"),
    "bod": WQIParameter("BOD (3 day, 27 C / 5 day, 20 C)", "mg/L", 0.0, 3.0,
                        "CPCB Class B/C maximum"),
    "ph":  WQIParameter("pH", "-", 7.0, 8.5, "CPCB Class A/B upper limit", two_sided=True),
    "turbidity": WQIParameter("Turbidity", "NTU", 0.0, 5.0,
                              "IS 10500 permissible limit (verify against current BIS text)"),
    "chl_a": WQIParameter("Chlorophyll-a", "ug/L", 0.0, 10.0,
                          "eutrophication reference value, not a CPCB standard"),
    "total_coliform": WQIParameter("Total coliform", "MPN/100 mL", 0.0, 500.0,
                                   "CPCB Class B maximum"),
    "conductivity": WQIParameter("Electrical conductivity", "uS/cm", 0.0, 2250.0,
                                 "CPCB Class E maximum"),
}

_TEMP_COLUMNS = ("temp_c", "temperature", "temp_surface")


# ---------------------------------------------------------------------------
# WQI tiers (single definition used by the whole repo)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WQITier:
    label: str
    lower: float
    upper: float      # exclusive, except the last tier which includes 100
    color: str


WQI_TIERS: list[WQITier] = [
    WQITier("Excellent", 0.0, 20.0, "#2166ac"),
    WQITier("Good", 20.0, 40.0, "#4dac26"),
    WQITier("Moderate", 40.0, 60.0, "#f7c000"),
    WQITier("Poor", 60.0, 80.0, "#f46d43"),
    WQITier("Very Poor", 80.0, 100.0, "#d73027"),
]


def wqi_tier(score: float) -> Optional[WQITier]:
    """Tier for a 0-100 score; None for NaN."""
    if score is None or not np.isfinite(score):
        return None
    for tier in WQI_TIERS[:-1]:
        if score < tier.upper:
            return tier
    return WQI_TIERS[-1]


def wqi_tier_label(score: float) -> Optional[str]:
    tier = wqi_tier(score)
    return tier.label if tier else None


# ---------------------------------------------------------------------------
# Quality rating and index
# ---------------------------------------------------------------------------

def _quality_rating(key: str, values: np.ndarray, temp_c: np.ndarray) -> np.ndarray:
    """Vectorised Qi in [0, 100]; NaN where the value is NaN."""
    p = PARAMETERS[key]
    v = np.asarray(values, dtype=float)
    if p.two_sided:
        qi = 100.0 * np.abs(v - p.ideal) / (p.standard - p.ideal)
    elif key == "do":
        ideal = do_saturation_mg_l(temp_c)
        qi = 100.0 * (v - ideal) / (p.standard - ideal)
    else:
        qi = 100.0 * (v - p.ideal) / (p.standard - p.ideal)
    return np.clip(qi, 0.0, 100.0)


def _unit_weights(keys: list[str]) -> dict[str, float]:
    inv = {k: 1.0 / PARAMETERS[k].standard for k in keys}
    total = sum(inv.values())
    return {k: v / total for k, v in inv.items()}


def compute_wqi_dataframe(df: pd.DataFrame, min_params: int = 2) -> pd.DataFrame:
    """Vectorised WQI over a frame. Uses whichever WQI parameters are present.

    Adds ``wqi`` (0-100, NaN if fewer than ``min_params`` parameters are available),
    ``wqi_tier``, ``n_wqi_params`` and one ``qi_<param>`` column per parameter
    (NaN where the parameter is missing for that row).
    """
    out = df.copy()
    n = len(out)
    temp = np.full(n, DEFAULT_TEMP_C)
    for col in _TEMP_COLUMNS:
        if col in out.columns:
            t = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
            temp = np.where(np.isfinite(t), t, temp)
            break

    present = [k for k in PARAMETERS if k in out.columns]
    qi = {k: _quality_rating(k, pd.to_numeric(out[k], errors="coerce").to_numpy(dtype=float), temp)
          for k in present}

    inv_w = np.array([1.0 / PARAMETERS[k].standard for k in present]) if present else np.array([])
    q_mat = np.column_stack([qi[k] for k in present]) if present else np.empty((n, 0))
    avail = np.isfinite(q_mat)
    w_mat = np.where(avail, inv_w[None, :], 0.0)
    w_sum = w_mat.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.where(w_sum > 0, (w_mat * np.where(avail, q_mat, 0.0)).sum(axis=1) / w_sum, np.nan)
    n_params = avail.sum(axis=1)
    score = np.where(n_params >= min_params, score, np.nan)

    out["wqi"] = score
    out["n_wqi_params"] = n_params
    out["wqi_tier"] = [wqi_tier_label(s) for s in score]
    for k in PARAMETERS:
        out[f"qi_{k}"] = qi[k] if k in qi else np.nan
    return out


def compute_wqi(*, do=None, bod=None, ph=None, turbidity=None, chl_a=None,
                total_coliform=None, conductivity=None, temp_c=None, min_params: int = 2) -> dict:
    """WQI for one sample. Pass only the parameters you actually have."""
    row = {"do": do, "bod": bod, "ph": ph, "turbidity": turbidity, "chl_a": chl_a,
           "total_coliform": total_coliform, "conductivity": conductivity}
    row = {k: (np.nan if v is None else float(v)) for k, v in row.items()}
    if temp_c is not None:
        row["temp_c"] = float(temp_c)
    res = compute_wqi_dataframe(pd.DataFrame([row]), min_params=min_params).iloc[0]
    used = [k for k in PARAMETERS if np.isfinite(res[f"qi_{k}"])]
    tier = wqi_tier(res["wqi"])
    return {
        "wqi": float(res["wqi"]),
        "wqi_tier": tier.label if tier else None,
        "n_params": int(res["n_wqi_params"]),
        "sub_index": {k: float(res[f"qi_{k}"]) for k in used},
        "weights": _unit_weights(used) if used else {},
    }


# ---------------------------------------------------------------------------
# CPCB designated-best-use class (A-E) from concentration criteria
# ---------------------------------------------------------------------------
# Source: CPCB "Water Quality Criteria" (designated best use), https://cpcb.gov.in/water-quality-criteria/
# ("min" = value must be >= limit, "max" = <= limit, "range" = within [lo, hi])

CPCB_CLASSES: list[tuple[str, str, dict]] = [
    ("A", "Drinking water source without conventional treatment but after disinfection",
     {"total_coliform": ("max", 50), "ph": ("range", 6.5, 8.5), "do": ("min", 6.0), "bod": ("max", 2.0)}),
    ("B", "Outdoor bathing (organised)",
     {"total_coliform": ("max", 500), "ph": ("range", 6.5, 8.5), "do": ("min", 5.0), "bod": ("max", 3.0)}),
    ("C", "Drinking water source after conventional treatment and disinfection",
     {"total_coliform": ("max", 5000), "ph": ("range", 6.0, 9.0), "do": ("min", 4.0), "bod": ("max", 3.0)}),
    ("D", "Propagation of wildlife and fisheries",
     {"ph": ("range", 6.5, 8.5), "do": ("min", 4.0), "free_ammonia": ("max", 1.2)}),
    ("E", "Irrigation, industrial cooling, controlled waste disposal",
     {"ph": ("range", 6.0, 8.5), "conductivity": ("max", 2250.0), "sar": ("max", 26.0), "boron": ("max", 2.0)}),
]

CPCB_CLASS_DESCRIPTIONS = {c: d for c, d, _ in CPCB_CLASSES}
CPCB_BELOW_E = "Below E"


def _criterion_pass(kind: tuple, values: np.ndarray) -> np.ndarray:
    if kind[0] == "min":
        return values >= kind[1]
    if kind[0] == "max":
        return values <= kind[1]
    return (values >= kind[1]) & (values <= kind[2])


def classify_cpcb_best_use(df: pd.DataFrame, min_criteria: int = 2) -> pd.DataFrame:
    """Add ``cpcb_class`` (A-E / 'Below E' / None), ``cpcb_criteria_assessed`` and
    ``cpcb_class_complete`` (True when every criterion of the assigned class was measured).

    A sample gets the best class (A first) for which every *available* criterion is met and
    at least ``min_criteria`` criteria could be evaluated.  ``free_ammonia`` is derived from
    ``ammonia_n``/``ph``/temperature when only total ammonia-N is present.
    """
    out = df.copy()
    n = len(out)

    def col(name: str) -> np.ndarray:
        if name in out.columns:
            return pd.to_numeric(out[name], errors="coerce").to_numpy(dtype=float)
        return np.full(n, np.nan)

    values = {k: col(k) for k in ("total_coliform", "ph", "do", "bod", "conductivity", "sar", "boron")}
    if "free_ammonia" in out.columns:
        values["free_ammonia"] = col("free_ammonia")
    else:
        temp = np.full(n, DEFAULT_TEMP_C)
        for tcol in _TEMP_COLUMNS:
            if tcol in out.columns:
                t = pd.to_numeric(out[tcol], errors="coerce").to_numpy(dtype=float)
                temp = np.where(np.isfinite(t), t, temp)
                break
        values["free_ammonia"] = free_ammonia_n(col("ammonia_n"), values["ph"], temp)

    cls = np.full(n, None, dtype=object)
    assessed_out = np.zeros(n, dtype=int)
    complete_out = np.zeros(n, dtype=bool)
    unassigned = np.ones(n, dtype=bool)
    best_any_assessed = np.zeros(n, dtype=int)

    for letter, _desc, criteria in CPCB_CLASSES:
        ok = np.ones(n, dtype=bool)
        assessed = np.zeros(n, dtype=int)
        for key, kind in criteria.items():
            v = values[key]
            have = np.isfinite(v)
            assessed += have
            ok &= np.where(have, _criterion_pass(kind, v), True)
        best_any_assessed = np.maximum(best_any_assessed, assessed)
        hit = unassigned & ok & (assessed >= min_criteria)
        cls[hit] = letter
        assessed_out[hit] = assessed[hit]
        complete_out[hit] = assessed[hit] == len(criteria)
        unassigned &= ~hit

    below = unassigned & (best_any_assessed >= min_criteria)
    cls[below] = CPCB_BELOW_E
    assessed_out[below] = best_any_assessed[below]

    out["cpcb_class"] = cls
    out["cpcb_criteria_assessed"] = assessed_out
    out["cpcb_class_complete"] = complete_out
    return out
