"""
CPCB Weighted-Arithmetic Water Quality Index (WQI) Engine
Aqua-Sense — Marutey (P2)

Implements the CPCB weighted-arithmetic WQI as defined in:
  Central Pollution Control Board, "Guidelines for Water Quality Monitoring"
  CPCB/NWM/Water Quality/2008

Formula:
    WQI = Σ(Wi * Qi) / Σ(Wi)

Where:
    Wi = weight of parameter i (based on relative importance)
    Qi = quality rating of parameter i

Quality rating:
    Qi = 100 * (Vi - Vs) / (Si - Vs)

Where:
    Vi = measured value of parameter i
    Vs = ideal value (pure water standard)
    Si = permissible limit (IS 10500 / CPCB Class C standard)

CPCB Water Quality Classes:
    A : Drinking (with treatment)  — WQI > 90  (Excellent)
    B : Outdoor bathing            — WQI 70-90  (Good)
    C : Drinking (with purification) — WQI 50-70 (Medium)
    D : Propagation of wildlife    — WQI 25-50  (Bad)
    E : Irrigation only            — WQI < 25   (Very Bad / Unsuitable)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# WQI Parameter Table (CPCB / IS 10500 standards)
# ---------------------------------------------------------------------------

@dataclass
class WQIParameter:
    name: str
    unit: str
    ideal_value: float    # Vs — value for pure water
    permissible: float    # Si — CPCB permissible limit
    weight: float         # Wi — assigned weight (higher = more important)
    higher_is_worse: bool = True  # True for pollutants, False for DO


PARAMETERS = {
    "do": WQIParameter(
        name="Dissolved Oxygen",
        unit="mg/L",
        ideal_value=14.6,   # pure water DO at 0°C
        permissible=5.0,    # CPCB minimum acceptable DO
        weight=4.0,
        higher_is_worse=False,  # MORE DO is BETTER
    ),
    "bod": WQIParameter(
        name="BOD",
        unit="mg/L",
        ideal_value=0.0,
        permissible=3.0,    # CPCB Class C limit
        weight=3.0,
        higher_is_worse=True,
    ),
    "turbidity": WQIParameter(
        name="Turbidity",
        unit="FNU",
        ideal_value=0.0,
        permissible=10.0,   # IS 10500 drinking limit; relaxed for Class C
        weight=2.0,
        higher_is_worse=True,
    ),
    "chl_a": WQIParameter(
        name="Chlorophyll-a",
        unit="µg/L",
        ideal_value=0.0,
        permissible=10.0,   # CPCB eutrophication alert threshold
        weight=2.0,
        higher_is_worse=True,
    ),
    "ph": WQIParameter(
        name="pH",
        unit="-",
        ideal_value=7.0,
        permissible=8.5,    # IS 10500; deviation from 7 is the measure
        weight=2.0,
        higher_is_worse=True,   # distance from 7 is worse
    ),
}

# Normalised weights (sum to 1)
_TOTAL_WEIGHT = sum(p.weight for p in PARAMETERS.values())
NORM_WEIGHTS = {k: v.weight / _TOTAL_WEIGHT for k, v in PARAMETERS.items()}


# ---------------------------------------------------------------------------
# Quality rating per parameter
# ---------------------------------------------------------------------------

def _quality_rating(value: float, param: WQIParameter) -> float:
    """
    Qi = 100 * (Vi - Vs) / (Si - Vs)   for pollutants (UNCAPPED — extreme values produce Qi > 100)
    For DO (higher is better):
        Qi = 100 * (Vs - Vi) / (Vs - Si)   — low DO → high Qi penalty
    For pH:
        Qi = 100 * |Vi - 7| / |Si - 7|
    Qi is NOT clamped so that severely polluted water bodies (e.g. Buddha Nullah)
    can produce WQI > 100 and land in Class E.
    """
    if param.name == "pH":
        ideal_diff = abs(param.permissible - param.ideal_value)
        if ideal_diff == 0:
            return 0.0
        qi = 100.0 * abs(value - param.ideal_value) / ideal_diff
    elif not param.higher_is_worse:
        # DO: low measured value → high penalty
        # At DO == ideal_value → Qi = 0 (perfect); at DO == permissible → Qi = 100
        ideal_diff = param.ideal_value - param.permissible
        if ideal_diff == 0:
            return 0.0
        qi = 100.0 * (param.ideal_value - value) / ideal_diff
    else:
        ideal_diff = abs(param.permissible - param.ideal_value)
        if ideal_diff == 0:
            return 0.0
        qi = 100.0 * (value - param.ideal_value) / ideal_diff

    # Clamp at 0 from below (negative = cleaner than ideal, treat as 0)
    return float(max(qi, 0.0))


# ---------------------------------------------------------------------------
# WQI class assignment
# ---------------------------------------------------------------------------

CPCB_CLASSES = [
    (90, "A", "Excellent — suitable for drinking with conventional treatment"),
    (70, "B", "Good — suitable for outdoor bathing"),
    (50, "C", "Medium — drinking with extensive purification"),
    (25, "D", "Bad — suitable for propagation of wildlife / fisheries"),
    (0,  "E", "Very Bad — suitable for irrigation only"),
]


def wqi_class(wqi_score: float) -> tuple[str, str]:
    """
    Returns (class_letter, description) for a given WQI score.
    WQI is on an inverted scale here: higher WQI → worse quality.

    CPCB convention used in this engine:
        < 25    → A (Excellent)
        25-50   → B (Good)
        50-75   → C (Medium)
        75-100  → D (Bad)
        > 100   → E (Very Bad)

    Note: Some CPCB documents use the direct score; others use 100 - score.
    We use the pollution-index convention (0 = pure, 100+ = heavily polluted).
    """
    if wqi_score < 25:
        return ("A", "Excellent — suitable for drinking with conventional treatment")
    elif wqi_score < 50:
        return ("B", "Good — suitable for outdoor bathing")
    elif wqi_score < 75:
        return ("C", "Medium — drinking with extensive purification")
    elif wqi_score <= 100:
        return ("D", "Bad — suitable for propagation of wildlife / fisheries")
    else:
        return ("E", "Very Bad — suitable for irrigation only / hazardous")


# ---------------------------------------------------------------------------
# Core WQI computation
# ---------------------------------------------------------------------------

def compute_wqi(
    do: float,
    bod: float,
    turbidity: float,
    chl_a: float,
    ph: float = 7.5,
) -> dict:
    """
    Compute CPCB weighted-arithmetic WQI from predicted water quality parameters.

    Args:
        do        : Dissolved Oxygen [mg/L]
        bod       : Biochemical Oxygen Demand [mg/L]
        turbidity : Turbidity [FNU]
        chl_a     : Chlorophyll-a [µg/L]
        ph        : pH (default 7.5 if not measured)

    Returns:
        dict with keys:
            wqi        : float, WQI score (0-100+, lower is better)
            wqi_class  : str, CPCB class "A"–"E"
            description: str, human-readable class description
            sub_index  : dict, per-parameter Qi contributions
            weights    : dict, normalised weights used
    """
    values = {
        "do":        do,
        "bod":       bod,
        "turbidity": turbidity,
        "chl_a":     chl_a,
        "ph":        ph,
    }

    sub_index = {}
    weighted_sum = 0.0

    for key, param in PARAMETERS.items():
        qi = _quality_rating(values[key], param)
        wi = NORM_WEIGHTS[key]
        sub_index[key] = round(qi, 2)
        weighted_sum += wi * qi

    wqi_score = round(weighted_sum, 2)
    cls, desc = wqi_class(wqi_score)

    return {
        "wqi":         wqi_score,
        "wqi_class":   cls,
        "description": desc,
        "sub_index":   sub_index,
        "weights":     NORM_WEIGHTS,
    }


# ---------------------------------------------------------------------------
# Batch computation over DataFrame
# ---------------------------------------------------------------------------

def compute_wqi_dataframe(df: pd.DataFrame, ph_default: float = 7.5) -> pd.DataFrame:
    """
    Vectorised WQI computation over a DataFrame.

    Required columns: do, bod, turbidity, chl_a
    Optional column : ph

    Adds columns: wqi, wqi_class, wqi_description, plus Qi sub-index columns.

    Args:
        df         : DataFrame with predicted parameters
        ph_default : Default pH if column not present

    Returns:
        df with WQI columns appended
    """
    df = df.copy()

    if "ph" not in df.columns:
        df["ph"] = ph_default

    required = ["do", "bod", "turbidity", "chl_a", "ph"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")

    results = df.apply(
        lambda row: compute_wqi(
            do=row["do"],
            bod=row["bod"],
            turbidity=row["turbidity"],
            chl_a=row["chl_a"],
            ph=row["ph"],
        ),
        axis=1,
    )

    df["wqi"]             = results.apply(lambda r: r["wqi"])
    df["wqi_class"]       = results.apply(lambda r: r["wqi_class"])
    df["wqi_description"] = results.apply(lambda r: r["description"])

    # Sub-index columns for dashboard breakdown
    for param_key in PARAMETERS:
        df[f"qi_{param_key}"] = results.apply(lambda r: r["sub_index"][param_key])

    return df


# ---------------------------------------------------------------------------
# CLI / quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Hand-computed test cases
    test_cases = [
        # (do, bod, turbidity, chl_a, label)
        (8.0,  1.5,  2.0,   4.0,  "Clean river — expect A/B"),
        (5.5,  3.0,  12.0,  15.0, "Moderate pollution — expect B/C"),
        (3.0,  8.0,  45.0,  60.0, "Eutrophic lake — expect C/D"),
        (1.2, 25.0, 200.0, 110.0, "Buddha Nullah equivalent — expect E"),
        (0.5, 80.0, 450.0, 150.0, "Severely polluted — expect E"),
    ]

    print("=" * 65)
    print(f"{'Case':<35} {'WQI':>6} {'Class':>6}")
    print("=" * 65)
    for do, bod, turb, chl, label in test_cases:
        result = compute_wqi(do=do, bod=bod, turbidity=turb, chl_a=chl)
        print(f"{label:<35} {result['wqi']:>6.1f} {result['wqi_class']:>6}")
    print("=" * 65)
