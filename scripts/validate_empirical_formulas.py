"""Check the literature formulas against REAL CPCB measurements instead of trusting unverified constants.

    python -m scripts.validate_empirical_formulas

Compares, on matched station-visits:
  * turbidity_empirical (Dogliotti/Nechad blend on Sentinel-2 red + NIR) vs measured turbidity (NTU)
  * do_empirical (saturation-based surrogate)                            vs measured DO (mg/L)
Chlorophyll-a has no in-situ source, so chl_a_empirical is reported as NOT VALIDATED.
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from src.data import nwdp

TABLE = nwdp.ROOT / "data" / "processed" / "train_real.parquet"
OUT = nwdp.ROOT / "reports" / "real" / "empirical_formula_validation.json"


def compare(measured: pd.Series, estimate: pd.Series, log: bool = False) -> dict:
    m = pd.concat([measured, estimate], axis=1, keys=["y", "x"]).dropna()
    m = m[np.isfinite(m).all(axis=1)]
    if len(m) < 10:
        return {"n": int(len(m)), "note": "too few pairs"}
    y, x = m["y"].to_numpy(), m["x"].to_numpy()
    rho, _ = stats.spearmanr(x, y)
    out = {
        "n": int(len(m)),
        "spearman_rho": round(float(rho), 3),
        "pearson_r": round(float(np.corrcoef(x, y)[0, 1]), 3),
        "median_ratio_est_over_measured": round(float(np.median(x / np.maximum(y, 1e-6))), 3),
        "rmse": round(float(np.sqrt(np.mean((x - y) ** 2))), 3),
        "r2_identity": round(float(1 - np.sum((y - x) ** 2) / np.sum((y - y.mean()) ** 2)), 3),
    }
    if log:
        pos = (x > 0) & (y > 0)
        if pos.sum() > 10:
            out["pearson_r_log"] = round(float(np.corrcoef(np.log10(x[pos]), np.log10(y[pos]))[0, 1]), 3)
            out["within_factor_2"] = round(float(np.mean((x[pos] / y[pos] > 0.5) & (x[pos] / y[pos] < 2))), 3)
    return out


def main():
    t = pd.read_parquet(TABLE)
    res = {
        "rows": int(len(t)),
        "turbidity_dogliotti_vs_measured": compare(t["turbidity"], t["turbidity_empirical"], log=True),
        "do_surrogate_vs_measured": compare(t["do"], t["do_empirical"]),
        "ndci_vs_measured_spearman": {
            k: compare(t[k], t["ndci"]).get("spearman_rho") for k in ("do", "bod", "turbidity")
        },
        "chl_a_empirical": "NOT VALIDATED - no in-situ chlorophyll-a exists in CPCB NWDP",
    }
    for water_type, g in t.groupby("water_body_type"):
        res.setdefault("turbidity_by_type", {})[water_type] = compare(
            g["turbidity"], g["turbidity_empirical"], log=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
