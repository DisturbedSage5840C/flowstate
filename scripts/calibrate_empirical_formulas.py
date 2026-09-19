"""Refit the turbidity power law T = a * B4^b against real CPCB measurements.

    python -m scripts.calibrate_empirical_formulas

Log-log linear regression (log T = log a + b * log B4) of measured turbidity (NTU)
against Sentinel-2 B4 (red) reflectance, on rows with both values positive, from
data/processed/train_real_large.parquet. Fit once globally and once per
water_body_type. Prints the coefficients to paste into TURBIDITY_CALIBRATION_GLOBAL /
TURBIDITY_CALIBRATION_BY_TYPE in src/features/feature_engineering.py — it does not
write that file, since the constants are checked-in literals, not runtime config.
"""
import numpy as np
import pandas as pd

from src.data import nwdp

TABLE = nwdp.ROOT / "data" / "processed" / "train_real_large.parquet"


def fit_power_law(turbidity: pd.Series, red: pd.Series) -> tuple[float, float, int]:
    m = pd.concat([turbidity, red], axis=1, keys=["t", "r"]).dropna()
    m = m[(m["t"] > 0) & (m["r"] > 0)]
    if len(m) < 10:
        return float("nan"), float("nan"), len(m)
    log_b, log_a = np.polyfit(np.log(m["r"]), np.log(m["t"]), 1)
    return float(np.exp(log_a)), float(log_b), len(m)


def main():
    df = pd.read_parquet(TABLE)
    a, b, n = fit_power_law(df["turbidity"], df["B4"])
    print(f"GLOBAL   a={a:.2f} b={b:.2f}  (n={n})")
    for water_type, g in df.groupby("water_body_type"):
        a_t, b_t, n_t = fit_power_law(g["turbidity"], g["B4"])
        print(f"{water_type:<10} a={a_t:.2f} b={b_t:.2f}  (n={n_t})")


if __name__ == "__main__":
    main()
