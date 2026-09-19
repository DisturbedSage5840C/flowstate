"""Cross-check Earth Engine against Planetary Computer on the same station-visits.

    python -m scripts.compare_backends --n 120

Samples visits (spread over states) that the Planetary Computer run marked 'ok', runs the Earth Engine extractor on
them and reports how often the two pick the same scene / status and how far the band medians differ.
Writes reports/real/backend_comparison.json. Needs Earth Engine access (docs/EARTH_ENGINE_SETUP.md).
"""
import argparse
import json
import logging

import numpy as np
import pandas as pd

from src.data import nwdp
from src.data.gee_extract import BANDS, GeeExtractor

INTERIM = nwdp.ROOT / "data" / "interim"
OUT = nwdp.ROOT / "reports" / "real" / "backend_comparison.json"


def load_stac() -> pd.DataFrame:
    files = sorted(INTERIM.glob("station_reflectance*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates(["station", "date"])


def compare(stac: pd.DataFrame, gee: pd.DataFrame) -> dict:
    m = stac.merge(gee, on=["station", "date"], suffixes=("_stac", "_gee"))
    both_ok = m[(m["status_stac"] == "ok") & (m["status_gee"] == "ok")]
    same_scene = both_ok[both_ok["day_diff_stac"].astype(float) == both_ok["day_diff_gee"].astype(float)]
    res = {
        "visits_compared": int(len(m)),
        "gee_status_counts": m["status_gee"].value_counts().to_dict(),
        "both_ok": int(len(both_ok)),
        "share_ok_in_both": round(float(len(both_ok) / max(len(m), 1)), 3),
        "same_day_diff_when_both_ok": round(float(len(same_scene) / max(len(both_ok), 1)), 3),
        "bands": {},
    }
    for b in BANDS:
        x, y = same_scene[f"{b}_stac"].to_numpy(float), same_scene[f"{b}_gee"].to_numpy(float)
        if len(x) < 3:
            continue
        rel = np.abs(y - x) / np.maximum(np.abs(x), 1e-4)
        res["bands"][b] = {"median_rel_diff": round(float(np.median(rel)), 4),
                           "p90_rel_diff": round(float(np.quantile(rel, 0.9)), 4),
                           "pearson_r": round(float(np.corrcoef(x, y)[0, 1]), 4)}
    if len(same_scene) >= 3:
        res["n_water_px_median_ratio_gee_over_stac"] = round(float(np.median(
            same_scene["n_water_px_gee"].astype(float) / same_scene["n_water_px_stac"].astype(float))), 3)
    res["note"] = ("Same scenes (same day_diff) only. Differences are expected from nearest-neighbour vs bilinear "
                   "resampling of the 20 m bands and coverage-weighted pixel counts.")
    return res


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--project", default="prayashack")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    stac = load_stac()
    master = pd.read_parquet(nwdp.ROOT / "data" / "ground_truth" / "insitu_master.parquet")
    meta = master[["station", "lat", "lon", "state"]].drop_duplicates("station")
    ok = stac[stac["status"] == "ok"].merge(meta, on="station")
    rng = np.random.default_rng(args.seed)
    ok = ok.assign(_r=rng.random(len(ok))).sort_values("_r")
    per_state = max(1, int(np.ceil(args.n / ok["state"].nunique())))
    sample = ok.groupby("state", group_keys=False).head(per_state).head(args.n)
    visits = sample[["station", "lat", "lon", "date"]]

    gee = GeeExtractor(project=args.project).extract(visits)
    result = compare(sample, gee)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
