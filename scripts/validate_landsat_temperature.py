"""Validate the Landsat 8/9 water-temperature retrieval against the Kaggle IoT sensors (Prayagraj, 2019-20).

    python -m scripts.validate_landsat_temperature

Needs the Kaggle files (python -c "from src.data.kaggle_sources import download; download()", Kaggle token).
Writes reports/real/landsat_temperature_validation.json and .csv.
"""
import json

import numpy as np
import pandas as pd

from src.data import kaggle_sources as ks
from src.data import nwdp
from src.data.landsat_temp import LandsatTempExtractor
from src.data.satellite_extract import ExtractionConfig

OUT = nwdp.ROOT / "reports" / "real"


def stats(insitu: np.ndarray, satellite: np.ndarray) -> dict:
    d = satellite - insitu
    return {"n": int(len(d)),
            "bias_c": round(float(d.mean()), 2), "rmse_c": round(float(np.sqrt((d ** 2).mean())), 2),
            "mae_c": round(float(np.abs(d).mean()), 2),
            "pearson_r": round(float(np.corrcoef(insitu, satellite)[0, 1]), 3) if len(d) > 2 else None}


def main():
    ex = LandsatTempExtractor(ExtractionConfig(day_tolerance=1, candidates=3, max_scene_cloud=80.0), workers=4)
    pairs = []
    for name in ks.LOCATIONS:
        frame = ks.station_frame(name).dropna(subset=["temp_c"])
        res = ex.extract(frame[["station", "lat", "lon", "date"]])
        merged = frame.merge(res, on=["station", "date"], how="left", suffixes=("_insitu", "_landsat"))
        merged["series"] = name
        pairs.append(merged)
    p = pd.concat(pairs, ignore_index=True)
    ok = p[p["temp_status"] == "ok"].copy()

    result = {
        "insitu_days_tested": int(len(p)),
        "landsat_retrievals": int(len(ok)),
        "independent_landsat_scenes": int(ok["temp_scene_date"].nunique()),     # consecutive days can share one scene
        "status_counts": p["temp_status"].value_counts().to_dict(),
        "overall": stats(ok["temp_c_insitu"].to_numpy(), ok["temp_c_landsat"].to_numpy()) if len(ok) else None,
        "by_series": {s: stats(g["temp_c_insitu"].to_numpy(), g["temp_c_landsat"].to_numpy())
                      for s, g in ok.groupby("series")},
        "bias_when_insitu_at_least_25c": (round(float((ok.loc[ok["temp_c_insitu"] >= 25, "temp_c_landsat"]
                                           - ok.loc[ok["temp_c_insitu"] >= 25, "temp_c_insitu"]).mean()), 2)
                              if (ok["temp_c_insitu"] >= 25).any() else None),
        "bias_when_insitu_below_25c": (round(float((ok.loc[ok["temp_c_insitu"] < 25, "temp_c_landsat"]
                                            - ok.loc[ok["temp_c_insitu"] < 25, "temp_c_insitu"]).mean()), 2)
                               if (ok["temp_c_insitu"] < 25).any() else None),
        "interpretation": ("Landsat ST is a skin temperature. It follows the seasonal cycle but exaggerates its amplitude "
                           "(too warm in the hot-season scenes, too cool in winter); do not use it as an accurate water "
                           "temperature (e.g. for DO saturation)."),
        "note": ("Sensor coordinates are those of the nearest CPCB stations (assumption); the in-situ value is the "
                 "median of plausible readings between 09:00 and 14:00 local time."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "landsat_temperature_validation.json").write_text(json.dumps(result, indent=2))
    ok[["series", "date", "temp_c_insitu", "temp_c_landsat", "temp_day_diff", "temp_n_px", "temp_scene_date"]].to_csv(
        OUT / "landsat_temperature_validation.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
