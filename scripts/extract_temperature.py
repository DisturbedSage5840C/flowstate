"""Landsat 8/9 surface temperature for the visits in the real training table (Planetary Computer, no login).

    python -m scripts.extract_temperature --workers 4

Resumable; writes data/interim/station_temperature.parquet. Rebuild the table afterwards with
python -m scripts.build_real_training_table so ``temp_surface`` is filled where a Landsat retrieval exists.
"""
import argparse
import logging

import pandas as pd

from src.data import nwdp
from src.data.landsat_temp import LandsatTempExtractor

TABLE = nwdp.ROOT / "data" / "processed" / "train_real.parquet"
OUT = nwdp.ROOT / "data" / "interim" / "station_temperature.parquet"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="only the first N visits (smoke test)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t = pd.read_parquet(TABLE)
    visits = t[["site", "lat", "lon", "date"]].rename(columns={"site": "station"}).drop_duplicates(["station", "date"])
    if args.limit:
        visits = visits.head(args.limit)
    result = LandsatTempExtractor(workers=args.workers).extract(visits, cache_path=args.out)
    print(result["temp_status"].value_counts().to_string())
    print("wrote", args.out)


if __name__ == "__main__":
    main()
