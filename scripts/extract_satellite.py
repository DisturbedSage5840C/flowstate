"""Match a sample of CPCB visits with Sentinel-2 L2A reflectance (Planetary Computer, no login).

    python -m scripts.extract_satellite --per-state 100 --workers 8

Resumable: results are checkpointed to data/interim/station_reflectance.parquet.
"""
import argparse
import logging

import pandas as pd

from src.data import nwdp
from src.data.insitu import select_dense_visits, select_visits
from src.data.satellite_extract import ExtractionConfig, StacExtractor

MASTER = nwdp.ROOT / "data" / "ground_truth" / "insitu_master.parquet"
OUT = nwdp.ROOT / "data" / "interim" / "station_reflectance.parquet"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-state", type=int, default=100)
    ap.add_argument("--per-station", type=int, default=6)
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2021-12-31")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dense", type=int, metavar="N_STATIONS",
                    help="instead of a capped sample, take ALL visits of N regularly-sampled stations "
                         "(gives temporal models a usable per-station history)")
    ap.add_argument("--min-visits", type=int, default=12, help="with --dense: minimum labelled visits per station")
    ap.add_argument("--limit", type=int, help="only the first N selected visits (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    table = pd.read_parquet(MASTER)
    if args.dense:
        visits = select_dense_visits(table, n_stations=args.dense, min_visits=args.min_visits,
                                     start=args.start, end=args.end, seed=args.seed)
    else:
        visits = select_visits(table, per_state=args.per_state, per_station=args.per_station,
                               start=args.start, end=args.end, seed=args.seed)
    if args.limit:
        visits = visits.head(args.limit)
    print(f"selected {len(visits):,} visits at {visits['station'].nunique():,} stations in "
          f"{visits['state'].nunique()} states")
    result = StacExtractor(ExtractionConfig(), workers=args.workers).extract(
        visits[["station", "lat", "lon", "date"]], cache_path=args.out)
    print(result["status"].value_counts().to_string())
    print("wrote", args.out)


if __name__ == "__main__":
    main()
