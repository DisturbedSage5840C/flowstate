"""Add soil-composition and land-use context features to the already-built real training tables.

    python -m scripts.backfill_soil_land_cover [--offline]

Sibling to scripts/backfill_context_features.py (same idempotent/resumable/cached pattern). Adds
src.models.schema.SOIL_COLS (src.data.soil, ISRIC SoilGrids) and LAND_COVER_COLS (src.data.land_cover,
ESA WorldCover via Planetary Computer) to both parquets.

**Has not been run to completion in this repo's development sandbox**: verified no network route to
either rest.isric.org or planetarycomputer.microsoft.com from there (see src/data/soil.py and
src/data/land_cover.py's module docstrings). Run this from an environment with real network access, then
ablate the new columns (clean, identical-row, identical-fold comparisons, the same method used for every
other feature in src/models/schema.py's FEATURE_SETS) before adding any of them to a feature set --
rainfall went through the same fetch-then-ablate process and did not survive it.

``--offline`` returns whatever is already cached with zero network calls (for testing the join/backfill
logic itself without needing the fetch to have run).
"""
import sys

import pandas as pd

from src.data import nwdp
from src.data.land_cover import fetch_land_cover_for_sites, land_cover_features
from src.data.soil import fetch_soil_for_sites, soil_features
from src.models.schema import LAND_COVER_COLS, SOIL_COLS

TABLE_LARGE = nwdp.ROOT / "data" / "processed" / "train_real_large.parquet"
TABLE_SMALL = nwdp.ROOT / "data" / "processed" / "train_real.parquet"
SOIL_CACHE = nwdp.ROOT / "data" / "interim" / "soil.parquet"
LAND_COVER_CACHE = nwdp.ROOT / "data" / "interim" / "land_cover.parquet"


def add_soil_land_cover(df: pd.DataFrame, soil_lookup: pd.DataFrame, land_cover_lookup: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    soil_feats = soil_features(df[["site"]], soil_lookup)
    for c in SOIL_COLS:
        df[c] = soil_feats[c].to_numpy()
    lc_feats = land_cover_features(df[["site"]], land_cover_lookup)
    for c in LAND_COVER_COLS:
        df[c] = lc_feats[c].to_numpy()
    return df


def main():
    offline = "--offline" in sys.argv
    big = pd.read_parquet(TABLE_LARGE)
    sites = big[["site", "lat", "lon"]].drop_duplicates(subset="site")

    SOIL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if offline:
        print("--offline: using only whatever is already cached on disk, no fetch attempted")
    soil_lookup = fetch_soil_for_sites(sites, cache_path=SOIL_CACHE, offline=offline)
    print(f"soil: {len(soil_lookup)}/{sites['site'].nunique()} stations")
    land_cover_lookup = fetch_land_cover_for_sites(sites, cache_path=LAND_COVER_CACHE, offline=offline)
    print(f"land cover: {len(land_cover_lookup)}/{sites['site'].nunique()} stations")

    big = add_soil_land_cover(big, soil_lookup, land_cover_lookup)
    big.to_parquet(TABLE_LARGE)
    print(f"wrote {TABLE_LARGE} ({len(big)} rows)")
    for c in [*SOIL_COLS, *LAND_COVER_COLS]:
        print(f"  {c}: {big[c].notna().mean():.1%} non-null")

    if TABLE_SMALL.exists():
        small = pd.read_parquet(TABLE_SMALL)
        small = add_soil_land_cover(small, soil_lookup, land_cover_lookup)
        small.to_parquet(TABLE_SMALL)
        print(f"wrote {TABLE_SMALL} ({len(small)} rows)")


if __name__ == "__main__":
    main()
