"""Add season/urban-proxy/rainfall context features to the already-built real training tables.

    python -m scripts.backfill_context_features

This machine has no rasterio/earthengine, so the raw satellite-extraction pipeline
(scripts/build_real_training_table.py) cannot be rerun here; instead this backfills the new
non-satellite features (src.models.schema.SEASONS/RAINFALL_COLS/URBAN_PROXY_COLS) directly onto
the already-built parquets, the same way is_river/is_lake/turbidity_calibrated were added earlier.
Idempotent: safe to rerun (rainfall is cached to data/interim/rainfall_daily.parquet by site, so a
rerun only fetches sites not already cached).
"""
import sys

import pandas as pd

from src.data import nwdp
from src.data.city_proximity import urban_proxy_features
from src.data.weather import antecedent_rainfall_features, compute_rainfall_windows, fetch_daily_rainfall
from src.models.schema import RAINFALL_COLS, add_season_onehot

TABLE_LARGE = nwdp.ROOT / "data" / "processed" / "train_real_large.parquet"
TABLE_SMALL = nwdp.ROOT / "data" / "processed" / "train_real.parquet"
RAINFALL_CACHE = nwdp.ROOT / "data" / "interim" / "rainfall_daily.parquet"


def add_context_features(df: pd.DataFrame, rainfall_daily: pd.DataFrame) -> pd.DataFrame:
    df = add_season_onehot(df)
    urban = urban_proxy_features(df["lat"], df["lon"])
    df = df.drop(columns=[c for c in urban.columns if c in df.columns]).join(urban)
    rain_feats = antecedent_rainfall_features(df[["site", "date"]], rainfall_daily)
    for c in RAINFALL_COLS:
        df[c] = rain_feats[c].to_numpy()
    return df


def main():
    offline = "--offline" in sys.argv
    big = pd.read_parquet(TABLE_LARGE)
    big["date"] = pd.to_datetime(big["date"])

    sites = big[["site", "lat", "lon"]].drop_duplicates(subset="site")
    windows = compute_rainfall_windows(big[["site", "date"]])
    print(f"rainfall windows: {len(windows)} (vs {sites['site'].nunique()} sites x full date range previously)")
    RAINFALL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if offline:
        print("--offline: using only the cached rainfall data already on disk, no fetch attempted")
    rainfall_daily = fetch_daily_rainfall(sites, windows, cache_path=RAINFALL_CACHE, offline=offline)
    print(f"rainfall: {rainfall_daily['site'].nunique()} sites, {len(rainfall_daily)} site-days")

    big = add_context_features(big, rainfall_daily)
    big.to_parquet(TABLE_LARGE)
    print(f"wrote {TABLE_LARGE} ({len(big)} rows)")
    for c in RAINFALL_COLS:
        print(f"  {c}: {big[c].notna().mean():.1%} non-null, median {big[c].median():.2f}")

    if TABLE_SMALL.exists():
        small = pd.read_parquet(TABLE_SMALL)
        small["date"] = pd.to_datetime(small["date"])
        small = add_context_features(small, rainfall_daily)
        small.to_parquet(TABLE_SMALL)
        print(f"wrote {TABLE_SMALL} ({len(small)} rows)")


if __name__ == "__main__":
    main()
