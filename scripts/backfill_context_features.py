"""Add season/urban-proxy/rainfall context features to the already-built real training tables.

    python -m scripts.backfill_context_features [--offline] [--all]

Backfills the non-satellite features (src.models.schema.SEASONS/RAINFALL_COLS/URBAN_PROXY_COLS) directly onto
the already-built parquets, without rerunning the satellite extraction.

Idempotent and, importantly, **non-destructive for rainfall**: the daily-rainfall cache lives in
data/interim/ (gitignored), so a clone that has the tables but not the cache would otherwise recompute every
rainfall feature from an empty cache and wipe the coverage already in the parquet. Instead only the visits
whose rainfall is still missing are fetched and filled; existing values are kept. ``--all`` forces a full
recompute (use after changing the rainfall feature definition).
"""
import argparse

import pandas as pd

from src.data import nwdp
from src.data.city_proximity import urban_proxy_features
from src.data.weather import antecedent_rainfall_features, compute_rainfall_windows, fetch_daily_rainfall
from src.models.schema import RAINFALL_COLS, add_season_onehot

TABLE_LARGE = nwdp.ROOT / "data" / "processed" / "train_real_large.parquet"
TABLE_SMALL = nwdp.ROOT / "data" / "processed" / "train_real.parquet"
RAINFALL_CACHE = nwdp.ROOT / "data" / "interim" / "rainfall_daily.parquet"


def missing_rainfall_mask(df: pd.DataFrame) -> pd.Series:
    """Rows whose rainfall features still need fetching (all of them, if the columns do not exist yet)."""
    if not all(c in df.columns for c in RAINFALL_COLS):
        return pd.Series(True, index=df.index)
    return df[RAINFALL_COLS].isna().any(axis=1)


def add_context_features(df: pd.DataFrame, rainfall_daily: pd.DataFrame, refresh_all: bool = False) -> pd.DataFrame:
    """Season + urban proxy (always recomputed; they are deterministic from date/lat/lon) and rainfall
    (only where missing, unless ``refresh_all``)."""
    df = add_season_onehot(df)
    urban = urban_proxy_features(df["lat"], df["lon"])
    df = df.drop(columns=[c for c in urban.columns if c in df.columns]).join(urban)

    for c in RAINFALL_COLS:
        if c not in df.columns:
            df[c] = float("nan")
    target = pd.Series(True, index=df.index) if refresh_all else missing_rainfall_mask(df)
    if not target.any() or not len(rainfall_daily):
        return df
    feats = antecedent_rainfall_features(df.loc[target, ["site", "date"]], rainfall_daily)
    for c in RAINFALL_COLS:
        new = feats[c]
        df.loc[target, c] = new.where(new.notna(), df.loc[target, c]) if not refresh_all else new
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="use only the cached rainfall on disk, no network calls")
    ap.add_argument("--all", action="store_true", dest="refresh_all",
                    help="recompute rainfall for every row (default: only rows still missing it)")
    args = ap.parse_args()

    big = pd.read_parquet(TABLE_LARGE)
    big["date"] = pd.to_datetime(big["date"])

    need = pd.Series(True, index=big.index) if args.refresh_all else missing_rainfall_mask(big)
    visits = big.loc[need, ["site", "date"]]
    windows = compute_rainfall_windows(visits)
    days = int(sum((w.end_date - w.start_date).days + 1 for w in windows.itertuples()))
    print(f"rainfall still needed for {int(need.sum())} of {len(big)} rows "
          f"({visits['site'].nunique()} sites) -> {len(windows)} windows, {days:,} site-days "
          f"(~{days / 14:,.0f} Open-Meteo call-units; free tier allows 6,000/hour, 10,000/day)")

    RAINFALL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if args.offline:
        print("--offline: using only the cached rainfall already on disk, no fetch attempted")
    sites = big.loc[need, ["site", "lat", "lon"]].drop_duplicates(subset="site")
    rainfall_daily = fetch_daily_rainfall(sites, windows, cache_path=RAINFALL_CACHE, offline=args.offline)
    print(f"rainfall cache: {rainfall_daily['site'].nunique()} sites, {len(rainfall_daily):,} site-days")

    for path in (TABLE_LARGE, TABLE_SMALL):
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        df = add_context_features(df, rainfall_daily, refresh_all=args.refresh_all)
        df.to_parquet(path)
        cov = {c: f"{df[c].notna().mean():.1%}" for c in RAINFALL_COLS}
        print(f"wrote {path.name} ({len(df)} rows) | rainfall coverage {cov}")


if __name__ == "__main__":
    main()
