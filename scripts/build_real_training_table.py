"""Join CPCB in-situ visits with the extracted Sentinel-2 reflectance -> data/processed/train_real.parquet.

    python -m scripts.build_real_training_table [--min-water-px 20]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.data import nwdp
from src.data.training_table import TARGETS, build_training_table

MASTER = nwdp.ROOT / "data" / "ground_truth" / "insitu_master.parquet"
REFL = nwdp.ROOT / "data" / "interim" / "station_reflectance.parquet"
OUT = nwdp.ROOT / "data" / "processed" / "train_real.parquet"
REPORT = nwdp.ROOT / "reports" / "real" / "dataset_summary.json"


def load_reflectance() -> pd.DataFrame:
    """Merge every data/interim/station_reflectance*.parquet (capped sample + dense run); prefer 'ok' rows."""
    files = sorted(REFL.parent.glob("station_reflectance*.parquet"))
    if not files:
        raise SystemExit("no extraction found - run python -m scripts.extract_satellite first")
    refl = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    refl["date"] = pd.to_datetime(refl["date"])
    refl["_ok"] = (refl["status"] == "ok").astype(int)
    refl = refl.sort_values("_ok", ascending=False).drop_duplicates(["station", "date"]).drop(columns="_ok")
    print(f"merged {len(files)} extraction file(s): {len(refl):,} station-visits")
    return refl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-water-px", type=int, default=20)
    ap.add_argument("--out", default=str(OUT), help="output parquet (default: data/processed/train_real.parquet)")
    ap.add_argument("--summary", default=str(REPORT), help="output summary json")
    args = ap.parse_args()

    insitu = pd.read_parquet(MASTER)
    refl = load_reflectance()
    temp_files = sorted(REFL.parent.glob("station_temperature*.parquet"))
    temperature = pd.concat([pd.read_parquet(f) for f in temp_files], ignore_index=True) if temp_files else None
    table = build_training_table(insitu, refl, min_water_px=args.min_water_px, temperature=temperature)
    out_path, summary_path = Path(args.out), Path(args.summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_path, index=False)

    status = refl["status"].value_counts().to_dict()
    summary = {
        "source": "CPCB NWDP manual monitoring (real, in-situ) + Sentinel-2 L2A via Planetary Computer",
        "is_proxy": False,
        "rows": int(len(table)),
        "stations": int(table["site"].nunique()),
        "states": int(table["state"].nunique()),
        "date_range": [str(table["date"].min().date()), str(table["date"].max().date())] if len(table) else None,
        "extraction_status_counts": {k: int(v) for k, v in status.items()},
        "labels_available": {t: int(table[t].notna().sum()) for t in TARGETS},
        "temperature_available": int(table["temp_surface"].notna().sum()),
        "by_water_body_type": table["water_body_type"].value_counts().to_dict(),
        "day_diff_days": table["day_diff"].value_counts().sort_index().to_dict() if len(table) else {},
        "min_water_px": args.min_water_px,
        "cpcb_class_counts": table["cpcb_class"].value_counts(dropna=False).to_dict(),
        "wqi_tier_counts": table["wqi_tier"].value_counts(dropna=False).to_dict(),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
