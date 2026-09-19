"""Pick a per-site MNDWI water threshold from real imagery (Otsu on a sampled composite).

Requires Earth Engine credentials. Writes config/mndwi_thresholds.yaml (never touches sites.yaml):

    python -m scripts.tune_mndwi --start 2024-01-01 --end 2024-06-30 [--sites bellandur dal_lake]

A site's Otsu split is accepted only if the sample is clearly bimodal (see masking.choose_threshold); otherwise the
standard default 0.0 is kept and the reason is recorded.
"""
import argparse
import datetime as dt

import numpy as np
import yaml

from src.acquisition import gee
from src.acquisition.sites import DEFAULT_THRESHOLDS, load_sites
from src.preprocessing import masking


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="*")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--max-cloud", type=float, default=20)
    ap.add_argument("--project")
    args = ap.parse_args()

    import ee
    gee.init_ee(args.project)
    sites = load_sites(thresholds_path=None)
    tuned = {}
    if DEFAULT_THRESHOLDS.exists():
        tuned = (yaml.safe_load(DEFAULT_THRESHOLDS.read_text()) or {}).get("thresholds", {})

    for name in args.sites or sites:
        site = sites[name]
        region = gee.site_region(site)
        col = gee._s2_collection(region, args.start, args.end, args.max_cloud)
        if col.size().getInfo() == 0:
            print(f"{name}: no scenes, skipped")
            continue
        img = col.median()
        try:
            thr, vals = masking.suggest_threshold(img, "S2", region)
        except RuntimeError as e:
            print(f"{name}: {e}")
            continue
        thr, method = masking.choose_threshold(vals)
        share_water = float((vals > thr).mean())
        pcts = np.percentile(vals, [5, 25, 50, 75, 95]).round(2).tolist()
        print(f"{name:26s} threshold={thr:+.3f} [{method}]  share above={share_water:.2f}  MNDWI p5/25/50/75/95={pcts}")
        tuned[name] = {"threshold": round(thr, 3), "method": method, "period": f"{args.start}..{args.end}",
                       "n_pixels": int(vals.size), "tuned_on": dt.date.today().isoformat()}

    DEFAULT_THRESHOLDS.write_text(yaml.safe_dump({"thresholds": tuned}, sort_keys=True))
    print(f"wrote {DEFAULT_THRESHOLDS}")


if __name__ == "__main__":
    main()
