"""Batch-export masked reflectance scenes for sites in config/sites.yaml.

  python -m scripts.run_acquisition --sites bellandur yamuna_delhi --start 2024-01-01 --end 2024-03-01
"""
import argparse

from src.acquisition import gee
from src.acquisition.sites import load_sites


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="*", help="site names (default: all)")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--sensor", choices=["S2", "LS"], default="S2")
    ap.add_argument("--max-cloud", type=float, default=30)
    ap.add_argument("--mndwi", type=float, default=0.0, help="MNDWI threshold; tune per site first")
    ap.add_argument("--project")
    args = ap.parse_args()

    gee.init_ee(args.project)
    sites = load_sites()
    for name in args.sites or sites:
        site = sites[name]
        col = gee.fetch_scenes(site, args.start, args.end, args.sensor, args.max_cloud, args.mndwi)
        dates = gee.list_dates(col)
        print(f"{name}: {len(dates)} scenes")
        for d in dates:
            print("  exported", gee.export_scene(site, col, d, args.sensor))


if __name__ == "__main__":
    main()
