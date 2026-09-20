"""ACOLITE dry run: correct one real Sentinel-2 L1C scene and compare it with the Sen2Cor values in the training table.

    ACOLITE_HOME=... ACOLITE_PYTHON="micromamba run -r ROOT -n acolite python" \\
      python -m scripts.acolite_dry_run --safe data/raw/l1c/<scene>.SAFE --name hyderabad_lakes \\
        --bbox 78.15 17.50 78.35 17.60 --scene-key 20201104T051011 --tile T43QHV

Steps: run_acolite (Dark Spectrum Fitting) -> acolite_to_contract (band-contract GeoTIFF + water mask) -> sample the
result at every CPCB station inside the bbox that the training table matched to the SAME scene -> compare band by band
with the Sen2Cor L2A reflectance stored in ``train_real_large.parquet`` (median ratio, Pearson r, Spearman rho).
Writes reports/real/acolite_dry_run.json.

The comparison is between two atmospheric corrections of the same acquisition, not against ground truth: it shows
the two agree or disagree, not which is right. Stations are sampled as the median of the valid water pixels in a
7 x 7 window (70 m), because CPCB stations sit on small tanks, which is coarser than the pipeline's own station-buffer extraction, so treat the numbers as
indicative.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rasterio.warp
from scipy.stats import pearsonr, spearmanr

from src.acquisition.sites import ROOT, Site
from src.data import sensors
from src.preprocessing.correction import correct_scene
from src.preprocessing.correction_convert import date_from_safe

TABLE = ROOT / "data" / "processed" / "train_real_large.parquet"
OUT = ROOT / "reports" / "real" / "acolite_dry_run.json"
BANDS = ["B2", "B3", "B4", "B5", "B6", "B8", "B11"]


def sample_station(src: rasterio.io.DatasetReader, lat: float, lon: float, band_idx: dict, half: int = 3) -> dict | None:
    x, y = rasterio.warp.transform("EPSG:4326", src.crs, [lon], [lat])
    row, col = src.index(x[0], y[0])
    if not (half <= row < src.height - half and half <= col < src.width - half):
        return None
    win = rasterio.windows.Window(col - half, row - half, 2 * half + 1, 2 * half + 1)
    data = src.read(window=win).astype("float64")
    water = data[band_idx["water"] - 1] > 0.5
    out = {}
    for b in BANDS:
        v = data[band_idx[b] - 1]
        v = v[water & (v > -9000) & np.isfinite(v)]
        if v.size:
            out[b] = float(np.median(v))
    return out if len(out) == len(BANDS) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--safe", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("W", "S", "E", "N"), required=True)
    ap.add_argument("--scene-key", required=True, help="acquisition stamp inside the table's scene_id, e.g. 20201104T051011")
    ap.add_argument("--tile", required=True, help="e.g. T43QHV")
    ap.add_argument("--reuse", action="store_true", help="skip ACOLITE if the contract raster already exists")
    args = ap.parse_args()

    site = Site(name=args.name, type="lake", state="", bbox=tuple(args.bbox))
    existing = ROOT / "data" / "interim" / f"{args.name}_S2_{date_from_safe(args.safe)}.tif"
    res = ({"contract_tif": str(existing)} if args.reuse and existing.exists()
           else correct_scene(os.path.abspath(args.safe), site, method="acolite"))
    if "contract_tif" not in res:
        raise SystemExit(f"ACOLITE/conversion failed: {res}")
    print("contract raster:", res["contract_tif"])

    df = pd.read_parquet(TABLE)
    w, s, e, n = args.bbox
    hit = df[df["scene_id"].astype(str).str.contains(args.scene_key) & df["scene_id"].astype(str).str.contains(args.tile)
             & df["lon"].between(w, e) & df["lat"].between(s, n)]
    stations = hit.groupby("site").first().reset_index()

    rows = []
    with rasterio.open(res["contract_tif"]) as src:
        names = src.tags().get("BANDS", "").split(",")
        band_idx = {nm: i + 1 for i, nm in enumerate(names)}
        for _, r in stations.iterrows():
            ac = sample_station(src, r["lat"], r["lon"], band_idx)
            if ac:
                rows.append({"site": r["site"], **{f"acolite_{b}": ac[b] for b in BANDS},
                             **{f"sen2cor_{b}": float(r[b]) for b in BANDS}})
    comp = pd.DataFrame(rows)
    report = {"scene": Path(args.safe).name, "bbox": args.bbox, "correction_method": "acolite_dsf",
              "acolite_version": "GitHub clone 2026-09-20", "stations_in_scene": int(len(stations)),
              "stations_compared": int(len(comp)), "bands": {}}
    for b in BANDS:
        a, z = comp[f"acolite_{b}"].to_numpy(), comp[f"sen2cor_{b}"].to_numpy()
        ok = np.isfinite(a) & np.isfinite(z) & (z > 0)
        report["bands"][b] = {
            "median_ratio_acolite_over_sen2cor": float(np.median(a[ok] / z[ok])) if ok.any() else None,
            "pearson_r": float(pearsonr(a[ok], z[ok])[0]) if ok.sum() > 2 else None,
            "spearman_rho": float(spearmanr(a[ok], z[ok])[0]) if ok.sum() > 2 else None,
            "acolite_median": float(np.median(a[ok])) if ok.any() else None,
            "sen2cor_median": float(np.median(z[ok])) if ok.any() else None}
    report["per_station"] = comp.round(5).to_dict(orient="records")
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_station"}, indent=2))


if __name__ == "__main__":
    main()
