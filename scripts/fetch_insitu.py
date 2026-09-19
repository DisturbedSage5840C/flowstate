"""Download CPCB surface-water data from NWDP and build the in-situ master table.

    python -m scripts.fetch_insitu [--states Karnataka Punjab] [--since 2019-01-01]

Writes data/ground_truth/insitu_master.parquet and prints a QC summary. No login is needed.
"""
import argparse

from src.data import nwdp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", nargs="*")
    ap.add_argument("--since", default="2019-01-01")
    ap.add_argument("--out", default=str(nwdp.ROOT / "data" / "ground_truth" / "insitu_master.parquet"))
    args = ap.parse_args()

    res = nwdp.fetch_all(states=args.states)
    print(f"cached {len(res)} NWDP resources in {nwdp.CACHE_DIR}")
    df = nwdp.build_insitu_table(states=args.states, since=args.since)
    surf = df[df["water_body_type"] != "groundwater"]
    print(f"{len(df):,} visits ({len(surf):,} surface-water) | {surf['station'].nunique():,} stations | "
          f"{surf['state'].nunique()} states | {df['date'].min().date()} .. {df['date'].max().date()}")
    print("labels available (surface):", surf[["do", "bod", "turbidity", "ph", "conductivity"]].notna().sum().to_dict())
    print("QC:", df.attrs.get("qc"))
    df.drop(columns=[c for c in df.columns if c.startswith("file_")]).to_parquet(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
