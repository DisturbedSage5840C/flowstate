"""Dual-tier atmospheric correction wrappers (ACOLITE DSF, C2RCC via SNAP GPT) with Hour-4 fallback.

Both tools need Sentinel-2 L1C SAFE products (from Copernicus Data Space, not Earth Engine).
Configure via env vars: ACOLITE_HOME (folder containing launch_acolite.py), SNAP_GPT (path to gpt).
"""
import logging
import os
import subprocess
import sys
from pathlib import Path

from src.acquisition.sites import ROOT, Site

log = logging.getLogger(__name__)
INTERIM = ROOT / "data" / "interim"


class CorrectionError(RuntimeError):
    pass


def _run(cmd: list[str], timeout: int) -> None:
    log.info("running: %s", " ".join(cmd))
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        raise CorrectionError(str(e)) from e
    if res.returncode != 0:
        raise CorrectionError(f"{cmd[0]} exited {res.returncode}: {res.stderr[-800:]}")


def run_acolite(safe_path: Path | str, site: Site, out_dir: Path | str = INTERIM / "acolite",
                timeout: int = 3600) -> Path:
    """Dark Spectrum Fitting -> Rrs GeoTIFFs clipped to the site bbox (tier 1: turbid / sunglint)."""
    home = os.environ.get("ACOLITE_HOME")
    if not home or not (Path(home) / "launch_acolite.py").exists():
        raise CorrectionError("ACOLITE_HOME not set or launch_acolite.py missing")
    out_dir = Path(out_dir) / site.name
    out_dir.mkdir(parents=True, exist_ok=True)
    w, s, e, n = site.bbox
    settings = out_dir / "acolite_settings.txt"
    settings.write_text("\n".join([
        f"inputfile={safe_path}", f"output={out_dir}",
        f"limit={s},{w},{n},{e}",  # S,W,N,E
        "atmospheric_correction_method=dark_spectrum",
        "l2w_parameters=Rrs_*", "rgb_rhot=False", "rgb_rhos=False", "l2r_export_geotiff=True",
        "dsf_residual_glint_correction=True",
    ]))
    _run([sys.executable, str(Path(home) / "launch_acolite.py"), "--cli", "--settings", str(settings)], timeout)
    if not list(out_dir.glob("*.tif")) and not list(out_dir.glob("*.nc")):
        raise CorrectionError("ACOLITE produced no output")
    return out_dir


def run_c2rcc(safe_path: Path | str, site: Site, out_dir: Path | str = INTERIM / "c2rcc",
              timeout: int = 3600) -> Path:
    """SNAP c2rcc.msi neural-network inversion (tier 2: hypereutrophic / CDOM-rich)."""
    gpt = os.environ.get("SNAP_GPT", "gpt")
    out_dir = Path(out_dir) / site.name
    out_dir.mkdir(parents=True, exist_ok=True)
    w, s, e, n = site.bbox
    wkt = f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"
    out = out_dir / f"{Path(safe_path).stem}_c2rcc.tif"
    _run([gpt, "c2rcc.msi", f"-Ssource={safe_path}", "-Pvalid_pixel_expression=B8 > 0",
          "-PoutputRrs=true", "-PoutputKd=true", f"-Pregion={wkt}", "-t", str(out), "-f", "GeoTIFF"], timeout)
    if not out.exists():
        raise CorrectionError("C2RCC produced no output")
    return out


def correct_scene(safe_path: Path | str, site: Site, method: str = "auto") -> dict:
    """Route by water type: rivers/turbid -> ACOLITE DSF, lakes -> both (C2RCC for eutrophic).

    Returns {'method': ..., 'outputs': [...]} or {'method': 'gee_sr_fallback'} when tooling fails,
    signalling the caller to use src.acquisition.gee.fetch_scenes surface reflectance instead.
    """
    tiers = {"acolite": [run_acolite], "c2rcc": [run_c2rcc],
             "auto": [run_acolite] if site.type == "river" else [run_acolite, run_c2rcc]}[method]
    outputs = []
    try:
        for fn in tiers:
            outputs.append(str(fn(safe_path, site)))
    except CorrectionError as e:
        log.warning("correction failed for %s (%s); falling back to GEE surface reflectance", site.name, e)
        return {"method": "gee_sr_fallback", "outputs": outputs, "error": str(e)}
    return {"method": "+".join(f.__name__.removeprefix("run_") for f in tiers), "outputs": outputs}
