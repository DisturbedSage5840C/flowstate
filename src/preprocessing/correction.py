"""Dual-tier atmospheric correction wrappers (ACOLITE DSF, C2RCC via SNAP GPT) with Hour-4 fallback.

Both tools need Sentinel-2 L1C SAFE products (from Copernicus Data Space, not Earth Engine).
Configure via env vars: ACOLITE_HOME (folder containing launch_acolite.py), SNAP_GPT (path to gpt), and optionally
ACOLITE_PYTHON (command to run ACOLITE with, e.g. ``micromamba run -r ROOT -n acolite python`` so the conda
activation variables are set; it needs GDAL's ``osgeo`` bindings, which pip cannot install on
Windows, so a conda-forge/micromamba environment is the practical route; defaults to the current interpreter).

ACOLITE dry run (2026-09-20, ``reports/real/acolite_dry_run.json``): one real Sentinel-2 L1C scene (Hyderabad lakes,
2020-11-04) was corrected end to end. Two practical limits found: a ~28 x 32 km subset died silently (memory), a
~10 x 20 km ``limit`` completes in about a minute, so process large sites in tiles; and ``ancillary_data=False`` is
needed without NASA Earthdata credentials.

C2RCC dry run (2026-09-20): the same scene was run through ``run_c2rcc`` end to end against a real ESA SNAP 14
install (``gpt``) for the first time -- completed in well under a minute for the Hyderabad lakes bbox, producing a
45-band GeoTIFF (Rrs per band, Kd, and c2rcc's IOP/uncertainty outputs) with physically plausible ranges. One
non-obvious gotcha: a SAFE product extracted from a Windows-made zip (``Compress-Archive``) can come out with its
``GRANULE``/``DATASTRIP`` directories missing the executable/traverse bit (mode ``644`` instead of ``755``), which
SNAP's Sentinel-2 reader reports as a plain "No product reader found for file ..." with no mention of permissions --
run ``chmod -R u+X`` (or ``find <SAFE> -type d -exec chmod 755 {} +``) on the extracted product if that error shows
up despite the file clearly being a valid SAFE product.
"""
import logging
import os
import shlex
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
        "l2w_export_geotiff=True",          # Rrs GeoTIFFs (what correction_convert reads)
        "dsf_residual_glint_correction=True",
        "l2w_mask=False",                   # the pipeline applies its own MNDWI/NIR water mask; ACOLITE's default
                                            # non-water test (SWIR > 0.0215) removed 6 of 8 small turbid Hyderabad tanks
        "ancillary_data=False",             # no NASA Earthdata login needed; uses default pressure/wind/ozone
    ]) + "\n")
    python = shlex.split(os.environ.get("ACOLITE_PYTHON", sys.executable), posix=os.name != "nt")
    _run([*python, str(Path(home) / "launch_acolite.py"), "--cli", "--settings", str(settings)], timeout)
    if not list(out_dir.glob("*.tif")) and not list(out_dir.glob("*.nc")):
        raise CorrectionError("ACOLITE produced no output")
    return out_dir


def c2rcc_graph_xml() -> str:
    """GPT graph: Read -> Resample -> Subset -> c2rcc.msi -> Write (GeoTIFF).

    Why a graph rather than ``gpt c2rcc.msi ...`` on the command line: c2rcc.msi needs a
    *resampled* L1C product (Sentinel-2 bands have mixed 10/20/60 m grids) and has no region
    parameter, so resampling and subsetting are separate operators. Variables ${input},
    ${output} and ${region} are passed with -P on the gpt command line.

    Verified working (2026-09-20) against a real ESA SNAP 14 install and a real Sentinel-2 L1C SAFE scene.
    """
    return """<graph id="c2rcc_msi">
  <version>1.0</version>
  <node id="Read">
    <operator>Read</operator>
    <parameters><file>${input}</file></parameters>
  </node>
  <node id="Resample">
    <operator>Resample</operator>
    <sources><sourceProduct refid="Read"/></sources>
    <parameters><targetResolution>20</targetResolution></parameters>
  </node>
  <node id="Subset">
    <operator>Subset</operator>
    <sources><sourceProduct refid="Resample"/></sources>
    <parameters><geoRegion>${region}</geoRegion><copyMetadata>true</copyMetadata></parameters>
  </node>
  <node id="C2RCC">
    <operator>c2rcc.msi</operator>
    <sources><sourceProduct refid="Subset"/></sources>
    <parameters><outputAsRrs>true</outputAsRrs><outputKd>true</outputKd></parameters>
  </node>
  <node id="Write">
    <operator>Write</operator>
    <sources><sourceProduct refid="C2RCC"/></sources>
    <parameters><file>${output}</file><formatName>GeoTIFF</formatName></parameters>
  </node>
</graph>
"""


def run_c2rcc(safe_path: Path | str, site: Site, out_dir: Path | str = INTERIM / "c2rcc",
              timeout: int = 3600) -> Path:
    """SNAP c2rcc.msi neural-network inversion (tier 2: hypereutrophic / CDOM-rich). Verified working 2026-09-20."""
    gpt = os.environ.get("SNAP_GPT", "gpt")
    out_dir = Path(out_dir) / site.name
    out_dir.mkdir(parents=True, exist_ok=True)
    w, s, e, n = site.bbox
    wkt = f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"
    graph = out_dir / "c2rcc_msi_graph.xml"
    graph.write_text(c2rcc_graph_xml())
    out = out_dir / f"{Path(safe_path).stem}_c2rcc.tif"
    _run([gpt, str(graph), f"-Pinput={safe_path}", f"-Poutput={out}", f"-Pregion={wkt}"], timeout)
    if not out.exists():
        raise CorrectionError("C2RCC produced no output")
    return out


def correct_scene(safe_path: Path | str, site: Site, method: str = "auto") -> dict:
    """Run the correction tier(s) for a site and, for ACOLITE, produce a pipeline-ready raster.

    Routing is a STATIC site-type rule, not per-pixel dynamic routing: rivers -> ACOLITE DSF; lakes and
    reservoirs -> ACOLITE and C2RCC. Returns ``{'method', 'outputs', 'correction_method'}``:

    * after a successful ACOLITE run ``contract_tif`` is a ``{site}_S2_{YYYYMMDD}.tif`` in the band contract
      (see correction_convert.py); a failed conversion is reported in ``conversion_error``;
    * when a tool fails, ``method`` is ``'gee_sr_fallback'`` and the caller should use
      ``src.acquisition.gee.fetch_scenes`` surface reflectance instead.
    """
    from src.preprocessing.correction_convert import ConversionError, acolite_to_contract, date_from_safe

    tiers = {"acolite": [run_acolite], "c2rcc": [run_c2rcc],
             "auto": [run_acolite] if site.type == "river" else [run_acolite, run_c2rcc]}[method]
    outputs = []
    try:
        for fn in tiers:
            outputs.append(str(fn(safe_path, site)))
    except CorrectionError as e:
        log.warning("correction failed for %s (%s); falling back to GEE surface reflectance", site.name, e)
        return {"method": "gee_sr_fallback", "correction_method": "gee_sr_fallback", "outputs": outputs, "error": str(e)}
    name = "+".join(f.__name__.removeprefix("run_") for f in tiers)
    result = {"method": name, "correction_method": name, "outputs": outputs}
    if run_acolite in tiers:
        try:
            dest = INTERIM / f"{site.name}_S2_{date_from_safe(safe_path)}.tif"
            result["contract_tif"] = str(acolite_to_contract(outputs[0], dest))
        except ConversionError as e:
            result["conversion_error"] = str(e)
    return result
