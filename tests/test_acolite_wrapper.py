"""run_acolite's settings file and interpreter handling, checked without ACOLITE installed.

These guard problems found by the real ACOLITE dry run (reports/real/acolite_dry_run.json): a settings file without
a trailing newline silently corrupts the next appended key; GeoTIFF export of the L2W (Rrs) product is what the
converter reads; ACOLITE's default non-water mask removed most small turbid inland tanks; and ACOLITE needs GDAL,
so the interpreter must be configurable.
"""
from pathlib import Path

from src.acquisition.sites import Site
from src.preprocessing import correction


def _run_with_fake(tmp_path, monkeypatch, **env):
    home = tmp_path / "acolite"
    home.mkdir()
    (home / "launch_acolite.py").write_text("")
    monkeypatch.setenv("ACOLITE_HOME", str(home))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    calls = []

    def fake_run(cmd, timeout):
        calls.append(cmd)
        out = Path(tmp_path / "out" / "lake")
        (out / "x_L2W_Rrs_665.tif").write_text("")

    monkeypatch.setattr(correction, "_run", fake_run)
    site = Site(name="lake", type="lake", state="X", bbox=(78.0, 17.0, 78.1, 17.1))
    correction.run_acolite("scene.SAFE", site, out_dir=tmp_path / "out")
    return calls[0], (tmp_path / "out" / "lake" / "acolite_settings.txt").read_text()


def test_settings_are_complete_and_newline_terminated(tmp_path, monkeypatch):
    (tmp_path / "out" / "lake").mkdir(parents=True)
    _, text = _run_with_fake(tmp_path, monkeypatch)
    assert text.endswith("\n")
    lines = text.splitlines()
    for required in ("l2w_export_geotiff=True", "l2w_mask=False", "ancillary_data=False",
                     "atmospheric_correction_method=dark_spectrum", "limit=17.0,78.0,17.1,78.1"):
        assert required in lines
    assert all("=" in ln and ln.count("=") == 1 for ln in lines)  # no two settings glued onto one line


def test_acolite_python_command_is_split(tmp_path, monkeypatch):
    (tmp_path / "out" / "lake").mkdir(parents=True)
    cmd, _ = _run_with_fake(tmp_path, monkeypatch, ACOLITE_PYTHON="mm run -n acolite python")
    assert cmd[:5] == ["mm", "run", "-n", "acolite", "python"]
    assert cmd[5].endswith("launch_acolite.py") and "--cli" in cmd
