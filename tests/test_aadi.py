import numpy as np
import pytest

from src.acquisition.sites import load_sites
from src.preprocessing.masking import otsu_threshold


def test_all_configured_sites_valid():
    sites = load_sites()
    assert len(sites) >= 8
    assert {s.type for s in sites.values()} >= {"lake", "river"}
    for s in sites.values():
        lat, lon = s.centroid()
        assert 6 < lat < 37 and 68 < lon < 98, s.name  # inside India's extent
        ring = s.geometry()["coordinates"][0]
        assert ring[0] == ring[-1]


def test_bad_type_rejected(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("sites:\n  - {name: x, type: ocean, bbox: [0, 0, 1, 1]}\n")
    with pytest.raises(ValueError):
        load_sites(p)


def test_otsu_separates_bimodal():
    rng = np.random.default_rng(0)
    land = rng.normal(-0.4, 0.08, 2000)
    water = rng.normal(0.4, 0.08, 2000)
    t = otsu_threshold(np.concatenate([land, water]))
    assert -0.3 < t < 0.3


# ---- per-site MNDWI thresholds, altitude, B6 export ------------------------------------

def test_site_altitude_loaded_and_plausible():
    sites = load_sites()
    assert sites["dal_lake"].altitude_m > 1000            # Kashmir valley
    assert sites["chilika"].altitude_m == 0
    assert all(0 <= s.altitude_m < 4000 for s in sites.values())


def test_mndwi_threshold_precedence(tmp_path):
    from src.acquisition import gee
    sites_yaml = tmp_path / "s.yaml"
    sites_yaml.write_text(
        "sites:\n"
        "  - {name: a, type: lake, bbox: [0, 0, 1, 1]}\n"
        "  - {name: b, type: lake, bbox: [0, 0, 1, 1], mndwi_threshold: 0.25}\n"
        "  - {name: c, type: lake, bbox: [0, 0, 1, 1]}\n")
    tuned = tmp_path / "t.yaml"
    tuned.write_text("thresholds:\n  a: {threshold: -0.1, method: otsu}\n  b: {threshold: 0.9}\n")
    sites = load_sites(sites_yaml, tuned)
    assert sites["a"].mndwi_threshold == -0.1              # from tuned file
    assert sites["b"].mndwi_threshold == 0.25              # inline value wins over tuned file
    assert sites["c"].mndwi_threshold is None
    assert gee.resolve_mndwi_threshold(sites["a"]) == -0.1
    assert gee.resolve_mndwi_threshold(sites["a"], override=0.3) == 0.3
    assert gee.resolve_mndwi_threshold(sites["c"]) == 0.0  # global default


def test_bad_mndwi_threshold_rejected(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("sites:\n  - {name: x, type: lake, bbox: [0, 0, 1, 1], mndwi_threshold: 3}\n")
    with pytest.raises(ValueError):
        load_sites(p, None)


def test_export_band_contract_includes_b6_and_rivers_keep_red_edge():
    from src.acquisition import gee
    from src.preprocessing import masking
    assert gee.OUT_BANDS["S2"] == ["B2", "B3", "B4", "B5", "B6", "B8", "B11", "water"]
    assert "B6" in masking.S2_BANDS                        # so it is reflectance-scaled like the others
    assert not hasattr(gee, "RIVER_S2_BANDS")              # rivers no longer lose B5/B11 (NDCI needs B5)


def test_c2rcc_graph_is_wellformed_and_uses_documented_structure():
    import xml.etree.ElementTree as ET
    from src.preprocessing.correction import c2rcc_graph_xml
    root = ET.fromstring(c2rcc_graph_xml())
    ops = [n.findtext("operator") for n in root.findall("node")]
    assert ops == ["Read", "Resample", "Subset", "c2rcc.msi", "Write"]        # resample BEFORE c2rcc
    c2 = [n for n in root.findall("node") if n.findtext("operator") == "c2rcc.msi"][0]
    assert c2.find("parameters/outputAsRrs").text == "true"
    text = c2rcc_graph_xml()
    assert "${input}" in text and "${output}" in text and "${region}" in text


def test_correct_scene_converts_acolite_output_and_falls_back_on_failure(tmp_path, monkeypatch):
    from src.acquisition.sites import Site
    from src.preprocessing import correction
    site = Site("lake_x", "lake", "K", (77.0, 12.0, 77.1, 12.1))
    monkeypatch.setattr(correction, "INTERIM", tmp_path)

    def fake_acolite(safe, s):
        return tmp_path / "acolite_out"

    def fake_c2rcc(safe, s):
        return tmp_path / "c2.tif"

    monkeypatch.setattr(correction, "run_acolite", fake_acolite)
    monkeypatch.setattr(correction, "run_c2rcc", fake_c2rcc)
    import src.preprocessing.correction_convert as conv
    monkeypatch.setattr(conv, "acolite_to_contract", lambda d, dest: dest)
    res = correction.correct_scene("S2A_MSIL1C_20200129T051041_N0208_R019_T43PGQ.SAFE", site, method="acolite")
    assert res["method"] == "fake_acolite" and res["correction_method"] == "fake_acolite"
    assert res["contract_tif"].endswith("lake_x_S2_20200129.tif")

    def boom(safe, s):
        raise correction.CorrectionError("java missing")

    monkeypatch.setattr(correction, "run_acolite", boom)
    res = correction.correct_scene("S2A_MSIL1C_20200129T051041_N0208_R019_T43PGQ.SAFE", site, method="acolite")
    assert res["method"] == "gee_sr_fallback" and "java missing" in res["error"]
