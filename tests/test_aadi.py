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
