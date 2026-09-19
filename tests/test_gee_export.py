import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from src.acquisition import gee
from src.acquisition.sites import Site, load_sites


def test_estimate_scales_with_area_resolution_and_bands():
    small = gee.estimate_export_bytes((77.63, 12.90, 77.70, 12.96), 10, 8)
    assert 12e6 < small < 22e6                                   # Bellandur at 10 m x 8 float32 bands: ~16 MB
    big = gee.estimate_export_bytes((80.2, 26.35, 80.45, 26.55), 10, 8)
    assert big > 8 * small                                       # a Kanpur river stretch is ~176 MB
    assert gee.estimate_export_bytes((77.63, 12.90, 77.70, 12.96), 20, 8) == pytest.approx(small / 4, rel=0.02)


def test_most_configured_sites_need_tiling_at_10m_and_every_tile_fits():
    sites = load_sites()
    n_bands = len(gee.OUT_BANDS["S2"])
    tiled = 0
    for site in sites.values():
        tiles = gee.tile_bounds(site.bounds(), 10, n_bands)
        tiled += len(tiles) > 1
        assert all(gee.estimate_export_bytes(t, 10, n_bands) <= gee.MAX_DOWNLOAD_BYTES for t in tiles), site.name
    assert tiled == 7                                             # 7 of 13 sites (e.g. Chilika ~750 MB) would have failed un-tiled


def test_tiles_cover_the_bounds_exactly_without_overlap():
    b = (80.2, 26.35, 80.45, 26.55)
    tiles = gee.tile_bounds(b, 10, 8)
    assert len(tiles) > 1
    assert min(t[0] for t in tiles) == b[0] and max(t[2] for t in tiles) == b[2]
    assert min(t[1] for t in tiles) == b[1] and max(t[3] for t in tiles) == b[3]
    area = sum((t[2] - t[0]) * (t[3] - t[1]) for t in tiles)
    assert area == pytest.approx((b[2] - b[0]) * (b[3] - b[1]), rel=1e-9)


def test_small_scene_is_a_single_tile():
    assert len(gee.tile_bounds((77.61, 12.97, 77.64, 12.99), 20, 8)) == 1


def test_site_bounds_from_bbox_and_geojson(tmp_path, monkeypatch):
    assert Site("a", "lake", "X", (1, 2, 3, 4)).bounds() == (1, 2, 3, 4)
    gj = tmp_path / "s.geojson"
    gj.write_text('{"type":"Polygon","coordinates":[[[70,10],[72,10],[72,12],[70,12],[70,10]]]}')
    from src.acquisition import sites as sites_mod
    monkeypatch.setattr(sites_mod, "ROOT", tmp_path)
    assert Site("b", "lake", "X", None, "s.geojson").bounds() == (70, 10, 72, 12)


def test_merge_tiles_produces_one_mosaic(tmp_path):
    paths = []
    for i, (x0, val) in enumerate(((0.0, 1.0), (1.0, 2.0))):
        p = tmp_path / f"t{i}.tif"
        with rasterio.open(p, "w", driver="GTiff", height=10, width=10, count=2, dtype="float32", crs="EPSG:4326",
                           transform=from_bounds(x0, 0, x0 + 1, 1, 10, 10), nodata=-9999) as dst:
            dst.write(np.full((2, 10, 10), val, dtype="float32"))
        paths.append(p)
    out = gee.merge_tiles(paths, tmp_path / "merged.tif")
    with rasterio.open(out) as src:
        data = src.read()
    assert data.shape == (2, 10, 20) and (data[:, :, :10] == 1.0).all() and (data[:, :, 10:] == 2.0).all()
