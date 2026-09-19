import math

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from src.data import sensors
from src.preprocessing.correction_convert import (
    ConversionError,
    acolite_to_contract,
    date_from_safe,
    read_provenance,
)

H = W = 8
WAVELENGTHS = {492: 0.004, 560: 0.008, 665: 0.005, 704: 0.009, 740: 0.007, 833: 0.002, 1614: 0.001}   # Rrs (sr^-1)


def write_band(directory, wl, value, prefix="S2A_MSI_2020_01_29_05_10_41_T43PGQ_L2W_Rrs_", shape=(H, W), nodata=None):
    arr = np.full(shape, value, dtype="float32")
    with rasterio.open(directory / f"{prefix}{wl}.tif", "w", driver="GTiff", height=shape[0], width=shape[1], count=1,
                       dtype="float32", crs="EPSG:32643", transform=from_bounds(780000, 1430000, 780080, 1430080, *shape[::-1]),
                       nodata=nodata) as dst:
        dst.write(arr, 1)


def make_dir(tmp_path, drop=(), **kw):
    for wl, v in WAVELENGTHS.items():
        if wl not in drop:
            write_band(tmp_path, wl, v, **kw)
    return tmp_path


def test_stack_follows_the_contract_order_units_and_provenance(tmp_path):
    out = acolite_to_contract(make_dir(tmp_path), tmp_path / "site_S2_20200129.tif")
    with rasterio.open(out) as src:
        data = src.read()
        assert src.count == len(sensors.OUT_BANDS["S2"]) == 8
    for i, band in enumerate(sensors.BANDS["S2"]):
        center = {"B2": 492, "B3": 560, "B4": 665, "B5": 704, "B6": 740, "B8": 833, "B11": 1614}[band]
        assert data[i, 0, 0] == pytest.approx(WAVELENGTHS[center] * math.pi, rel=1e-5)     # Rrs * pi
    assert (data[-1] == 1).all()                                          # green >> SWIR and low NIR -> water
    tags = read_provenance(out)
    assert tags["CORRECTION_METHOD"] == "acolite_dsf" and "pi" in tags["UNITS"] and tags["BANDS"].endswith("water")


def test_nearest_wavelength_is_used_within_tolerance(tmp_path):
    d = tmp_path
    for wl, v in {490: 0.004, 559: 0.008, 664: 0.005, 703: 0.009, 739: 0.007, 832: 0.002, 1610: 0.001}.items():
        write_band(d, wl, v)
    out = acolite_to_contract(d, tmp_path / "x.tif")
    with rasterio.open(out) as src:
        assert src.read(3)[0, 0] == pytest.approx(0.005 * math.pi, rel=1e-5)         # B4 came from the 664 nm file


def test_missing_swir_raises_instead_of_guessing(tmp_path):
    with pytest.raises(ConversionError, match="B11"):
        acolite_to_contract(make_dir(tmp_path, drop=(1614,)), tmp_path / "x.tif")


def test_no_files_and_grid_mismatch_are_reported(tmp_path):
    with pytest.raises(ConversionError, match="no '"):
        acolite_to_contract(tmp_path, tmp_path / "x.tif")
    make_dir(tmp_path)
    write_band(tmp_path, 665, 0.005, shape=(H + 2, W + 2))
    with pytest.raises(ConversionError, match="different grid"):
        acolite_to_contract(tmp_path, tmp_path / "x.tif")


def test_nodata_pixels_stay_nodata_and_are_not_water(tmp_path):
    d = make_dir(tmp_path)
    write_band(d, 665, -32768.0, nodata=-32768.0)
    out = acolite_to_contract(d, tmp_path / "x.tif")
    with rasterio.open(out) as src:
        data = src.read()
    assert (data[2] == -9999.0).all() and (data[-1] == 1).all()          # missing B4 marked as nodata; mask uses B3/B8/B11


def test_result_feeds_predict_raster_without_error(tmp_path):
    import pandas as pd
    from src.preprocessing.predict_raster import predict_raster
    out = acolite_to_contract(make_dir(tmp_path), tmp_path / "interim" / "s_S2_20200129.tif")
    maps = predict_raster(out, lambda f: pd.DataFrame({"do": 5.0}, index=f.index), sensor="S2", out_dir=tmp_path / "p")
    assert "do" in maps                                                   # every band is finite here, so all pixels are valid


def test_date_from_safe_name():
    assert date_from_safe("S2A_MSIL1C_20200129T051041_N0208_R019_T43PGQ_20200129T081234.SAFE") == "20200129"
    with pytest.raises(ConversionError):
        date_from_safe("not-a-product")
