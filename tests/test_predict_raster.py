import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds

from src.data import sensors
from src.preprocessing.predict_raster import NODATA, predict_arrays, predict_raster

H = W = 12


def s2_layers(seed=0):
    rng = np.random.default_rng(seed)
    return {b: rng.uniform(0.02, 0.09, (H, W)) for b in sensors.BANDS["S2"]}


def fake_model(feats: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"do": 5.0 + 10 * feats["ndci"].to_numpy(), "bod": np.full(len(feats), 7.0)}, index=feats.index)


def test_predictions_only_on_water_and_nan_elsewhere():
    bands = s2_layers()
    water = np.zeros((H, W), dtype=bool)
    water[:, :6] = True
    out = predict_arrays(bands, water, fake_model)
    assert set(out) == {"do", "bod", "idx_ndci"}
    assert np.isfinite(out["do"][:, :6]).all() and np.isnan(out["do"][:, 6:]).all()
    ndci = (bands["B5"] - bands["B4"]) / (bands["B5"] + bands["B4"])
    assert out["do"][0, 0] == pytest.approx(5.0 + 10 * ndci[0, 0], rel=1e-3)
    assert out["idx_ndci"][3, 3] == pytest.approx(ndci[3, 3], rel=1e-3)


def test_nodata_and_nan_pixels_are_excluded():
    bands = s2_layers()
    bands["B4"][0, 0] = NODATA
    bands["B8"][1, 1] = np.nan
    out = predict_arrays(bands, np.ones((H, W), dtype=bool), fake_model)
    assert np.isnan(out["do"][0, 0]) and np.isnan(out["do"][1, 1]) and np.isfinite(out["do"][2, 2])


def test_missing_bands_raise_a_clear_error_for_rivers_that_dropped_red_edge():
    bands = {k: v for k, v in s2_layers().items() if k not in ("B5", "B11")}
    with pytest.raises(KeyError, match="B5"):
        predict_arrays(bands, np.ones((H, W), dtype=bool), fake_model)


def test_landsat_b5_is_never_treated_as_red_edge():
    ls = {b: np.full((H, W), v) for b, v in zip(sensors.BANDS["LS"], (0.05, 0.04, 0.30, 0.01))}     # B5 = NIR
    seen = {}

    def capture(feats):
        seen["cols"] = set(feats.columns)
        return pd.DataFrame({"turbidity": feats["turbidity_empirical"]}, index=feats.index)

    out = predict_arrays(ls, np.ones((H, W), dtype=bool), capture, sensor="LS", extra_layers=("ndci",))
    assert "ndci" not in seen["cols"] and "idx_ndci" not in out and "turbidity" in out


def test_smoothing_reduces_speckle_without_touching_the_mask():
    bands = s2_layers(1)
    water = np.zeros((H, W), dtype=bool)
    water[2:10, 2:10] = True
    raw = predict_arrays(bands, water, fake_model)["do"]
    smooth = predict_arrays(bands, water, fake_model, smooth_px=3)["do"]
    assert np.isnan(smooth[0, 0]) and np.isfinite(smooth[5, 5])
    assert np.nanstd(smooth) < np.nanstd(raw)


def test_no_water_gives_no_layers():
    assert predict_arrays(s2_layers(), np.zeros((H, W), dtype=bool), fake_model) == {}


def test_predict_raster_roundtrip_with_contract_band_order(tmp_path):
    layers = s2_layers(2)
    water = np.zeros((H, W))
    water[:, :6] = 1
    stack = np.stack([layers[b] for b in sensors.BANDS["S2"]] + [water]).astype("float32")
    path = tmp_path / "interim" / "site_S2_20200101.tif"
    path.parent.mkdir()
    with rasterio.open(path, "w", driver="GTiff", height=H, width=W, count=stack.shape[0], dtype="float32",
                       crs="EPSG:4326", transform=from_bounds(77, 12, 77.1, 12.1, W, H)) as dst:
        dst.write(stack)
    outs = predict_raster(path, fake_model, sensor="S2", out_dir=tmp_path / "pred")
    assert set(outs) == {"do", "bod", "idx_ndci"}
    with rasterio.open(outs["bod"]) as src:
        data = src.read(1)
        assert src.nodata == NODATA and src.crs.to_string() == "EPSG:4326"
    assert (data[:, :6] == 7.0).all() and (data[:, 6:] == NODATA).all()


def test_wrong_band_count_is_rejected(tmp_path):
    path = tmp_path / "bad.tif"
    with rasterio.open(path, "w", driver="GTiff", height=H, width=W, count=4, dtype="float32", crs="EPSG:4326",
                       transform=from_bounds(0, 0, 1, 1, W, H)) as dst:
        dst.write(np.ones((4, H, W), dtype="float32"))
    with pytest.raises(ValueError, match="contract expects 8"):
        predict_raster(path, fake_model, sensor="S2", out_dir=tmp_path / "o")
