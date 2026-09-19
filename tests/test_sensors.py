import pytest

from src.data import sensors


@pytest.mark.parametrize("name, code", [("S2", "S2"), ("Sentinel-2", "S2"), ("sentinel_2", "S2"), ("S2A", "S2"),
                                         ("LS", "LS"), ("L8", "LS"), ("L9", "LS"), ("Landsat-8", "LS"), ("landsat_9", "LS")])
def test_every_spelling_maps_to_one_code(name, code):
    assert sensors.normalize_sensor(name) == code


def test_unknown_sensor_raises_but_tolerance_has_a_default():
    with pytest.raises(ValueError):
        sensors.normalize_sensor("MODIS")
    assert sensors.day_tolerance("MODIS") == 3


def test_tolerances_follow_the_plan():
    assert sensors.day_tolerance("S2") == 3 and sensors.day_tolerance("LS") == 5


def test_export_contract_and_landsat_limits():
    assert sensors.OUT_BANDS["S2"] == ["B2", "B3", "B4", "B5", "B6", "B8", "B11", "water"]
    assert sensors.OUT_BANDS["LS"] == ["B3", "B4", "B5", "B6", "water"]
    assert sensors.SUPPORTS_CHL_A == {"S2": True, "LS": False}


def test_gee_and_features_use_the_shared_contract():
    from src.acquisition import gee
    from src.features import feature_engineering as fe
    assert gee.OUT_BANDS is sensors.OUT_BANDS
    assert fe.S2_REQUIRED == sensors.REQUIRED_FOR_FEATURES["S2"]


def test_validate_bands_names_the_missing_ones():
    sensors.validate_bands("S2", ["B3", "B4", "B5", "B8", "B11", "B2"])
    with pytest.raises(KeyError, match="B5"):
        sensors.validate_bands("S2", ["B2", "B3", "B4", "B8"])           # a river export that dropped B5/B11
    with pytest.raises(KeyError, match="B6"):
        sensors.validate_bands("LS", ["B3", "B4", "B5"])
