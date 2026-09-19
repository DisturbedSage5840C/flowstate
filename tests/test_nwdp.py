"""Offline tests for the CPCB NWDP loader (synthetic CSV fixtures mirroring the real schema)."""

import numpy as np
import pandas as pd
import pytest

from src.data import nwdp

COMMON = {"Station": "S", "Agency": "CPCB", "State": "Karnataka", "District": "D", "River": "-", "Basin": "-",
          "Latitude": 12.9, "Longitude": 77.6, "Data Acquisition Time": "11-01-2021 08:30"}


def chem_row(**kw):
    row = {**COMMON, "Dissolved oxygen (mg/L)": 6.0, "Potential of Hydrogen (pH)": 7.5,
           "Amonia N (mgN/L)": 0.4, "Sodium Adsorption Ratio (%)": "-", "Boron (mg/L)": "-"}
    row.update(kw)
    return row


@pytest.mark.parametrize("name, expected", [
    ("Surface Water Quality Chemical Parameters CPCB Karnataka (2021 - 2025) Manual", ("Karnataka", "2021-2025")),
    ("Surface Water Quality Manual Chemical Parameters CPCB Andhra Pradesh (1961-2020)", ("Andhra Pradesh", "1961-2020")),
    ("Surface Water Quality Manual Biological Parameters CPCB Arunachal Pradesh(1961-2020)", ("Arunachal Pradesh", "1961-2020")),
    ("SWQ_Manual_Chemical_Parameters_CPCB_PB_1961_2020.csv", ("Punjab", "1961-2020")),
    ("SWQ_Manual_Chemical_Parameters_CPCB_UP_1961_2020.csv", ("Uttar Pradesh", "1961-2020")),
])
def test_resource_names_of_every_style_parse(name, expected):
    assert nwdp.parse_resource_name(name) == expected


def test_unknown_resource_name_is_none():
    assert nwdp.parse_resource_name("something else entirely") is None


def test_normalize_renames_and_parses_dayfirst_timestamps():
    df = nwdp.normalize_frame(pd.DataFrame([chem_row(), chem_row(**{"Data Acquisition Time": "18-02-2021 09:15"})]), "chemical")
    assert {"station", "lat", "lon", "timestamp", "do", "ph", "ammonia_n"} <= set(df.columns)
    assert df["timestamp"].iloc[1] == pd.Timestamp("2021-02-18 09:15")
    assert df["do"].tolist() == [6.0, 6.0]
    assert np.isnan(df["sar"].iloc[0])                       # "-" means missing


def test_glued_numbers_become_nan_but_row_survives():
    df = nwdp.normalize_frame(pd.DataFrame([chem_row(**{"Dissolved oxygen (mg/L)": "3.16.2"})]), "chemical")
    assert len(df) == 1 and np.isnan(df["do"].iloc[0]) and df["ph"].iloc[0] == 7.5
    assert df.attrs["qc"]["corrupt_numeric"] == 1


def test_colour_words_in_secondary_columns_null_the_group_but_keep_core_values():
    # real 2020 pattern: ammonia holds "Clear" while DO / pH are valid
    row = chem_row(**{"Amonia N (mgN/L)": "Clear", "Dissolved oxygen (mg/L)": 4.4, "Nitrate N (mgN/L)": 7.0})
    df = nwdp.normalize_frame(pd.DataFrame([chem_row(), row]), "chemical")
    assert len(df) == 2
    assert df["do"].tolist() == [6.0, 4.4] and df["ph"].tolist() == [7.5, 7.5]      # core kept
    assert np.isnan(df["ammonia_n"].iloc[1]) and np.isnan(df["nitrate_n"].iloc[1])   # whole secondary group nulled
    assert df["ammonia_n"].iloc[0] == 0.4                                            # untouched row unaffected
    assert df.attrs["qc"]["secondary_groups_nulled"] == 1 and df.attrs["qc"]["rows_dropped_core_text"] == 0


def test_text_in_a_core_column_drops_the_row():
    df = nwdp.normalize_frame(pd.DataFrame([chem_row(), chem_row(**{"Dissolved oxygen (mg/L)": "Clear"})]), "chemical")
    assert len(df) == 1 and df["do"].iloc[0] == 6.0
    assert df.attrs["qc"]["rows_dropped_core_text"] == 1


def test_null_tokens_do_not_drop_the_row():
    df = nwdp.normalize_frame(pd.DataFrame([chem_row(**{"Amonia N (mgN/L)": "BDL"})]), "chemical")
    assert len(df) == 1 and np.isnan(df["ammonia_n"].iloc[0])


def test_out_of_bounds_values_are_nulled():
    df = nwdp.normalize_frame(pd.DataFrame([chem_row(**{"Dissolved oxygen (mg/L)": 55.0,
                                                          "Potential of Hydrogen (pH)": 0.2})]), "chemical")
    assert np.isnan(df["do"].iloc[0]) and np.isnan(df["ph"].iloc[0])
    assert df.attrs["qc"]["out_of_bounds"] == 2


def test_swapped_and_offshore_coordinates():
    df = pd.DataFrame({"lat": [12.9, 77.6, 0.0, 12.9], "lon": [77.6, 12.9, 0.0, 200.0]})
    out = nwdp.fix_coordinates(df)
    assert out.loc[1, "lat"] == 12.9 and out.loc[1, "lon"] == 77.6 and out.loc[1, "coord_swapped_fixed"]
    assert out["coord_ok"].tolist() == [True, True, False, False]


@pytest.mark.parametrize("station, river, expected", [
    ("BELLANDUR LAKE", "-", "lake"), ("HUSSAIN SAGAR", "-", "lake"), ("KRS DAM RESERVOIR", "-", "reservoir"),
    ("GANGA AT KANPUR", "Ganga", "river"),
    ("BOREWELL AT RAJIV GRUHA NEAR AP PAPER MAILLS", "-", "groundwater"), ("HAND PUMP KATHERU", "-", "groundwater"),
    ("TUBEWELL SECTOR 5", "-", "groundwater"), ("GROUND WATER AT BHUJ", "-", "groundwater"), ("B/W - MANAKONDUR (V)", "-", "groundwater"),
    ("SWELLING RIVER POINT", "River", "river"), ("BUDDHA NALLAH D/S", "-", "river"), ("SOMEWHERE", "-", "unknown"),
])
def test_water_body_heuristic(station, river, expected):
    assert nwdp.classify_water_body(station, river) == expected


def _write(cache, kind, state_slug, rows):
    pd.DataFrame(rows).to_csv(cache / f"{kind}__{state_slug}__2021-2025__abcd1234.csv", index=False)


def test_build_insitu_table_merges_kinds_dedupes_and_filters(tmp_path):
    _write(tmp_path, "chemical", "karnataka", [chem_row(), chem_row(), chem_row(**{"Data Acquisition Time": "05-03-2018 08:30"})])
    _write(tmp_path, "biological", "karnataka", [{**COMMON, "Biochemical Oxygen Demand (mg/L)": 4.0,
                                                  "Chemical Oxygen Demand (mg/L)": 20.0,
                                                  "Fecal Coliform (MPN/100mL)": "-", "Total Coliform (MPN/100mL)": 900}])
    _write(tmp_path, "physical", "karnataka", [{**COMMON, "Electric Conductivity (μS/cm)": 700, "Turbidity (NTU)": 12.5,
                                                "Temperature (ºC)": "-", "Total Solids (mg/L)": "-"}])
    t = nwdp.build_insitu_table(cache_dir=tmp_path, since="2019-01-01")
    assert len(t) == 1                                        # duplicate merged, 2018 row filtered
    r = t.iloc[0]
    assert (r["do"], r["bod"], r["turbidity"], r["conductivity"], r["total_coliform"]) == (6.0, 4.0, 12.5, 700.0, 900.0)
    assert r["date"] == pd.Timestamp("2021-01-11") and r["state"] == "Karnataka" and not r["is_proxy"]


def test_rows_present_in_only_one_kind_are_kept(tmp_path):
    _write(tmp_path, "chemical", "karnataka", [chem_row()])
    _write(tmp_path, "biological", "karnataka", [{**COMMON, "Data Acquisition Time": "20-05-2021 08:30",
                                                  "Biochemical Oxygen Demand (mg/L)": 9.0, "Chemical Oxygen Demand (mg/L)": "-",
                                                  "Fecal Coliform (MPN/100mL)": "-", "Total Coliform (MPN/100mL)": "-"}])
    t = nwdp.build_insitu_table(cache_dir=tmp_path, since="2019-01-01")
    assert len(t) == 2 and t["state"].notna().all()
    assert t["bod"].notna().sum() == 1 and t["do"].notna().sum() == 1
