import numpy as np
import pandas as pd

from src.data.insitu import select_visits


def _table():
    rows = []
    for state, n_st in (("A", 12), ("B", 2)):
        for s in range(n_st):
            for d in range(10):
                rows.append({"state": state, "station": f"{state}{s}", "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=15 * d),
                             "water_body_type": "lake", "do": 5.0, "bod": 2.0,
                             "turbidity": 10.0 if d % 2 == 0 else np.nan})
    rows.append({"state": "A", "station": "GW", "date": pd.Timestamp("2020-02-01"), "water_body_type": "groundwater",
                 "do": 5.0, "bod": 1.0, "turbidity": 1.0})
    rows.append({"state": "A", "station": "NOLABEL", "date": pd.Timestamp("2020-02-01"), "water_body_type": "river",
                 "do": np.nan, "bod": np.nan, "turbidity": np.nan})
    rows.append({"state": "A", "station": "OLD", "date": pd.Timestamp("2015-02-01"), "water_body_type": "river",
                 "do": 5.0, "bod": 1.0, "turbidity": 1.0})
    return pd.DataFrame(rows)


def test_caps_exclusions_and_date_window():
    out = select_visits(_table(), per_state=20, per_station=3, seed=1)
    assert out.groupby("station").size().max() <= 3
    assert out.groupby("state").size().max() <= 20
    assert not {"GW", "NOLABEL", "OLD"} & set(out["station"])
    assert out["date"].between("2019-01-01", "2021-12-31").all()


def test_small_states_are_kept_whole_and_turbidity_is_oversampled():
    out = select_visits(_table(), per_state=20, per_station=6, turbidity_share=0.5, seed=1)
    assert len(out[out.state == "B"]) == 12 or len(out[out.state == "B"]) == 2 * 6
    a = out[out.state == "A"]
    assert a["turbidity"].notna().sum() >= 10          # half the quota requested from turbidity visits


def test_deterministic_for_a_seed():
    a = select_visits(_table(), per_state=8, per_station=2, seed=5)
    b = select_visits(_table(), per_state=8, per_station=2, seed=5)
    pd.testing.assert_frame_equal(a, b)


def test_dense_selection_takes_all_visits_of_regular_stations():
    from src.data.insitu import select_dense_visits
    t = _table()                                   # A0..A11 and B0..B1 all have 10 visits
    t = pd.concat([t, pd.DataFrame([{"state": "A", "station": "SPARSE", "date": pd.Timestamp("2020-03-01"),
                                     "water_body_type": "lake", "do": 5.0, "bod": 2.0, "turbidity": np.nan}])])
    out = select_dense_visits(t, n_stations=100, min_visits=10, per_state_cap=3, seed=2)
    assert "SPARSE" not in set(out["station"])                       # too few visits
    assert out.groupby("station").size().eq(10).all()                # every visit of a chosen station is kept
    assert out.groupby("state")["station"].nunique().max() <= 3      # per-state cap
    assert {"GW", "NOLABEL", "OLD"}.isdisjoint(out["station"])


def test_dense_selection_respects_total_station_limit():
    from src.data.insitu import select_dense_visits
    out = select_dense_visits(_table(), n_stations=4, min_visits=5, per_state_cap=10, seed=0)
    assert out["station"].nunique() == 4
