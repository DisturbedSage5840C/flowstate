"""Selecting which in-situ visits to match with satellite scenes."""

from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_COLS = ("do", "bod", "turbidity")


def select_visits(table: pd.DataFrame, per_state: int = 100, per_station: int = 6,
                  start: str = "2019-01-01", end: str = "2021-12-31", turbidity_share: float = 0.5,
                  exclude_types: tuple[str, ...] = ("groundwater",), seed: int = 0) -> pd.DataFrame:
    """Spatially diverse, capped sample of surface-water visits that carry at least one target label.

    * groundwater points are dropped (they cannot be seen from space);
    * at most ``per_station`` visits per station and ``per_state`` per state, so no region dominates;
    * ``turbidity_share`` of each state's quota is taken from visits with a turbidity reading
      (the scarcest label) when enough exist.
    """
    rng = np.random.default_rng(seed)
    t = table[~table["water_body_type"].isin(exclude_types)].copy()
    t = t[(t["date"] >= pd.Timestamp(start)) & (t["date"] <= pd.Timestamp(end))]
    t = t[t[list(LABEL_COLS)].notna().any(axis=1)]
    t = t.drop_duplicates(["station", "date"])

    # cap per station first (random visits, not the first ones)
    t = t.assign(_r=rng.random(len(t))).sort_values("_r")
    t = t.groupby("station", group_keys=False).head(per_station)

    picked = []
    for _state, g in t.groupby("state"):
        if len(g) <= per_state:
            picked.append(g)
            continue
        with_turb = g[g["turbidity"].notna()]
        n_turb = min(int(round(per_state * turbidity_share)), len(with_turb))
        first = with_turb.head(n_turb)
        rest = g.drop(first.index).head(per_state - n_turb)
        picked.append(pd.concat([first, rest]))
    out = pd.concat(picked) if picked else t.iloc[:0]
    return out.drop(columns="_r").sort_values(["state", "station", "date"]).reset_index(drop=True)


def select_dense_visits(table: pd.DataFrame, n_stations: int = 150, min_visits: int = 12,
                        start: str = "2019-01-01", end: str = "2021-12-31", per_state_cap: int = 8,
                        exclude_types: tuple[str, ...] = ("groundwater",), seed: int = 0) -> pd.DataFrame:
    """ALL visits of stations with regular sampling, so each station has a usable history for temporal models.

    Stations qualify with at least ``min_visits`` labelled visits in the window; at most ``per_state_cap``
    stations per state are taken (random within a state) so no region dominates.
    """
    rng = np.random.default_rng(seed)
    t = table[~table["water_body_type"].isin(exclude_types)].copy()
    t = t[(t["date"] >= pd.Timestamp(start)) & (t["date"] <= pd.Timestamp(end))]
    t = t[t[list(LABEL_COLS)].notna().any(axis=1)].drop_duplicates(["station", "date"])
    counts = t.groupby("station").size()
    eligible = counts[counts >= min_visits].index
    st = t[t["station"].isin(eligible)].groupby("station").agg(state=("state", "first")).reset_index()
    st = st.assign(_r=rng.random(len(st))).sort_values("_r")
    chosen = st.groupby("state", group_keys=False).head(per_state_cap).head(n_stations)["station"]
    return t[t["station"].isin(chosen)].sort_values(["state", "station", "date"]).reset_index(drop=True)
