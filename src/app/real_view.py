"""Streamlit view for the REAL data: CPCB in-situ measurements matched to Sentinel-2 scenes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.app.map_utils import NO_DATA_COLOR, PARAM_RANGES, WQI_CLASSES, _param_color, _wqi_color
from src.app.real_data import (
    PARAM_LABELS,
    REAL_DIR,
    RealBundle,
    complete_class_count,
    filter_visits,
    latest_per_station,
    metric_pivot,
    share_meeting_class,
    tier_counts,
    with_display_columns,
)

try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:            # dashboard still works with a plotly map
    HAS_FOLIUM = False

PARAM_RANGES.setdefault("bod", (0, 30))
UNITS = {"do": "mg/L", "bod": "mg/L", "turbidity": "NTU", "wqi": ""}


def _fmt(v, digits=2) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{digits}f}"


def render(bundle: RealBundle) -> None:
    df_all = bundle.table
    st.warning(
        "**Real data.** Water-quality values are CPCB in-situ grab samples (National Water Data Portal) "
        "matched to Sentinel-2 L2A reflectance within ±3 days. Chlorophyll-a is **not** shown as measured: "
        "CPCB does not measure it, so it has no ground truth here. Model numbers are out-of-fold "
        "(stations never seen in training).",
        icon="ℹ️",
    )

    with st.sidebar:
        st.title("💧 Aqua-Sense")
        st.caption("Real CPCB measurements · Sentinel-2")
        states = st.multiselect("State / UT", sorted(df_all["state"].dropna().unique()), key="rv_states")
        types = st.multiselect("Water-body type", sorted(df_all["water_body_type"].unique()), key="rv_types")
        dmin, dmax = df_all["date"].min().date(), df_all["date"].max().date()
        start, end = st.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax, key="rv_dates") \
            if dmin != dmax else (dmin, dmax)
        options = {"Measured (in-situ)": "measured"}
        if bundle.has_predictions:
            options["XGBoost (out-of-fold)"] = "xgboost"
        if bundle.has_dl_predictions:
            options["DL model (out-of-fold)"] = "dl"
        source_label = st.radio("Values shown", list(options), key="rv_source")
        source = options[source_label]
        parameter = st.radio("Parameter", list(PARAM_LABELS), format_func=PARAM_LABELS.get, key="rv_param")

    df = filter_visits(df_all, states or None, types or None, start, end)
    if df.empty:
        st.info("No visits match the current filters.")
        return
    shown = with_display_columns(df, source)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stations", f"{shown['site'].nunique():,}")
    c2.metric("Matched visits", f"{len(shown):,}")
    c3.metric("Median DO", f"{_fmt(shown['do'].median())} mg/L", help="CPCB Class B needs ≥ 5 mg/L")
    share = share_meeting_class(df)
    n_complete = complete_class_count(df)
    c4.metric("Meeting CPCB class A–C", "n/a" if not np.isfinite(share) else f"{share:.0%}",
              help=f"Only visits where every criterion of the assigned class was measured ({n_complete:,} of "
                   f"{len(df):,} visits). Coliform is missing for most visits, so class A-C cannot be confirmed for the rest.")

    tab_map, tab_station, tab_aoi, tab_perf, tab_shap, tab_data = st.tabs(
        ["🗺️ Map", "📍 Station", "🛰️ Any-AOI map", "📊 Model performance", "🔍 Interpretability",
         "🧾 Data & provenance"])

    with tab_map:
        _map_tab(shown, parameter, source)
    with tab_station:
        _station_tab(bundle, df)
    with tab_aoi:
        _aoi_tab(bundle)
    with tab_perf:
        _performance_tab(bundle)
    with tab_shap:
        _shap_tab()
    with tab_data:
        _data_tab(bundle)


def _marker_color(value: float, parameter: str) -> str:
    if not np.isfinite(value):
        return NO_DATA_COLOR
    return _wqi_color(value) if parameter == "wqi" else _param_color(value, parameter)


def _map_tab(shown: pd.DataFrame, parameter: str, source: str) -> None:
    latest = latest_per_station(shown)
    latest = latest[latest[parameter].notna()] if parameter in latest.columns else latest.iloc[0:0]
    st.caption(f"Latest {source} value per station ({len(latest):,} stations with a {PARAM_LABELS[parameter]} value).")
    if latest.empty:
        st.info("No values for this parameter in the current selection.")
        return
    if HAS_FOLIUM:
        m = folium.Map(location=[22.5, 80.0], zoom_start=5, tiles="OpenStreetMap", prefer_canvas=True)
        for _, r in latest.iterrows():
            tip = (f"<b>{r['site']}</b><br>{r['state']} · {r['water_body_type']}<br>{r['date']:%Y-%m-%d}<br>"
                   f"{PARAM_LABELS[parameter]}: {_fmt(r[parameter])}")
            folium.CircleMarker([r["lat"], r["lon"]], radius=5, weight=0.5, color="white", fill=True,
                                fill_color=_marker_color(float(r[parameter]), parameter), fill_opacity=0.85,
                                tooltip=folium.Tooltip(tip)).add_to(m)
        st_folium(m, width="100%", height=560, returned_objects=[])
    else:
        fig = px.scatter_map(latest, lat="lat", lon="lon", color=parameter, hover_name="site",
                             hover_data={"state": True, "water_body_type": True}, zoom=4,
                             center={"lat": 22.5, "lon": 80.0}, map_style="open-street-map",
                             color_continuous_scale="RdYlGn_r" if parameter != "do" else "RdYlGn")
        st.plotly_chart(fig, width="stretch")

    if parameter == "wqi":
        st.markdown("**WQI tiers** (0 = pristine, 100 = worst): " + " · ".join(
            f"<span style='color:{i['color']}'>●</span> {k} {i['range'][0]:g}–{i['range'][1]:g}"
            for k, i in WQI_CLASSES.items()), unsafe_allow_html=True)
        counts = tier_counts(shown)
        if not counts.empty:
            st.bar_chart(counts.reindex([k for k in WQI_CLASSES if k in counts.index]))


def _station_tab(bundle: RealBundle, df: pd.DataFrame) -> None:
    stations = sorted(df["site"].unique())
    site = st.selectbox("Station", stations, key="rv_station")
    g = df[df["site"] == site].sort_values("date")
    st.caption(f"{g['state'].iloc[0]} · {g['water_body_type'].iloc[0]} · {len(g)} matched visits · "
               f"{g['lat'].iloc[0]:.4f}, {g['lon'].iloc[0]:.4f}")
    param = st.selectbox("Parameter", ["do", "bod", "turbidity"], format_func=PARAM_LABELS.get, key="rv_sparam")
    plot = g[["date", param]].rename(columns={param: "Measured"})
    if f"{param}_pred" in g.columns:
        plot["XGBoost (out-of-fold)"] = g[f"{param}_pred"].to_numpy()
    if f"{param}_pred_dl" in g.columns:
        plot["DL model (out-of-fold)"] = g[f"{param}_pred_dl"].to_numpy()
    long = plot.melt("date", var_name="series", value_name=PARAM_LABELS[param]).dropna()
    if long.empty:
        st.info("No values for this parameter at this station.")
    else:
        st.plotly_chart(px.line(long, x="date", y=PARAM_LABELS[param], color="series", markers=True),
                        width="stretch")
    cols = ["date", "do", "bod", "turbidity", "ph", "cpcb_class", "wqi", "wqi_tier", "day_diff", "n_water_px"]
    st.dataframe(g[[c for c in cols if c in g.columns]].reset_index(drop=True), width="stretch")


AOI_LAYER_LABELS = {
    "idx_ndci": "NDCI (index from reflectance, not a model)",
    "idx_turbidity_empirical": "Turbidity, literature formula (unvalidated, overestimates)",
    "do": "DO (model, mg/L)", "bod": "BOD (model, mg/L)", "turbidity": "Turbidity (model, NTU)",
    "wqi": "WQI (from model outputs)",
}
AOI_CMAPS = {"idx_ndci": "YlGn", "idx_turbidity_empirical": "YlOrBr", "do": "RdYlBu", "bod": "YlOrRd",
             "turbidity": "YlOrBr", "wqi": "RdYlGn_r"}


def _skill_summary(bundle: RealBundle) -> str:
    if bundle.metrics is None:
        return "no out-of-fold metrics available"
    m = bundle.metrics[(bundle.metrics["model"] == "xgboost_oof") & (bundle.metrics["water_body_type"] == "overall")]
    parts = [f"{r.target} R² {r.R2:.2f}" for r in m.itertuples()]
    return ", ".join(parts) if parts else "no out-of-fold metrics available"


def _aoi_tab(bundle: RealBundle) -> None:
    import datetime as dt

    from src.data import aoi as aoi_mod
    from src.app import raster_layers

    st.markdown(
        "Pick **any point on a water body in India** and a date. The app fetches the nearest clear Sentinel-2 scene "
        "(Planetary Computer, no login), masks the water, and maps the index and the trained models' output.")
    st.warning(
        f"**Skill warning.** Out-of-fold skill of the models: {_skill_summary(bundle)}. R² near 0 means the model "
        "cannot tell sites apart, so its maps sit near the training average. The NDCI layer is computed directly "
        "from reflectance and is the more trustworthy layer here.", icon="⚠️")

    c1, c2, c3, c4 = st.columns(4)
    lat = c1.number_input("Latitude", value=12.9359, format="%.4f", key="aoi_lat")
    lon = c2.number_input("Longitude", value=77.6701, format="%.4f", key="aoi_lon")
    date = c3.date_input("Date", value=dt.date(2020, 1, 29), key="aoi_date")
    half_km = c4.slider("Half-width (km)", 0.5, 5.0, 2.0, 0.5, key="aoi_half")
    st.caption("Default: Bellandur Lake, Bengaluru. Needs internet; takes ~15 s.")

    if st.button("Fetch scene and predict", key="aoi_run"):
        with st.spinner("Searching Sentinel-2 and predicting..."):
            try:
                st.session_state["aoi_result"] = aoi_mod.predict_aoi(lat, lon, date, half_size_m=half_km * 1000)
                st.session_state.pop("aoi_error", None)
            except aoi_mod.AOIError as e:
                st.session_state["aoi_error"] = str(e)
                st.session_state.pop("aoi_result", None)
            except Exception as e:                       # network / STAC problems
                st.session_state["aoi_error"] = f"Could not complete the request: {e}"
                st.session_state.pop("aoi_result", None)

    if st.session_state.get("aoi_error"):
        st.error(st.session_state["aoi_error"])
    res = st.session_state.get("aoi_result")
    if not res:
        return

    st.caption(f"Scene {res['scene_id']} · {res['scene_date']} ({res['day_diff']} day(s) from the requested date) · "
               f"scene cloud {res['scene_cloud']:.1f}% · {res['n_water_px']:,} water pixels")
    layers = res["layers"]
    key = st.selectbox("Layer", list(layers), format_func=lambda k: AOI_LAYER_LABELS.get(k, k), key="aoi_layer")
    if not HAS_FOLIUM:
        st.info("Install streamlit-folium to draw the map.")
        return
    import rasterio

    with rasterio.open(layers[key]) as src:
        data = src.read(1)
    valid = data != raster_layers.NODATA
    if not valid.any():
        st.info("No valid pixels in this layer.")
        return
    lo, hi = np.percentile(data[valid], [2, 98])
    m = raster_layers.base_map(res["lat"], res["lon"], zoom=14)
    raster_layers.PARAM_CMAPS.setdefault(key, AOI_CMAPS.get(key, "viridis"))
    raster_layers.add_raster_overlay(m, layers[key], key, name=AOI_LAYER_LABELS.get(key, key), vmin=float(lo),
                                     vmax=float(hi), opacity=0.75)
    folium.Marker([res["lat"], res["lon"]], tooltip="requested point").add_to(m)
    folium.LayerControl().add_to(m)
    st_folium(m, width="100%", height=520, returned_objects=[])
    st.caption(f"Colour scale: 2nd-98th percentile, {lo:.3g} to {hi:.3g}. {res['note']}")


def _performance_tab(bundle: RealBundle) -> None:
    if bundle.metrics is None:
        st.info("Run `python -m scripts.train_real_models` to produce out-of-fold metrics.")
        return
    st.markdown("Out-of-fold, site-blocked spatial cross-validation. **R² ≤ 0 means no better than predicting "
                "the training average**; the baselines are scored on the same folds.")
    if bundle.comparison is not None and len(bundle.comparison):
        st.markdown("#### Production choice per target")
        st.caption("XGBoost is the default; the DL model is chosen only when its out-of-fold RMSE is lower by more than 5%. "
                   "`beats_baseline` needs an RMSE at least 2 % below the best 'predict the average' baseline; `skill` labels the "
                   "out-of-fold R² (log scale for BOD/turbidity): none ≤ 0.05 < weak ≤ 0.30 < moderate ≤ 0.60 < good.")
        cols = ["target", "n", "chosen_R2", "chosen_R2_log", "chosen_RMSE", "production_model", "beats_baseline", "skill"]
        st.dataframe(bundle.comparison[[c for c in cols if c in bundle.comparison.columns]].round(3), width="stretch")
    if bundle.dl_summary:
        share = bundle.dl_summary.get("share_of_samples_with_history")
        if share is not None:
            st.caption(f"DL model: {share:.0%} of samples had at least one earlier visit of the same station within "
                       f"{bundle.dl_summary.get('max_gap_days')} days (window {bundle.dl_summary.get('window')}); the "
                       "temporal part of the model only matters for those.")
    types = sorted(bundle.metrics["water_body_type"].unique(), key=lambda t: (t != "overall", t))
    wt = st.selectbox("Water-body type", types, key="rv_wt")
    st.dataframe(metric_pivot(bundle.metrics, wt).round(3), width="stretch")
    st.caption("`baseline_mean` / `baseline_median` predict the training-fold mean / median. `xgboost_oof` and "
               "`dl_oof` are out-of-fold on the same site-blocked folds.")

    pred_options = {}
    if bundle.has_predictions:
        pred_options["XGBoost"] = "_pred"
    if bundle.has_dl_predictions:
        pred_options["DL model"] = "_pred_dl"
    if pred_options:
        target = st.selectbox("Measured vs predicted", ["do", "bod", "turbidity"], format_func=PARAM_LABELS.get, key="rv_sc")
        model_name = st.radio("Model", list(pred_options), horizontal=True, key="rv_scm")
        col = f"{target}{pred_options[model_name]}"
        t = bundle.table[[target, col, "water_body_type"]].dropna()
        if len(t):
            fig = px.scatter(t, x=target, y=col, color="water_body_type", opacity=0.5,
                             labels={target: f"measured {PARAM_LABELS[target]}", col: f"{model_name} (out-of-fold)"})
            lim = float(max(t[target].max(), t[col].max()))
            fig.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim, line=dict(dash="dash", color="grey"))
            if target != "do":
                fig.update_xaxes(type="log")
                fig.update_yaxes(type="log")
            st.plotly_chart(fig, width="stretch")


def _shap_tab() -> None:
    files = sorted((REAL_DIR / "shap_plots").glob("*.png")) if (REAL_DIR / "shap_plots").exists() else []
    if not files:
        st.info("No SHAP plots yet. Run `python -m scripts.train_real_models`.")
        return
    st.caption("SHAP attributions of the XGBoost models trained on the real data.")
    for f in files:
        st.image(str(f), caption=f.stem, width="stretch")


def _data_tab(bundle: RealBundle) -> None:
    s = bundle.summary
    st.markdown("### Source")
    st.markdown(
        "- **Labels:** CPCB National Water Quality Monitoring manual (grab-sample) data from the "
        "[National Water Data Portal](https://nwdp.nwic.gov.in) (open licence). DO, BOD, pH, coliform, conductivity, "
        "turbidity where reported.\n"
        "- **Satellite:** Sentinel-2 L2A surface reflectance (Planetary Computer), median of water pixels within "
        "500 m of the station, nearest clear scene within ±3 days.\n"
        "- **Not available:** in-situ chlorophyll-a (never measured by CPCB) and water temperature (empty in the files).\n"
        "- Rows in the older CPCB files that carry colour words instead of numbers in core columns are dropped; "
        "groundwater points (bore wells, hand pumps) are excluded.")
    if s:
        st.markdown("### This dataset")
        st.json({k: v for k, v in s.items() if k not in ("day_diff_days",)})
    if bundle.validation:
        st.markdown("### Empirical formulas vs real measurements")
        st.caption("The literature formulas are checked against the measurements instead of being trusted.")
        st.json(bundle.validation)
