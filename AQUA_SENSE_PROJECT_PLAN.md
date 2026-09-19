# Aqua-Sense — Final Hackathon Execution Plan

**Spatiotemporal AI and Edge-Spectral Deep Learning for Water Quality Indexing of Indian Inland Waters (Lakes, Rivers, Reservoirs)**

**Team:** Aadi (P1) · Marutey (P2) · Navya (P3) · Jashan (P4)

## TL;DR

We predict Chlorophyll-a, Turbidity, Dissolved Oxygen and an aggregated CPCB Water Quality Index (WQI) for **any inland water body in India** from Sentinel-2 and Landsat-8/9 imagery, and ship it as a Streamlit early-warning dashboard for regulators and municipal authorities. Bengaluru's lakes (Bellandur, Varthur, Ulsoor) and Punjab's rivers (Sutlej, Beas, Ghaggar, Buddha Nullah) are **demo sites, not the scope**: the pipeline is driven by a site config, and the dashboard lets a user pick a preset site or upload/draw any AOI.

Three differentiators for the pitch:

1. **Spatial-kfold cross-validation.** Random pixel splits leak spatial autocorrelation and inflate R². We train on many water bodies across regions and validate on geographically held-out blocks, so the reported score reflects performance on water bodies the model has never seen.
2. **Dual-tier atmospheric correction.** ACOLITE (Dark Spectrum Fitting) for turbid/sunglint-prone water, C2RCC (neural inversion) for hypereutrophic/CDOM-rich water. Sen2Cor is built for land, not Case-II inland water.
3. **DO as a surrogate variable.** DO has no direct spectral signature; it is inferred from Chl-a, turbidity and temperature dynamics by the sequential model.

This document adds what the concept note lacked: a repo layout, a plan for having no ground truth yet, a per-person work split, and hard checkpoints so the riskiest component (ACOLITE/C2RCC) cannot sink the 24 hours.

---

## 1. Critical Feasibility Notes

- **ACOLITE/C2RCC are not pip installs.** ACOLITE needs its own Python environment plus a working setup; C2RCC is a plugin of ESA SNAP (run via the GPT command line). Install and smoke-test them **before** the clock starts (Section 6). We keep them as the primary path because they are the scientific differentiator.
- **Hour-4 fallback (pre-approved, Aadi's call):** if ACOLITE/C2RCC output is not validated and flowing by Hour 4, switch to Earth Engine `COPERNICUS/S2_SR_HARMONIZED` and Landsat Collection-2 Level-2 surface reflectance. No mid-hackathon debate.
- **MNDWI threshold is scene-dependent.** Do not hardcode `MNDWI > 0.42`. Plot the histogram per scene/site and choose the threshold that visibly separates water from land (typically near 0, higher in urban areas).
- **No ground truth exists yet.** Biggest schedule risk; gets a timeboxed sprint (Section 3).
- **R² > 0.85 is a stretch target.** The pitch strength is that our number is honest (spatially blocked). Say so to judges before they ask.
- **Treat numeric claims in the original concept note (thresholds, coefficients, some 2026-dated citations) as "verify on our data".**

### Lakes vs rivers: what changes for India-wide coverage

| Concern | Lakes / reservoirs | Rivers |
|---|---|---|
| Resolution | 10 m and 20 m bands fine | Narrow channels: use 10 m bands only (B2, B3, B4, B8); Landsat 30 m is unusable for channels under ~90 m wide |
| Mask | Standard MNDWI + small erosion | Erode the water mask by 1 pixel to avoid mixed bank pixels; skip sites narrower than ~30 m |
| Turbidity | Mostly low-to-moderate, red/green branch | Monsoon sediment loads are extreme (Ganga, Yamuna, Brahmaputra): switch to NIR branch above the FNU threshold; validate switch point per region |
| Clouds | Monsoon gaps | Same, worse: use Landsat gap-fill and 10-day composites in Jun-Sep |
| Chl-a signal | Strong (eutrophic urban lakes) | Weaker in fast, turbid rivers: expect lower R² for Chl-a in rivers, report per water-body-type metrics |

Report metrics **per water-body type (lake / river / reservoir) and overall**. This is more honest and gives judges a richer story than one blended number.

---

## 2. Repository Layout

```
prayashack/
├── AQUA_SENSE_PROJECT_PLAN.md
├── README.md
├── requirements.txt
├── config/
│   └── sites.yaml            # name, type (lake/river/reservoir), state, bbox or GeoJSON path
├── data/
│   ├── raw/                  # untouched GEE exports / downloaded scenes
│   ├── interim/              # masked, atmospherically-corrected rasters
│   ├── processed/            # final feature tables joined with ground truth
│   └── ground_truth/         # sourced / proxy in-situ data
├── notebooks/
├── src/
│   ├── acquisition/          # GEE ingestion (geemap), reads config/sites.yaml
│   ├── preprocessing/        # cloud/QA masking, MNDWI, ACOLITE/C2RCC wrappers
│   ├── features/             # NDCI, 2BDM/3BDM, turbidity, spatial-temporal join
│   ├── models/
│   │   ├── spatial_cv.py         # spatial-kfold wrapper
│   │   ├── xgboost_pipeline.py   # Optuna-tuned XGBoost
│   │   └── dl_model.py           # 1D-CNN-BiLSTM-Attention (PyTorch)
│   ├── wqi/
│   │   └── wqi_engine.py         # CPCB weighted-arithmetic WQI + A-E class
│   └── app/
│       ├── streamlit_app.py
│       └── map_utils.py          # folium / deck.gl layers
└── reports/
    └── shap_plots/
```

**Example `config/sites.yaml` seed (extend freely):**

| Site | Type | State |
|---|---|---|
| Bellandur, Varthur, Ulsoor | lake | Karnataka |
| Sutlej / Beas / Ghaggar stretches | river | Punjab |
| Buddha Nullah (Ludhiana) | river | Punjab |
| Yamuna (Delhi stretch) | river | Delhi |
| Ganga (Kanpur / Varanasi stretch) | river | UP |
| Hussain Sagar | lake | Telangana |
| Dal Lake | lake | J&K |
| Chilika | lagoon/lake | Odisha |

Pick 8-15 sites across at least 4 states so spatial blocks are genuinely regional.

---

## 3. Ground-Truth Data Sourcing (Hour 0-2, hard deadline)

No in-situ Chl-a/Turbidity/DO with GPS + date exists yet. Resolve in this order, **locked by end of Hour 2** (Marutey's call):

1. **Hackathon organizer data / problem statement.** Check first.
2. **National and state public sources:**
   - CPCB National Water Quality Monitoring Programme (NWMP) and real-time WQMS stations.
   - India-WRIS (Water Resources Information System) and data.gov.in.
   - State pollution control boards (KSPCB, PPCB, DPCC, UPPCB, TSPCB, etc.) bulletins.
   - Published academic datasets and NGO/citizen-science reports for the chosen sites.
   - Note: CPCB stations report BOD, DO, pH, coliform, conductivity, turbidity and TSS more often than Chl-a. Where Chl-a is missing, model DO/turbidity/WQI directly and treat Chl-a as an intermediate feature.
3. **Fallback: literature-calibrated proxy labels.** If no geo/time-matched data is found, generate pseudo-ground-truth from published empirical algorithms (Nechad turbidity, NDCI/OC3-style Chl-a, published coefficients). **Disclose this openly as a proxy approach.** Train ML as a nonlinear/residual corrector on the empirical baseline, and sanity-check against documented real events (Bellandur froth incidents, Yamuna foam at Kalindi Kunj, Buddha Nullah pollution, monsoon flood turbidity spikes).
4. Freeze the decision by Hour 2 regardless of path.

**Matching rule:** join satellite features to station readings within +/-3 days (Sentinel-2) or +/-5 days (Landsat) and within a small radius of the station; log how many matched pairs each site contributes. Drop sites with too few pairs rather than padding them.

---

## 4. Team Roles and Ownership

| Person | Role | Owns |
|---|---|---|
| **Aadi (P1)** | Geospatial / Data Engineer | Site config, GEE ingestion, cloud/QA masking, MNDWI, ACOLITE + C2RCC correction (Hour-4 fallback authority), full-extent prediction rasters, dashboard map layer |
| **Marutey (P2)** | Data Scientist | Ground-truth sourcing sprint, spatial-temporal join, feature engineering (NDCI, 2BDM/3BDM, turbidity switch), CPCB WQI engine, DO surrogate inputs |
| **Navya (P3)** | ML Engineer A | spatial-kfold, Optuna-tuned XGBoost, SHAP interpretability, metrics reporting (R², RMSE, MAE by water-body type) |
| **Jashan (P4)** | ML Engineer B / Full-Stack | PyTorch 1D-CNN-BiLSTM-Attention, then Streamlit dashboard lead |

### Per-person deliverables

**Aadi**
- `config/sites.yaml` + loader; GEE fetch function `fetch_scenes(site, start, end)`.
- Masked, corrected BOA reflectance rasters per site/date in `data/interim/`.
- Written record of MNDWI threshold chosen per site type.
- Prediction rasters per site for the map; folium/deck.gl layer in `map_utils.py`.

**Marutey**
- `data/ground_truth/` master table: `site, lat, lon, date, chl_a, turbidity, do, bod, source`.
- `features` module producing the training table (schema below).
- `wqi_engine.py`: takes predicted parameters, returns WQI (0-100 scale as defined in plan) and CPCB class A-E; unit-tested with hand-computed cases.
- Data-source log (what was found, what was proxied).

**Navya**
- `spatial_cv.py` with fold assignment by clustered lat/lon (KMeans/BisectingKMeans), ensuring whole water bodies never straddle train/test.
- Optuna study for XGBoost (learning_rate, max_depth, subsample, colsample_bytree, L2 lambda) optimizing spatial-CV RMSE.
- Saved model artifact + `predict(df)` function; SHAP summary plots (B5 red-edge influence on Chl-a); metrics table.

**Jashan**
- `dl_model.py`: parallel `Conv1d` (k=1,3,5) -> `LSTM(bidirectional=True)` -> dot-product attention -> dense head; trained on Navya's folds.
- Streamlit app: site selector (preset or uploaded GeoJSON), date range, parameter toggle (Chl-a / Turbidity / DO / WQI), metrics and SHAP panels.
- Same `predict(df)` interface for the DL model so the app can swap models.

### Handoff interfaces (agree these in Hour 0-1 to avoid blocking)

1. **Aadi -> Marutey:** corrected rasters named `{site}_{sensor}_{YYYYMMDD}.tif`, bands ordered B2,B3,B4,B5,B8,B11 (S2) / B3,B4,B5,B6 (Landsat), plus water mask.
2. **Marutey -> Navya/Jashan:** one CSV/parquet at `data/processed/train.parquet` with columns `site, water_body_type, lat, lon, date, sensor, B2..B11, ndci, bdm2, bdm3, red_green, nir, temp_surface, chl_a, turbidity, do, wqi`. Until real data lands, Marutey publishes a **synthetic file with the identical schema by Hour 2** so Navya and Jashan can build against it.
3. **Navya/Jashan -> dashboard:** each model exposes `predict(df) -> DataFrame[chl_a, turbidity, do]`; Marutey's `wqi_engine` consumes that output.

---

## 5. Hour-by-Hour Plan (24 hours)

| Hours | Aadi | Marutey | Navya | Jashan |
|---|---|---|---|---|
| **0-1** Kickoff | Repo scaffold, `sites.yaml` draft, GEE auth check | Start data-source sprint; agree schema | Agree schema; set up env | Agree schema; set up env |
| **1-2** | Fetch first scenes for 2-3 sites | **Lock ground-truth strategy (Hour 2)**; publish synthetic `train.parquet` | Build `spatial_cv.py` on synthetic data | Build `dl_model.py` skeleton on synthetic data |
| **2-4** | Masking + MNDWI tuning; ACOLITE/C2RCC runs | Build WQI engine + tests; start collecting station data | Optuna + XGBoost skeleton | Train loop, data loader for time windows |
| **4** | **Correction go/no-go:** validated BOA or fall back to GEE SR | Support fall-back decision | - | - |
| **4-7** | Batch-process all sites; export rasters | Spatial-temporal join, feature engineering, real `train.parquet` | Swap in real data; first spatial-CV numbers | Swap in real data; first DL run |
| **7-12** | Full-extent prediction rasters; start map layer | QA feature table; DO surrogate inputs; per-type metrics | Optuna tuning, SHAP plots | Tune/benchmark DL on same folds; start Streamlit shell |
| **12** | Model go/no-go input | Model go/no-go input | **Model selection by spatial-CV** (default XGBoost) | Model selection |
| **12-19** | Choropleth/heatmap layer, WQI colour tiers | Wire WQI engine + A-E class into app; sub-index breakdown | SHAP + metrics panel; final model export | Lead Streamlit; upload-your-own-AOI flow |
| **19-22** | Integration testing on fresh AOIs | Validate WQI outputs against known events | Fix inference/latency issues | Fix UI issues |
| **22-24** | Record demo video/screenshots | Pitch: data + WQI story | Pitch: methodology + validation | Pitch: live demo + impact |

If a model is not clearly better by Hour 12, ship XGBoost as production and present the DL model as a benchmarked comparison.

---

## 6. Pre-Hackathon Checklist (before the clock starts)

**Aadi**
- [ ] Install ACOLITE and ESA SNAP + C2RCC plugin; confirm GPT command line works.
- [ ] Set up Earth Engine service-account auth; run `ee.Initialize()`.
- [ ] Run **one full Sentinel-2 scene end-to-end**: GEE fetch -> mask -> ACOLITE -> C2RCC -> sane BOA reflectance. If this fails, know it now, not at Hour 4.
- [ ] Draft `sites.yaml` with 8-15 sites; collect AOI polygons.

**Marutey**
- [ ] Scout CPCB NWMP, India-WRIS, data.gov.in, state PCB downloads; note which sites have usable station data and date ranges.
- [ ] Confirm CPCB WQI parameter weights and permissible limits from the official criteria.

**Navya**
- [ ] Pin and install `xgboost`, `optuna`, `spatial-kfold`, `shap`, `scikit-learn`; run a toy spatial-CV example.

**Jashan**
- [ ] Install `torch`, `streamlit`, `folium`; run a "hello world" Streamlit map; scaffold the CNN-BiLSTM-attention module on random tensors.

**Everyone**
- [ ] Shared `requirements.txt` committed; Git workflow agreed (branch per person, merge at checkpoints).

---

## 7. Go / No-Go Checkpoints

- **Pre-hackathon:** sample scene passes the full correction chain (Aadi).
- **Hour 2:** ground-truth strategy frozen, synthetic + schema published (Marutey).
- **Hour 4:** atmospheric correction validated, or fall back to GEE SR (Aadi).
- **Hour 12:** production model chosen by spatial-CV metrics (Navya + Jashan).
- **Hour 22-23:** end-to-end run on a fresh, previously unseen AOI in a different state, live in the dashboard (whole team).

## 8. Pitch Outline

1. Problem: unmonitored or sparsely monitored Indian water bodies (Bengaluru lake froth, Yamuna foam, Punjab river pollution); station sampling is sparse and slow.
2. Solution: satellite-based, India-wide, any-AOI water quality indexing with a CPCB-aligned WQI.
3. Novelty: spatial-kfold honesty, dual-tier atmospheric correction, DO surrogate modeling.
4. Results: per-type spatial-CV R²/RMSE/MAE, SHAP showing red-edge (B5) drives Chl-a.
5. Live demo: pick a preset, then upload a new AOI; backup video ready in case of venue wifi failure.
6. Impact and roadmap: early-warning alerts, more parameters, Sentinel-3/hyperspectral (PACE/EnMAP), state-board integration.
