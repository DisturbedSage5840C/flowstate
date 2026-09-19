# Aqua-Sense — Final Hackathon Execution Plan

**Spatiotemporal AI and Edge-Spectral Deep Learning for Water Quality Indexing of Indian Inland Waters (Lakes, Rivers, Reservoirs)**

**Team:** Aadi (P1) · Marutey (P2) · Navya (P3) · Jashan (P4)

## TL;DR

We estimate Dissolved Oxygen, BOD and turbidity (trained and validated on **real CPCB measurements**), an experimental Chlorophyll-a index (no ground truth exists for it), and a satellite-adapted WQI plus the official CPCB best-use class, for **any inland water body in India** from Sentinel-2 imagery (Landsat-8/9 as a turbidity/water-mask gap-filler only), and ship it as a Streamlit early-warning dashboard for regulators and municipal authorities. Bengaluru's lakes (Bellandur, Varthur, Ulsoor) and Punjab's rivers (Sutlej, Beas, Ghaggar, Buddha Nullah) are **demo sites, not the scope**: the pipeline is driven by a site config, and the dashboard lets a user pick a preset site or upload/draw any AOI.

Three differentiators for the pitch:

1. **Site-blocked spatial cross-validation.** Random splits leak spatial autocorrelation and inflate R². Station coordinates are clustered (KMeans) so neighbouring stations always share a fold, and every score is out-of-fold and compared with a "predict the average" baseline on the same folds. (This is our own implementation, not the `spatial-kfold` library.)
2. **Dual-tier atmospheric correction (status: wrappers written, not yet run on a real scene).** ACOLITE (Dark Spectrum Fitting) for turbid/sunglint-prone water, C2RCC (neural inversion) for hypereutrophic/CDOM-rich water. The results reported so far use Sen2Cor L2A surface reflectance, which is built for land; do not claim the dual-tier result until the pre-hackathon dry run succeeds.
3. **DO from real labels, with a physical surrogate checked against them.** DO has no direct spectral signature. It is learned from CPCB DO measurements; the oxygen-saturation-based surrogate is validated against those measurements instead of being trusted.

This document adds what the concept note lacked: a repo layout, a real ground-truth source (CPCB NWDP, see Section 3), a per-person work split, and hard checkpoints so the riskiest component (ACOLITE/C2RCC) cannot sink the 24 hours.

---

## 1. Critical Feasibility Notes

- **ACOLITE/C2RCC are not pip installs.** ACOLITE needs its own Python environment plus a working setup; C2RCC is a plugin of ESA SNAP (run via the GPT command line). Install and smoke-test them **before** the clock starts (Section 6). We keep them as the primary path because they are the scientific differentiator.
- **Hour-4 fallback (pre-approved, Aadi's call):** if ACOLITE/C2RCC output is not validated and flowing by Hour 4, switch to Earth Engine `COPERNICUS/S2_SR_HARMONIZED` and Landsat Collection-2 Level-2 surface reflectance. No mid-hackathon debate.
- **MNDWI threshold is scene-dependent.** Do not hardcode `MNDWI > 0.42`. Plot the histogram per scene/site and choose the threshold that visibly separates water from land (typically near 0, higher in urban areas).
- **Ground truth exists for DO, BOD, pH, coliform, conductivity and (partly) turbidity** (CPCB NWDP, Section 3). It does **not** exist for chlorophyll-a or water temperature; Chl-a is never presented as measured.
- **R² > 0.85 is not a promise.** Real satellite-to-DO/BOD relationships are weak. Report the out-of-fold R² next to the baseline's, including when it is low or negative; that honesty is the strength.
- **Treat numeric claims in the original concept note (thresholds, coefficients, some 2026-dated citations) as "verify on our data".**

### Lakes vs rivers: what changes for India-wide coverage

| Concern | Lakes / reservoirs | Rivers |
|---|---|---|
| Resolution | 10 m and 20 m bands fine | Narrow channels: keep B5/B6/B11 (20 m, resampled to the 10 m grid) so NDCI/MNDWI still work, and treat Chl-a as low-confidence; Landsat 30 m is unusable for channels under ~90 m wide |
| Mask | Standard MNDWI + small erosion | Erode the water mask by 1 pixel to avoid mixed bank pixels; skip sites narrower than ~30 m |
| Turbidity | Mostly low-to-moderate, red/green branch | Monsoon sediment loads are extreme (Ganga, Yamuna, Brahmaputra): switch to NIR branch above the FNU threshold; validate switch point per region |
| Clouds | Monsoon gaps | Same, worse: use Landsat gap-fill and 10-day composites in Jun-Sep |
| Chl-a signal | Strong (eutrophic urban lakes) | Weaker in fast, turbid rivers: expect lower R² for Chl-a in rivers, report per water-body-type metrics |

**Landsat 8/9 has no red-edge band** (its B5 is NIR), so it cannot produce NDCI or Chl-a; it is only used for turbidity and the water mask.

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
│   ├── acquisition/          # GEE ingestion (earthengine-api), reads config/sites.yaml
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

**Outcome (verified against the live API):** CPCB's National Water Data Portal (`nwdp.nwic.gov.in`, open CKAN API, no login) serves state-wise CSVs of manual grab-sample monitoring with station name, latitude, longitude and timestamp: DO, pH, BOD, COD, coliform, conductivity, ammonia, turbidity. About 80,000 surface-water visits at ~3,100 stations across 36 states/UTs for 2019-2021 (a handful later) are usable. Caveats found while loading it: chlorophyll-a and temperature are absent; turbidity exists mostly for 2020; bore-well/hand-pump points appear in the surface-water files and are excluded; older files contain rows with colour words in numeric columns and corrupt numbers; the "2026-2030" files are empty. Code: `src/data/nwdp.py`, `scripts/fetch_insitu.py`.

Original sourcing plan (kept for reference; the proxy fallback in step 3 is no longer needed for DO/BOD/turbidity). Resolve in this order, **locked by end of Hour 2** (Marutey's call):

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
4. Results: per-type out-of-fold R²/RMSE/MAE next to the baseline's, and SHAP exactly as computed (report what it shows, not what was hypothesised).
5. Live demo: pick a preset, then upload a new AOI; backup video ready in case of venue wifi failure.
6. Impact and roadmap: early-warning alerts, more parameters, Sentinel-3/hyperspectral (PACE/EnMAP), state-board integration.

---

## 9. Quality outputs, as implemented

- **WQI (0-100).** Aqua-Sense's own satellite-adapted weighted-arithmetic pollution index (0 pristine, 100 worst; tiers
  Excellent < 20, Good < 40, Moderate < 60, Poor < 80, Very Poor <= 100). Qi = 100·(Vi − V0)/(Si − V0) clipped to [0, 100]
  (pH: |Vi − 7|/(8.5 − 7)); weights Wi ∝ 1/Si renormalised over the parameters actually present; nothing is imputed and
  at least two parameters are required. The ideal DO value is the saturation at the water temperature (default 25 °C), not
  14.6 mg/L. Chl-a is included only as an eutrophication indicator (reference 10 µg/L) — it is not a CPCB parameter.
  Code: `src/wqi/wqi_engine.py`.
- **CPCB designated-best-use class (A-E)** is a *separate* output, assigned from the CPCB concentration criteria
  (coliform, pH, DO, BOD for A-C; pH, DO, free ammonia for D; pH, conductivity, SAR, boron for E; source
  https://cpcb.gov.in/water-quality-criteria/). It is only reported when at least two criteria were measured;
  `cpcb_class_complete` says whether all criteria of the assigned class were available. Free ammonia is derived from
  total ammonia-N, pH and temperature (Emerson 1975).

## 10. Deliverable status (as built, 2026-09-19)

| Deliverable | Status | Evidence |
|---|---|---|
| Real in-situ labels (CPCB NWDP) | done | `src/data/nwdp.py`, `data/ground_truth/data_source_log.md` |
| Sentinel-2 matching at stations | done (Planetary Computer) | `src/data/satellite_extract.py`, `reports/real/dataset_summary.json` |
| Landsat thermal temperature | done, ~40 % coverage | `src/data/landsat_temp.py` |
| Spatial CV (site-blocked KMeans) | done | `src/models/spatial_cv.py` |
| XGBoost + Optuna + SHAP on real data | done; **low skill** | `reports/real/metrics_table.csv` |
| CNN-BiLSTM-attention over visit windows | done; only helps where history exists | `reports/real/dl_summary.json` |
| WQI + CPCB class | done | section 9 |
| Dashboard on real data + any-AOI map | done | `src/app/real_view.py`, `src/data/aoi.py` |
| Dual-tier atmospheric correction (ACOLITE + C2RCC) | **not run**; wrappers + converter written, untested | `src/preprocessing/correction*.py` |
| Earth Engine | enabled and verified live; extraction agrees with Planetary Computer (r 0.97-0.98); 8,461-row table built and now the default training table | `src/data/gee_extract.py`, `reports/real/backend_comparison.json` |
| Kaggle datasets | Ganga/Sangam used only to validate Landsat temperature (its pH/conductivity are unreliable) | `src/data/kaggle_sources.py` |
| Turbidity formula recalibration | literature Nechad/Dogliotti had an 11.7x median overestimate vs measured CPCB turbidity; refit per-water-body-type power law (`turbidity_calibrated`) brings the median ratio to ~1.0 and raises pooled Spearman 0.23 -> 0.33 | `src/features/feature_engineering.py`, `reports/real/empirical_formula_validation.json` |
| WQI-tier classifier (satellite-only) | done; leakage-free (features exclude do/bod/ph/turbidity/etc.); OOF accuracy 0.29 vs 0.27 majority-class baseline, macro F1 0.21 vs 0.14 — a real but modest edge | `src/models/tier_classifier.py`, `reports/real/tier_classification_metrics.json` |
| R² > 0.85 | **not achieved and not achievable with this signal**; out-of-fold raw R² stays near zero even after recalibration and more data — this is a genuine ceiling of single-satellite-scene optical reflectance for chemistry parameters under honest spatial CV, not a bug. Rank correlation (Spearman 0.14-0.24) and log-scale R² are the honest headline numbers and both beat naive baselines | `reports/real/metrics_table.csv` |
