# Aqua-Sense — Fix Plan

## 2026-09-19 (later): evidence-driven model redesign superseded the numbers below

Everything in this file predates the model redesign in `AQUA_SENSE_PROJECT_PLAN.md` §11 and
`data/ground_truth/data_source_log.md` §4. In short: the low out-of-fold R² for DO/BOD/turbidity noted
throughout this checklist is not a defect to keep chasing — a variance decomposition confirmed 85-100% of
each target's variance is between-station, invisible to a single satellite visit. Pollution **screening**
(binary, e.g. BOD>3 mg/L, out-of-fold AUC ~0.75) is now the headline deliverable; row-level regression is
kept as a secondary, explicitly flagged result. This file's checklist below is left as the historical
record of what was reviewed and fixed up to that point; it is not being rewritten.

# Submission status (updated 2026-09-20)

Read this first. The checklist and tables below are the historical record; this is where things stand.

| Item | Final status |
|---|---|
| Flow State UI wired to the backend | **Done.** `frontend/` + `src/api/`; see `AQUA_SENSE_PROJECT_PLAN.md` section 13. Dark mode included. |
| Spatial-KNN model in the UI | **Done.** Held-out estimate next to the measured value, with distance to the nearest station and overall skill (`reports/real/spatial_knn_oof.parquet`). |
| Model improvements | **Done.** Distance/spread features in the inner model, sweep and land-cover ablations recorded, defaults kept where the evidence said so. |
| 5.5 CI | **Done and green** (`.github/workflows/ci.yml`: Python tests + frontend build). Its first runs caught two real problems, a missing `httpx2` test dependency and NaN leaking into API JSON under pandas 3, both fixed. |
| 2.5 Nechad coefficients | **Closed as won't-verify.** The constants are unverified and not relied on: the pipeline uses an empirical refit against measured CPCB turbidity. Do not cite the literature formula as validated. |
| 4.6 / 4.7 ACOLITE + C2RCC on a real scene | **Open; needs software this project never had (SNAP, ACOLITE).** All results use Sen2Cor L2A. The pitch must not claim dual-tier correction. |
| Section 6 pre-hackathon checklist in the project plan | Historical; superseded by the real-data pipeline. |

# Status (updated 2026-09-19)

Ticked boxes below have evidence (code + tests, or a run recorded in `reports/`). Items left open, and why:

| Item | Status |
|---|---|
| 2.5 Nechad coefficients | **Open.** The original tables could not be retrieved, so the constants are still unverified. They were instead checked against real CPCB turbidity: the formula overestimates by roughly an order of magnitude (`reports/real/empirical_formula_validation.json`). Do not cite it as a validated retrieval. |
| 4.6 Correction connected to the pipeline | **Partly done.** ACOLITE output -> band-contract raster converter written and unit-tested on synthetic files named as assumed; never run on real ACOLITE output (not installed). C2RCC output is not converted. Routing is still a static site-type rule (now described as such). |
| 4.7 C2RCC command line | **Partly done.** The wrong `gpt` arguments were replaced by a documented Read -> Resample -> Subset -> c2rcc.msi -> Write graph; **untested** (SNAP not installed). |
| 5.5 CI | **Open** (changing shared CI needs an explicit go-ahead). |
| Earth Engine paths (2.7 thresholds, 4.5 export) | **Now verified live** (API enabled 2026-09-19): small-site export, fetch_scenes, MNDWI tuning and the station extraction all ran; two bugs were found and fixed (missing NIR cap in the raster water mask; Otsu on land-dominated histograms). Multi-tile export of large sites is still only unit-tested. |
| Kaggle datasets | Ganga/Sangam downloaded and quality-checked (pH and conductivity unreliable; temperature used only to validate Landsat). `anbarivan` (2003-2014) predates Sentinel-2 and cannot be matched. |
| Phase 7 | See the final acceptance section (filled in after the last full run). |

Findings that change the pitch (details in `data/ground_truth/data_source_log.md`): real out-of-fold skill for DO, BOD and
turbidity is low; the literature turbidity formula fails against measurements; the "temporal" DL model only matters for the
few visits that have a recent earlier visit at the same station.

---


Snapshot reviewed: `origin/main` @ `279a2af` (all four contributors' work integrated). 70 tests pass, but the tests do not cover the problems below. Every item lists the problem, the evidence, the fix and a "done when" check. Items are unassigned and ordered by priority. Tick the box when the "done when" check passes.

Items marked **(unverified)** were read from the code but not run or checked against a source.

## Decisions to make first

These change how several items are fixed. A recommended default is given so nothing blocks.

| # | Decision | Recommended default |
|---|---|---|
| D1 | Label strategy: real in-situ data vs. openly synthetic/proxy | Try for real, cited data for a time-boxed slot. Otherwise ship as an **openly synthetic demo** (banner on dashboard, wording in pitch). Never present proxy numbers as measurements. |
| D2 | WQI scale | One bounded 0–100 pollution index (sub-indices capped at 100), plus a **separate** CPCB best-use class A–E from concentration criteria. |
| D3 | Deep model: make it truly temporal, or rename it | Build real time windows if multi-date data exists; otherwise call it an MLP and drop temporal claims. |
| D4 | Spatial CV: use the `spatial-kfold` library or the in-house KMeans site blocks | Keep the in-house `SpatialKFold`, and change all wording (plan, pitch, dashboard) to "site-blocked KMeans spatial CV". |
| D5 | Uploaded-AOI feature: real prediction or preview only | Preview only for now, labelled as such. Real prediction is a stretch item (4.7). |

---

## Phase 0 — Make it run and stop showing fake numbers

- [x] **0.1 Fresh install must work.** `requirements.txt` lacks `pyyaml` and `streamlit-folium`. `plotly>=5.18` resolves to 7.x, where `px.scatter_mapbox` no longer exists, so [streamlit_app.py:393](src/app/streamlit_app.py#L393) crashes on the map tab when `streamlit-folium` is absent.
  Fix: add both packages; switch to `px.scatter_map`; pin every dependency to the versions actually tested (streamlit 1.64, plotly 7.1, matplotlib 3.11, torch 2.14, xgboost 3.4, pandas 3.0, Python 3.14). `width='stretch'` is used in the app, so streamlit's minimum must support it.
  Done when: a clean venv + `pip install -r requirements.txt` + `streamlit.testing.v1.AppTest` run gives zero exceptions.
- [x] **0.2 Raster map layer crashes.** [raster_layers.py:45](src/app/raster_layers.py#L45) calls `cm.get_cmap`, removed in matplotlib 3.9 (reproduced on 3.11).
  Fix: `matplotlib.colormaps[name]`; add an explicit `import matplotlib.image`.
  Done when: `add_raster_overlay` renders a test GeoTIFF (add as a test).
- [x] **0.3 Remove every hardcoded result from the dashboard** ([streamlit_app.py](src/app/streamlit_app.py)).
  - Benchmark table (~L572): XGBoost values "0.770 / 10.10 / 0.830 / 16.40 / 0.710 / 1.95" are invented.
  - Fallback SHAP dict (~L616): invented, and it contradicts the real SHAP plots (real `ndci` ≈ 134 vs `B5` ≈ 5 for Chl-a).
  - Text claims: "RMSE bounded within ±2.2 mg/L of in-situ station monitoring" (no in-situ data exists), "Honest spatial-kfold cross-validation on held-out water bodies", "DL achieves R² > 0.90 on held-out sites".
  - The "About" tab must describe what each module actually does.
  Fix: read the benchmark table from the metrics files produced by 1.4/1.5; show the real SHAP PNGs only; rewrite the text to match reality.
  Done when: `grep` finds no numeric literal in the dashboard that isn't read from a file.
- [x] **0.4 Dashboard must show model output, not dataset labels.** `run_model_predict` ([L183](src/app/streamlit_app.py#L183)) is never called. `get_predictions` displays the parquet's label columns as "predictions", and the model radio does nothing.
  Fix: call the selected model on the feature columns and display its output. XGBoost loader must use `WaterQualityXGB.load("reports/models")`; today it looks for `reports/xgboost_model.pkl` and imports a module-level `predict` that does not exist, and the failure is swallowed. If model artifacts are missing, show `st.error`, do not fall back silently.
  Done when: switching the model radio changes the displayed values.
- [x] **0.5 Add a persistent banner** while data is synthetic/proxy: "Demonstration data — not measurements" (see D1).

## Phase 1 — Make the numbers honest

- [x] **1.1 Labels are exact formulas of the features (leakage).** Verified: `chl_a == 10^(1.35·ndci+1.58)` (max diff 3e-5), `turbidity == Nechad(B4,B8)` (diff 0.0). A site-grouped gradient-boosting model gets R² 0.998 / 0.991 / 0.891 (Chl-a / turbidity / DO). Real SHAP confirms the models just recover the formula (Chl-a: `ndci`; turbidity: `B4`).
  Cause: [generate_synthetic_train.py](scripts/generate_synthetic_train.py) draws latent `_chl_true` / `_turb_true`, then never uses them and overwrites the labels with formulas of the reflectances.
  Fix: use the latent values as labels; generate reflectances from them **plus independent noise**; generate DO from its own latent driver plus noise, not from the same features. Keep the per-site ranges the script already declares.
  Done when: no label column equals a formula of any feature column, and the leakage test in 5.5 passes.
- [x] **1.2 Keep formula outputs out of the model inputs.** `compute_all_features` ([feature_engineering.py:169](src/features/feature_engineering.py#L169)) writes `chl_a`, `turbidity`, `do` columns, which collide with the target names.
  Fix: rename to `chl_a_empirical`, `turbidity_empirical`, `do_empirical`; make sure they are never in `FEATURE_COLS`.
- [x] **1.3 Ground truth must be traceable.** [master_table.csv](data/ground_truth/master_table.csv) has 40 hand-entered rows with no source file; [data_source_log.md](data/ground_truth/data_source_log.md) says CPCB/PPCB stations publish the BOD/DO values and the pitch text says "CPCB published BOD/DO values". Its turbidity medians differ from `train.parquet` by ~30x (e.g. Buddha Nullah 412 vs 12.6), and only 2 (site, date) pairs overlap.
  Fix: add `source_document`, `source_url`, `retrieved_on`, `is_proxy` columns; every row without a checkable source is marked `illustrative`. Remove or reword any claim of CPCB provenance that cannot be traced. Reconcile the master table and the training table's ranges.
  Done when: every row either cites a retrievable source or is marked illustrative, and the log/pitch text matches.
- [x] **1.4 XGBoost metrics are in-sample.** [train_models.py:124](scripts/train_models.py#L124) predicts on the same data the final models were refit on, then prints and saves it as "SPATIAL CROSS-VALIDATION METRICS" (`metrics_summary.json`, `metrics_table.csv`). The true CV figure is only an RMSE (`spatial_cv_best_rmse`). Also, [xgboost_pipeline.py:287](src/models/xgboost_pipeline.py#L287) early-stops on the validation fold, which leaks it into the CV score.
  Fix: collect **out-of-fold predictions** using the tuned parameters and report R²/RMSE/MAE (overall, per type, per fold, per site) from them. Drop early stopping inside CV (or nest it). Label in-sample fit separately as "train fit". Note best-trial selection bias.
  Done when: reported R² comes only from out-of-fold predictions.
- [x] **1.5 DL evaluation does not use spatial CV.** [train_dl_model.py](scripts/train_dl_model.py) holds out the last three sites alphabetically (Ulsoor, Varthur, Yamuna) — this splits the Bengaluru cluster (Bellandur is in train) — and uses a random row split for validation. Per-type "river" is a single site. Reported: Chl-a R² 0.91, turbidity −1.34, DO −0.51 (worse than a constant), versus 0.999 / 1.0 / 0.96 for a plain gradient-boosting model on the same split.
  Fix: use `SpatialKFold` for all splits; report out-of-fold metrics with the same code as XGBoost (1.4) so the comparison is fair; regenerate `reports/metrics.json` (currently stale).
- [x] **1.6 One metrics format, one comparison table** written by one script and read by the dashboard.

## Phase 2 — Scientific correctness

- [x] **2.1 DO surrogate is wrong.** [feature_engineering.py:150](src/features/feature_engineering.py#L150): `14.62 − 0.3898·T` gives 4.9 / 3.7 / 1.0 mg/L at 25 / 28 / 35 °C versus true saturation 8.3 / 7.8 / 7.0 (errors of −3.4 / −4.1 / −6.0). DO has the largest WQI weight.
  Fix: use a standard freshwater saturation equation (Benson–Krause or Garcia–Gordon) and check 20/25/30/35 °C against 9.09/8.26/7.56/6.95 mg/L; consider an altitude correction (Dal Lake ≈ 1580 m). Re-derive the depression terms and document them as heuristic.
  Done when: unit test on those four temperatures passes.
- [x] **2.2 WQI is degenerate and defined three ways.**
  - Verified: all 390 rows in `train.parquet` are Class E, WQI 205–1458; the dashboard headline reads "Avg WQI 1015, Class E, Avg DO 0.79 mg/L".
  - Sub-indices are uncapped ([wqi_engine.py:106](src/wqi/wqi_engine.py#L106)); the plan says "0–100".
  - Tier tables disagree: [raster_layers.py](src/app/raster_layers.py) (<50 / 50–100 / >100), [wqi_engine.py](src/wqi/wqi_engine.py) and [map_utils.py](src/app/map_utils.py) (A–E at 25/50/75/100); the unused `CPCB_CLASSES` list in the engine contradicts its own `wqi_class`.
  - Weights are hand-assigned, but the plan defines Wi = K/Si.
  - "CPCB class A–E" is really a set of best-use classes defined by concentration criteria (DO, BOD, pH, coliform…), not WQI bands. **(unverified: confirm thresholds against the official CPCB table before coding)**
  Fix (per D2): one bounded index computed with K/Si weights; a separate best-use class function from concentrations; define tiers **once** in `wqi_engine` and import them everywhere. Chl-a and turbidity are not CPCB parameters — label the index "satellite-adapted WQI".
  Done when: parquet WQI values spread across several tiers, and there is exactly one tier definition in the repo.
- [x] **2.3 BOD is invented in the dashboard.** [streamlit_app.py:115](src/app/streamlit_app.py#L115) (`bod = turbidity·0.15 + …`) and the models do not predict BOD, but the WQI needs it.
  Fix: either add BOD as a documented proxy target, or compute WQI from the predicted parameters only (renormalised weights). Never fabricate an input silently.
- [x] **2.4 Chl-a proxy formula.** `10^(1.35·NDCI+1.58)` is described as "Mishra & Mishra 2012, calibrated for Indian inland waters". It does not match the published NDCI polynomial (repo 96.6 vs 57.4 µg/L at NDCI 0.3), and the "Indian calibration" is unsupported.
  Fix: use a cited published relation or rename the function "illustrative" and delete the claim.
- [ ] **2.5 Nechad coefficients (unverified).** `A_T=228.1, B_T=0.1641, C_T=0.1728`: 0.1641 looks like a `C` parameter from a published set being used as an intercept, and the NIR set (1528 / 0.3742) needs checking. Nechad's model takes water-leaving reflectance, while Earth Engine SR is bottom-of-atmosphere reflectance incl. glint.
  Fix: check each constant against the paper's table for the chosen wavelength; document the reflectance-definition mismatch.
- [x] **2.6 2BDM/3BDM do not match the plan.** Plan: `2BDM = B5/B4`. Code: `1/B4 − 1/B5`. `3BDM` needs B6 (740 nm), which is not exported, so the code silently substitutes B8 (842 nm).
  Fix: implement the plan's formulas; export B6 or drop `bdm3` — no silent substitution.
- [x] **2.7 MNDWI threshold per site.** Default 0.0 everywhere.
  Fix: store `mndwi_threshold` per site in [sites.yaml](config/sites.yaml) from `suggest_threshold`/Otsu, and record how it was chosen.

## Phase 3 — Models

- [x] **3.1 The "temporal" model has no temporal input (D3).** Dataset yields shape `(1, 12)`; `window_size` is stored but never used. Verified: attention weights are exactly 1.0 for every sample; conv kernels 3/5, BiLSTM and attention are no-ops, so the model is an MLP.
  Fix: build per-site dated windows `(T, 12)` with T ≥ 3 (needs regular multi-date observations — make the synthetic generator emit seasonal per-site time series), **or** rename the model and remove every temporal claim.
  Done when: a test asserts sequence length > 1 and attention weights are not all 1.0 (or the temporal claims are gone).
- [x] **3.2 Targets are unscaled.** Chl-a mean ≈ 224 vs DO ≈ 2 with Huber δ = 1, so Chl-a dominates the loss.
  Fix: standardise (or log-transform) targets, invert in `predict`, weight the loss per target.
- [x] **3.3 Silent bad inputs.** `AquaSenseDataset` fills a missing column with 0.0 ([dl_model.py:73](src/models/dl_model.py#L73)); a missing `temp_surface` becomes z ≈ −7.7 after scaling (verified). `predict()` returns garbage from untrained weights with only a warning when `.pt`/`.pkl` are absent (both are gitignored; verified outputs ≈ 0).
  Fix: raise on missing feature columns; store train medians for NaN filling; bundle weights + feature scaler + target scaler in one artifact; raise if it is missing.
- [x] **3.4 Single source of truth for the model interface.** `FEATURE_COLS`/`TARGET_COLS` are duplicated in `dl_model.py` and `xgboost_pipeline.py`. The old `assign_spatial_folds`/`get_fold_splits` contract in `spatial_cv.py` was replaced by `SpatialKFold`, and `SpatialKFold` hardcodes `["lat","lon"]` ignoring `lat_col`/`lon_col`. XGBoost clip bound for Chl-a is 500 while labels reach 668.
  Fix: one `schema.py`; use `self.lat_col/lon_col`; derive clip bounds from data or documented limits; add a module-level `predict(df)` to `xgboost_pipeline.py` matching the plan's handoff contract.
- [x] **3.5 Spatial CV wording (D4).** The code uses its own KMeans site blocks, not the `spatial-kfold` library named in the plan/pitch. Update all wording, or wrap the library.
- [x] **3.6 Lazy imports.** [src/models/__init__.py](src/models/__init__.py) eagerly imports xgboost, optuna and shap, so anything importing `src.models.*` (dashboard, DL script) pulls them all. Import lazily.
- [x] **3.7 Model selection.** One script trains/evaluates both models on the same folds and writes the comparison table (feeds 0.3/1.6); winner chosen by out-of-fold RMSE.

## Phase 4 — Pipeline integration

- [x] **4.1 Band contract.** Verified failures: river rasters (B2,B3,B4,B8 only) → `compute_all_features` raises `KeyError: 'B5'`; Landsat rasters (B3,B4,B5,B6) → `KeyError: 'B8'`. Worse, Landsat B5 is NIR and must never be treated as red-edge. Landsat has no red-edge band at all, so NDCI/Chl-a cannot come from Landsat.
  Fix: one band-contract table in the repo. Rivers: keep B5/B6/B11 but resample to 10 m, and flag Chl-a as low-confidence. Landsat: turbidity/MNDWI only (gap-filler for turbidity, not Chl-a). `compute_all_features` gets a per-sensor path and raises a clear error listing missing bands.
- [x] **4.2 Sensor codes.** The acquisition code emits `S2`/`LS`; the join only knows `S2`/`L8`/`L9`/`LANDSAT`, so `LS` silently gets a 3-day tolerance instead of 5 (verified).
  Fix: one shared sensor enum/constants module.
- [x] **4.3 `temp_surface` has no producer.** Not exported by the raster pipeline, not produced by `compute_all_features`.
  Fix: add it as a band from a real source (e.g. ERA5-Land daily temperature in Earth Engine, or Landsat ST_B10 where available); missing → NaN, never 0.
- [x] **4.4 Missing script: rasters → training table.** Nothing turns exported `.tif` files into `train.parquet`.
  Fix: `scripts/build_training_table.py`: read tif → per-pixel frame with lat/lon from the transform → `compute_all_features` → `spatial_temporal_join` → parquet in the agreed schema. The join's outputs are named `chl_a_gt`, `turbidity_gt`, … but the agreed schema expects `chl_a`, `turbidity`, `do`, `wqi`; fix the mapping. Vectorise the join (row-wise `apply` over every pixel × every station will not scale) and filter by date before distance.
- [x] **4.5 Export size (unverified).** `gee.export_scene` used `geemap.ee_export_image` (now Earth Engine's own download URL; geemap removed). At 10 m, most river bounding boxes in [sites.yaml](config/sites.yaml) would exceed Earth Engine's direct-download size cap (estimated from bbox × bands × 4 bytes).
  Fix: estimate bytes first; tile large AOIs or use `ee.batch.Export` to Drive/GCS.
- [ ] **4.6 Atmospheric correction is not connected to anything.** `correct_scene` returns file paths; nothing converts ACOLITE/C2RCC outputs to the `{site}_{sensor}_{date}.tif` contract, so the downstream pipeline only ever sees the Earth Engine fallback. "Routing" is a static site-type rule, not the per-pixel dynamic routing the pitch describes.
  Fix: converter from ACOLITE/C2RCC outputs to the band contract (+ water mask); a `correction_method` provenance column; either implement a simple data-driven routing rule or reword the pitch as "site-type based".
- [ ] **4.7 C2RCC command line (unverified).** [correction.py:64](src/preprocessing/correction.py#L64) uses `-Ssource=` and `-Pregion=`; SNAP's `c2rcc.msi` may take `-SsourceProduct=`, may not accept `-Pregion`, and typically needs a resampled/subset product first.
  Fix: run `gpt c2rcc.msi -h`, correct the arguments, and execute one real scene end-to-end. This is the still-open pre-hackathon dry run.
- [x] **4.8 Dashboard ↔ raster integration.** The dashboard never imports `raster_layers`; the map shows one circle per site, not pixel maps. Upload-AOI only previews the polygon and then selects **all** sites (D5).
  Fix: if prediction rasters exist for a site, render them; otherwise show markers with an explicit "no raster" note. Label uploaded-AOI as preview only, or implement fetch → predict as a stretch.
- [x] **4.9 `predict_raster` band handling.** It raises on band-count mismatch but has no river/Landsat band lists; wire it to the band contract in 4.1.

## Phase 5 — Repo hygiene and tests

- [x] **5.1 `.gitignore` regression.** The latest change removed `.venv/`, `.pytest_cache/`, `*.pyc` and `*.tif` (a `.venv` exists in the working directory and would show as thousands of untracked files) and added `data/processed/`, `data/ground_truth/`, which hide new files in folders that contain tracked demo data.
  Fix: restore the removed lines; decide which data files are tracked demo assets and un-ignore them.
- [x] **5.2 Two synthetic-data generators.** [generate_synthetic_train.py](scripts/generate_synthetic_train.py) and [generate_synthetic_data.py](scripts/generate_synthetic_data.py) both write `data/processed/train.parquet`; README references both.
  Fix: keep one (the fixed one from 1.1); delete the other; update README and the error message in `train_models.py`.
- [x] **5.3 Weak or vacuous tests.**
  - `test_loss_decreases_over_epochs` only asserts `> 0`.
  - `test_wqi_above_100` asserts `> 75`; `test_class_is_e` accepts D or E.
  - No tests for: feature formulas vs hand-computed values, the join (tolerance, `LS`, radius, <3-pair drop), masking logic, `predict_raster` round trip, `raster_layers`, the dashboard.
  Fix: tighten the two tests; add the missing ones; add `tests/test_no_label_leakage.py` (targets are not exact functions of features and are not in `FEATURE_COLS`).
- [x] **5.4 Dashboard smoke test.** Add an `AppTest`-based test that runs the app and asserts zero exceptions and that model output changes with the radio.
- [x] **5.5 CI (green on GitHub as of 2026-09-20).** GitHub Actions: fresh venv, `pip install -r requirements.txt`, `pytest`, dashboard smoke test.

## Phase 6 — Correct the project plan document

Update [AQUA_SENSE_PROJECT_PLAN.md](AQUA_SENSE_PROJECT_PLAN.md):
- [x] River rule: "10 m bands only" contradicts NDCI needing B5; replace with the 4.1 contract.
- [x] Landsat is not a Chl-a gap-filler (no red-edge band); state its actual role.
- [x] Define the WQI scale and the separate best-use class exactly as implemented (2.2).
- [x] State the label strategy and the disclosure wording (1.3, D1).
- [x] Replace "spatial-kfold library" wording per D4.
- [x] Pitch claims: "R² > 0.85" only if it comes from out-of-fold predictions on real data; SHAP statement must reflect what SHAP shows (NDCI dominates; B5 enters through NDCI).
- [x] Mark status of each deliverable, and record which claims are backed by evidence.

## Phase 7 — Final acceptance check

- [x] Clean clone → new venv → `pip install -r requirements.txt` → `pytest` all green. *(Verified 2026-09-19 from commit 4735438: install OK, 253 passed.)*
- [x] `streamlit run src/app/streamlit_app.py`: no exceptions, banner visible, model radio changes displayed values, no hardcoded numbers. *(Verified headless with Streamlit AppTest in the clean clone.)*
- [x] Leakage test passes; XGBoost and DL metrics both come from out-of-fold predictions on the same folds; no metric is negative R² without an explanation. *(Both come from `SpatialKFold` out-of-fold predictions; the negative/near-zero R² values are explained by the skill labels and baselines in the dashboard and `data_source_log.md`.)*
- [x] Every number quoted in the pitch can be traced to a file in `reports/`.
- [x] One real Sentinel-2 scene runs through the correction chain and into a raster prediction, or the dashboard/pitch states plainly that Earth Engine surface reflectance is used. *(Not run through ACOLITE/C2RCC. A real scene does run into raster predictions via Planetary Computer L2A (`src/data/aoi.py`); every document says plainly that Sen2Cor L2A, not Earth Engine or ACOLITE, was used.)*

---

## Appendix — evidence from the review

| Check | Result |
|---|---|
| `chl_a` vs formula of `ndci` | max abs diff 3.05e-05 |
| `turbidity` vs formula of `B4`,`B8` | max abs diff 0.0 |
| Site-grouped gradient boosting R² (Chl-a / turbidity / DO) | 0.998 / 0.991 / 0.891 |
| DL held-out R² (Chl-a / turbidity / DO) | 0.908 / −1.341 / −0.509 |
| Gradient boosting on the DL script's split | 0.999 / 1.000 / 0.956 |
| DO surrogate at 10/20/25/28/30/35 °C vs true saturation | 10.72/6.82/4.88/3.71/2.93/0.98 vs 11.29/9.09/8.26/7.83/7.56/6.95 |
| `train.parquet` WQI | min 204.9, median 607.9, max 1458.4; 390/390 rows Class E |
| `LS` sensor tolerance in the join | 3 days (plan: 5) |
| River bands into `compute_all_features` | `KeyError: 'B5'` |
| Landsat bands into `compute_all_features` | `KeyError: 'B8'` |
| `add_raster_overlay` on matplotlib 3.11 | `AttributeError: get_cmap` |
| Dashboard on fresh install (plotly 7.1, no streamlit-folium) | `AttributeError: scatter_mapbox` |
| DL attention weights | 1.0 for all 8 sampled rows (sequence length 1) |
| Missing `temp_surface` at inference | scaled to z ≈ −7.69 |
| Real SHAP, Chl-a | `ndci` ≈ 134, `red_green` ≈ 31, `B5` ≈ 5 |
| Real SHAP, turbidity | `B4` ≈ 2.1, others < 0.3 |
| XGBoost `metrics_summary.json` R² | 0.98–0.999, computed on the training data |
| Tests | 70 pass; none cover the items above |
