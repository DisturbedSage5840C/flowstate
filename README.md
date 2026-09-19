# Aqua-Sense

**Water quality of Indian inland waters from CPCB measurements and Sentinel-2 imagery**

Measured DO, BOD and turbidity (CPCB National Water Quality Monitoring, ~3,100 surface-water stations) are matched to
Sentinel-2 L2A reflectance and validated with site-blocked spatial cross-validation against "predict the average"
baselines. A satellite-adapted WQI (0-100) and the official CPCB designated-best-use class (A-E) are computed from
measured or predicted parameters.

**Headline result: pollution screening, not exact concentrations.** DO and BOD are not optically active — reflectance
cannot see dissolved oxygen or sewage loading directly, and a variance decomposition confirms 85-100% of each
target's variance is between-station (set by local discharge, invisible from space), not something one satellite
visit can resolve. Exact-value regression on those targets is near-zero skill (honestly reported, not hidden) and is
kept only as a secondary, flagged result. What the same data *can* do well is a binary pollution screen — "is this
station's water likely polluted, for an inspector to prioritise" — out-of-fold **AUC ≈ 0.72-0.78** across BOD>3 mg/L
(CPCB Class C limit), BOD>6 mg/L, DO<4 mg/L and CPCB-class screens, plus a station-level BOD ranking (the strongest
regression result in the project). See `AQUA_SENSE_PROJECT_PLAN.md` §11 and `reports/real/screening_metrics.json`
for the full evidence.

## Team
- **Aadi (P1)** — Geospatial / Data Engineer
- **Marutey (P2)** — Data Scientist
- **Navya (P3)** — ML Engineer A
- **Jashan (P4)** — ML Engineer B / Full-Stack

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows; use source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

No logins are needed for the real-data pipeline (CPCB NWDP and Planetary Computer are open).
Optional: Kaggle (`~/.kaggle/access_token`) and Google Earth Engine (step-by-step in `docs/EARTH_ENGINE_SETUP.md`).
**Never commit credentials** (`.gitignore` blocks common token file names).

## Real-data pipeline

```bash
python -m scripts.fetch_insitu                      # 1. download CPCB surface-water CSVs -> data/ground_truth/insitu_master.parquet
python -m scripts.extract_satellite --per-state 100 # 2. Sentinel-2 reflectance at station-visits (resumable, ~1 s/visit)
python -m scripts.build_real_training_table         # 3. join -> data/processed/train_real.parquet + reports/real/dataset_summary.json
python -m scripts.validate_empirical_formulas       # 4. test literature formulas against the real measurements
python -m scripts.validate_landsat_temperature      #    (optional, needs a Kaggle token) Landsat temperature vs in-situ IoT sensors
python -m scripts.train_real_models --trials 30     # 5. secondary: DO/BOD/turbidity regression, out-of-fold, SHAP -> reports/real/
python -m scripts.train_screening                   # 6. headline: pollution-screening classifiers -> reports/real/screening_metrics.json
streamlit run src/app/streamlit_app.py              # dashboard (real-data mode when train_real.parquet exists; see the Screening tab)
```

What the real data does and does not contain (details in `data/ground_truth/data_source_log.md`):
DO, BOD, pH, coliform, conductivity and turbidity are measured; **chlorophyll-a and water temperature are not**.
Chl-a values in the repo are formula estimates (`chl_a_empirical`) and are not validated. Season and urban-proximity
context (`src/models/schema.py::FEATURE_SETS`) feed the BOD models; antecedent rainfall was fetched and validated
(`src/data/weather.py`) but tested out of every model — it stays in the parquet as unused data.

## Other commands

```bash
python -m pytest tests/ -v                          # all tests
python scripts/generate_synthetic_train.py          # synthetic DEMO data only (used when no real table exists)
python scripts/train_models.py --trials 10          # models on the synthetic table
```

See `AQUA_SENSE_PROJECT_PLAN.md` for the plan and `FIX_PLAN.md` for the review checklist.
