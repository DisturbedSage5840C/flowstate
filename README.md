# Aqua-Sense

**Water quality of Indian inland waters from CPCB measurements and Sentinel-2 imagery**

Measured DO, BOD and turbidity (CPCB National Water Quality Monitoring, ~3,100 surface-water stations) are matched to
Sentinel-2 L2A reflectance; XGBoost models are validated with site-blocked spatial cross-validation and compared
with "predict the average" baselines. A satellite-adapted WQI (0-100) and the official CPCB designated-best-use
class (A-E) are computed from measured or predicted parameters.

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
Optional: Kaggle (`~/.kaggle/access_token`) and Google Earth Engine (`earthengine authenticate`).
**Never commit credentials** (`.gitignore` blocks common token file names).

## Real-data pipeline

```bash
python -m scripts.fetch_insitu                      # 1. download CPCB surface-water CSVs -> data/ground_truth/insitu_master.parquet
python -m scripts.extract_satellite --per-state 100 # 2. Sentinel-2 reflectance at station-visits (resumable, ~1 s/visit)
python -m scripts.build_real_training_table         # 3. join -> data/processed/train_real.parquet + reports/real/dataset_summary.json
python -m scripts.validate_empirical_formulas       # 4. test literature formulas against the real measurements
python -m scripts.train_real_models --trials 30     # 5. XGBoost, out-of-fold spatial CV, baselines, SHAP -> reports/real/
streamlit run src/app/streamlit_app.py              # dashboard (real-data mode when train_real.parquet exists)
```

What the real data does and does not contain (details in `data/ground_truth/data_source_log.md`):
DO, BOD, pH, coliform, conductivity and turbidity are measured; **chlorophyll-a and water temperature are not**.
Chl-a values in the repo are formula estimates (`chl_a_empirical`) and are not validated.

## Other commands

```bash
python -m pytest tests/ -v                          # all tests
python scripts/generate_synthetic_train.py          # synthetic DEMO data only (used when no real table exists)
python scripts/train_models.py --trials 10          # models on the synthetic table
```

See `AQUA_SENSE_PROJECT_PLAN.md` for the plan and `FIX_PLAN.md` for the review checklist.
