# Aqua-Sense

**Spatiotemporal AI and Edge-Spectral Deep Learning for Water Quality Indexing of Indian Inland Waters**

Predict Chlorophyll-a, Turbidity, Dissolved Oxygen and a CPCB Water Quality Index (WQI) for any inland water body in India from Sentinel-2 and Landsat-8/9 imagery.

## Team
- **Aadi (P1)** — Geospatial / Data Engineer
- **Marutey (P2)** — Data Scientist
- **Navya (P3)** — ML Engineer A
- **Jashan (P4)** — ML Engineer B / Full-Stack

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Generate synthetic training data (Hour-2 deliverable)
python scripts/generate_synthetic_train.py

# Run WQI engine tests
python -m pytest tests/test_wqi_engine.py -v
```

See `AQUA_SENSE_PROJECT_PLAN.md` for full details.
