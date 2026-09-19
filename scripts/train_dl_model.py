"""
train_dl_model.py — Training script for AquaSense CNN-BiLSTM-Attention model.

Usage:
    python scripts/train_dl_model.py
"""

import sys
sys.path.insert(0, ".")

import json
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH   = Path("data/processed/train.parquet")
REPORTS_DIR = Path("reports")
MODEL_PATH  = REPORTS_DIR / "dl_model.pt"
SCALER_PATH = REPORTS_DIR / "dl_scaler.pkl"
METRICS_PATH = REPORTS_DIR / "metrics.json"
METRICS_TABLE_PATH = REPORTS_DIR / "dl_metrics_table.csv"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Imports from src ──────────────────────────────────────────────────────────
from src.models.dl_model import (
    AquaSenseDLModel,
    AquaSenseTrainer,
    FEATURE_COLS,
    TARGET_COLS,
)
from src.models.spatial_cv import SpatialKFold
from src.models.metrics import MetricsReporter

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS      = 60
RANDOM_SEED = 42
N_FOLDS     = 5   # site-blocked folds; fold 0 -> test, fold 1 -> val, rest -> train

np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────
print(f"Loading data from {DATA_PATH} ...")
df = pd.read_parquet(DATA_PATH)
print(f"  Loaded {len(df)} rows × {df.shape[1]} cols")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Site-blocked split (same SpatialKFold as the XGBoost pipeline): every
# site's rows land entirely in one of train/val/test, so val/test measure
# generalization to unseen water bodies rather than leaking spatial
# autocorrelation between rows of the same site.
# ─────────────────────────────────────────────────────────────────────────────
skf = SpatialKFold(n_folds=N_FOLDS, random_state=RANDOM_SEED)
fold_labels = skf.get_fold_labels(df)

test_mask  = fold_labels == 0
val_mask   = fold_labels == 1
train_mask = ~(test_mask | val_mask)

test_df  = df[test_mask].copy()
val_df   = df[val_mask].copy()
train_df = df[train_mask].copy()

print(f"\nTest  sites: {sorted(df.loc[test_mask, 'site'].unique().tolist())}")
print(f"Val   sites: {sorted(df.loc[val_mask, 'site'].unique().tolist())}")
print(f"Train sites: {sorted(df.loc[train_mask, 'site'].unique().tolist())}")
print(f"\nSplit sizes — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Build model + scaler, train for 60 epochs
# ─────────────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
model  = AquaSenseDLModel()

trainer = AquaSenseTrainer(
    model,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=64,
    scaler=scaler,
)

print(f"\nTraining for {EPOCHS} epochs ...")
trainer.train(train_df, val_df, epochs=EPOCHS, save_path=str(MODEL_PATH))
print(f"\nBest model weights saved -> {MODEL_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Save scaler
# ─────────────────────────────────────────────────────────────────────────────
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)
print(f"Scaler saved -> {SCALER_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Evaluate on test set (load best weights)
# ─────────────────────────────────────────────────────────────────────────────
import torch

# Reload best checkpoint
best_model = AquaSenseDLModel()
best_model.load_state_dict(torch.load(str(MODEL_PATH), map_location="cpu"))

best_trainer = AquaSenseTrainer(
    best_model,
    batch_size=64,
    scaler=scaler,
)

# Get raw predictions for computing metrics
def get_preds_targets(trainer_obj, split_df):
    loader = trainer_obj._make_loader(split_df, shuffle=False, fit_scaler=False)
    _, preds, targets = trainer_obj._step(loader, train=False)
    return preds, targets

# Overall test metrics
test_preds, test_targets = get_preds_targets(best_trainer, test_df)

# Reset index so row order matches test_preds/test_targets (built with
# shuffle=False), then hand off to the same MetricsReporter XGBoost uses
# (src/models/metrics.py) so both models are scored with identical,
# NaN-aware R2/RMSE/MAE code -- no separately hand-rolled safe_r2/safe_rmse.
test_df_aligned = test_df.reset_index(drop=True)
df_true = test_df_aligned[TARGET_COLS].copy()
if "water_body_type" in test_df_aligned.columns:
    df_true["water_body_type"] = test_df_aligned["water_body_type"]
else:
    df_true["water_body_type"] = "unknown"
df_pred = pd.DataFrame(test_preds, columns=TARGET_COLS)

reporter = MetricsReporter(targets=TARGET_COLS)
metrics_table = reporter.report(df_true, df_pred)
metrics_table.to_csv(METRICS_TABLE_PATH, index=False)
reporter.print_table(metrics_table)

# Dashboard-facing summary derived from the same tidy table (no separate
# computation) -- overall row per target, plus a per-type pivot.
overall = {}
for target in TARGET_COLS:
    row = metrics_table[
        (metrics_table["target"] == target) & (metrics_table["water_body_type"] == "overall")
    ].iloc[0]
    overall[f"{target}_r2"] = float(row["R2"])
    overall[f"{target}_rmse"] = float(row["RMSE"])

per_type = {}
for wtype in ("lake", "river"):
    wtype_rows = metrics_table[metrics_table["water_body_type"] == wtype]
    if wtype_rows.empty:
        per_type[wtype] = {f"{c}_r2": None for c in TARGET_COLS}
        continue
    type_metrics = {}
    for target in TARGET_COLS:
        match = wtype_rows[wtype_rows["target"] == target]
        type_metrics[f"{target}_r2"] = float(match.iloc[0]["R2"]) if len(match) else None
    per_type[wtype] = type_metrics

# ─────────────────────────────────────────────────────────────────────────────
# 7. Save metrics.json
# ─────────────────────────────────────────────────────────────────────────────
metrics = {
    "model": "CNN-BiLSTM-Attention",
    "overall": overall,
    "per_type": per_type,
    "n_train": int(len(train_df)),
    "n_val":   int(len(val_df)),
    "n_test":  int(len(test_df)),
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nMetrics saved -> {METRICS_PATH}")
print(f"Tidy metrics table (same schema as XGBoost's metrics_table.csv) -> {METRICS_TABLE_PATH}")
print("\nFinal metrics:")
print(json.dumps(metrics, indent=2))
print("\n[OK] Training complete.")
