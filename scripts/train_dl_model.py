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
from sklearn.metrics import r2_score, mean_squared_error

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH   = Path("data/processed/train.parquet")
REPORTS_DIR = Path("reports")
MODEL_PATH  = REPORTS_DIR / "dl_model.pt"
SCALER_PATH = REPORTS_DIR / "dl_scaler.pkl"
METRICS_PATH = REPORTS_DIR / "metrics.json"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Imports from src ──────────────────────────────────────────────────────────
from src.models.dl_model import (
    AquaSenseDLModel,
    AquaSenseTrainer,
    FEATURE_COLS,
    TARGET_COLS,
)
from src.models.spatial_cv import SpatialKFold

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

def safe_r2(y_true, y_pred):
    mask = ~np.isnan(y_true)
    if mask.sum() < 2:
        return 0.0
    return float(r2_score(y_true[mask], y_pred[mask]))

def safe_rmse(y_true, y_pred):
    mask = ~np.isnan(y_true)
    if mask.sum() == 0:
        return 0.0
    return float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))

overall = {}
for i, col in enumerate(TARGET_COLS):
    overall[f"{col}_r2"]   = safe_r2(test_targets[:, i], test_preds[:, i])
    overall[f"{col}_rmse"] = safe_rmse(test_targets[:, i], test_preds[:, i])

# Per-type metrics (on test set). A type absent from this particular
# test-fold draw (e.g. the fold happened to be all lakes) gets None/NaN,
# not a literal 0.0 -- 0.0 would misleadingly read as "the model scored
# zero on this type" when really it was never evaluated on it at all.
per_type = {}
if "water_body_type" in test_df.columns:
    for wtype in ("lake", "river"):
        mask_type = test_df["water_body_type"] == wtype
        if mask_type.sum() == 0:
            per_type[wtype] = {f"{c}_r2": None for c in TARGET_COLS}
            continue
        sub_df = test_df[mask_type].copy()
        # Align indices to preds array
        idx = test_df.index[mask_type.values].tolist()
        all_idx = test_df.index.tolist()
        row_positions = [all_idx.index(i) for i in idx]
        sub_preds   = test_preds[row_positions, :]
        sub_targets = test_targets[row_positions, :]
        type_metrics = {}
        for i, col in enumerate(TARGET_COLS):
            type_metrics[f"{col}_r2"] = safe_r2(sub_targets[:, i], sub_preds[:, i])
        per_type[wtype] = type_metrics
else:
    per_type = {
        "lake":  {f"{c}_r2": None for c in TARGET_COLS},
        "river": {f"{c}_r2": None for c in TARGET_COLS},
    }

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
print("\nFinal metrics:")
print(json.dumps(metrics, indent=2))
print("\n[OK] Training complete.")
