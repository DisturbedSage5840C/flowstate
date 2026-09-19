"""Model selection from out-of-fold metrics (XGBoost vs DL vs baselines), one decision per target."""

from __future__ import annotations

import numpy as np
import pandas as pd

CANDIDATES = ("xgboost_oof", "dl_oof")
BASELINES = ("baseline_mean", "baseline_median")

# Convention for labelling out-of-fold R^2 (on the log scale for heavy-tailed targets, where reported).
SKILL_BINS = ((0.60, "good"), (0.30, "moderate"), (0.05, "weak"))


def skill_label(r2: float) -> str:
    if r2 is None or not np.isfinite(r2):
        return "unknown"
    for threshold, label in SKILL_BINS:
        if r2 > threshold:
            return label
    return "none"


def compare_models(xgb_metrics: pd.DataFrame, dl_metrics: pd.DataFrame | None = None,
                   water_type: str = "overall", clear_margin: float = 0.05,
                   baseline_margin: float = 0.02) -> pd.DataFrame:
    """One row per target: metrics of every model plus a production decision.

    * XGBoost is the default; the DL model is chosen only if its out-of-fold RMSE is lower by more than
      ``clear_margin`` (relative), as the project plan prescribes.
    * ``beats_baseline`` is True only when the chosen model's RMSE is at least ``baseline_margin`` (2 %) below the best
      "predict the training average" baseline on the same folds; a marginal edge is not reported as skill.
    * ``skill`` labels the chosen model's out-of-fold R^2 (log-scale R^2 for heavy-tailed targets when available):
      none <= 0.05 < weak <= 0.30 < moderate <= 0.60 < good.
    """
    frames = [xgb_metrics]
    if dl_metrics is not None:
        frames.append(dl_metrics)
    m = pd.concat(frames, ignore_index=True)
    m = m[m["water_body_type"] == water_type].drop_duplicates(["model", "target"], keep="first")

    rows = []
    for target, g in m.groupby("target"):
        by_model = g.set_index("model")
        row = {"target": target}
        for name in CANDIDATES + BASELINES:
            if name in by_model.index:
                row[f"{name}_R2"] = float(by_model.loc[name, "R2"])
                row[f"{name}_RMSE"] = float(by_model.loc[name, "RMSE"])
                if "R2_log" in by_model.columns:
                    row[f"{name}_R2_log"] = float(by_model.loc[name, "R2_log"])
                # Passthrough for screening/classifier rows sharing this table (src.models.metrics.classification_scores):
                # never used for the regression production-model decision below, only reported alongside it.
                for auc_col in ("auc", "ap", "lift"):
                    if auc_col in by_model.columns:
                        row[f"{name}_{auc_col}"] = float(by_model.loc[name, auc_col])
        row["n"] = int(by_model["n"].max()) if "n" in by_model else None

        available = [c for c in CANDIDATES if f"{c}_RMSE" in row]
        choice = "xgboost_oof" if "xgboost_oof_RMSE" in row else (available[0] if available else None)
        if "dl_oof_RMSE" in row and "xgboost_oof_RMSE" in row:
            if row["dl_oof_RMSE"] < (1.0 - clear_margin) * row["xgboost_oof_RMSE"]:
                choice = "dl_oof"
        base_rmse = min((row[f"{b}_RMSE"] for b in BASELINES if f"{b}_RMSE" in row), default=np.nan)
        row["production_model"] = choice
        if choice is not None:
            row["chosen_RMSE"] = row[f"{choice}_RMSE"]
            row["chosen_R2"] = row[f"{choice}_R2"]
            r2_log = row.get(f"{choice}_R2_log", np.nan)
            row["chosen_R2_log"] = r2_log
            row["beats_baseline"] = bool(np.isfinite(base_rmse)
                                         and row[f"{choice}_RMSE"] < (1.0 - baseline_margin) * base_rmse)
            row["skill"] = skill_label(r2_log if np.isfinite(r2_log) else row["chosen_R2"])
        rows.append(row)
    return pd.DataFrame(rows)
