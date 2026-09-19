"""
src/models/metrics.py
=====================
Evaluation metrics for water-quality model predictions.

Computes R², RMSE, and MAE:
  * per water-body type  (lake / river / reservoir / lagoon)
  * per target parameter (chl_a / turbidity / do)
  * overall (all types combined)

Usage
-----
    from src.models.metrics import MetricsReporter
    reporter = MetricsReporter()
    table = reporter.report(df_true, df_pred)
    reporter.print_table(table)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from typing import Optional


# Target parameters Navya's models predict
TARGET_COLS = ["chl_a", "turbidity", "do"]
WATER_BODY_TYPES = ["lake", "river", "reservoir", "lagoon"]


def _drop_nan_targets(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop rows with missing ground truth (some targets aren't measured for every row)."""
    mask = ~np.isnan(y_true)
    return y_true[mask], y_pred[mask]


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _drop_nan_targets(y_true, y_pred)
    if len(y_true) == 0:
        return float("nan")
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² can be negative; return NaN if fewer than 2 (non-missing) samples."""
    y_true, y_pred = _drop_nan_targets(y_true, y_pred)
    if len(y_true) < 2:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def _safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _drop_nan_targets(y_true, y_pred)
    if len(y_true) == 0:
        return float("nan")
    return float(mean_absolute_error(y_true, y_pred))


def _constant_within_every_group(y_pred: np.ndarray, groups: np.ndarray) -> bool:
    """True when every group has at most one distinct (non-NaN) predicted value."""
    s = pd.Series(y_pred, index=pd.Index(groups))
    return bool(s.groupby(level=0).nunique(dropna=True).le(1).all())


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray, groups: Optional[np.ndarray] = None) -> float:
    """Rank correlation: the honest headline number for heavy-tailed targets, where raw R2
    can be near-zero/negative even when the model recovers useful relative ordering.

    ``groups`` (e.g. site): a constant "predict the training-fold mean/median" baseline has no
    within-group rank information at all -- every row in a group gets the identical prediction -- so a
    pooled Spearman over such predictions is an artifact of between-group offsets, not genuine rank
    skill, and reports large-magnitude values (observed -0.2 to -0.38) that are meaningless. When every
    group's prediction is constant, return NaN instead.
    """
    mask = ~np.isnan(np.asarray(y_true, dtype=float))
    if groups is not None:
        groups = np.asarray(groups)[mask]
    y_true, y_pred = _drop_nan_targets(y_true, y_pred)
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    if groups is not None and len(groups) == len(y_pred) and _constant_within_every_group(y_pred, groups):
        return float("nan")
    return float(spearmanr(y_true, y_pred).correlation)


class MetricsReporter:
    """Compute and format spatial-CV evaluation metrics.

    Parameters
    ----------
    targets : list[str]
        Target column names to evaluate (default: chl_a, turbidity, do).
    type_col : str
        Column in df identifying water body type.
    group_col : str
        Column in df_true identifying the group used to detect a per-group-constant prediction (a
        "predict the fold mean/median" baseline), whose pooled Spearman is an artifact (see
        _safe_spearman). Ignored when not present in df_true.
    """

    def __init__(
        self,
        targets: list[str] | None = None,
        type_col: str = "water_body_type",
        group_col: str = "site",
    ):
        self.targets = targets or TARGET_COLS
        self.type_col = type_col
        self.group_col = group_col

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def report(
        self,
        df_true: pd.DataFrame,
        df_pred: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return a tidy DataFrame with rows = (target, water_body_type).

        Parameters
        ----------
        df_true : pd.DataFrame
            Must contain [water_body_type] + target columns with true values.
        df_pred : pd.DataFrame
            Same index as df_true; contains predicted values for each target.
            Column names: same as self.targets  (e.g. "chl_a", "turbidity").

        Returns
        -------
        pd.DataFrame with columns [target, water_body_type, n, R2, RMSE, MAE]
        """
        rows = []
        groups_all = df_true[self.group_col].values if self.group_col in df_true.columns else None

        for target in self.targets:
            if target not in df_true.columns:
                continue
            y_true_all = df_true[target].values
            y_pred_all = df_pred[target].values

            # ── Overall row ───────────────────────────────────────────
            rows.append(
                self._row(target, "overall", y_true_all, y_pred_all, groups_all)
            )

            # ── Per water-body type ───────────────────────────────────
            for wbt in df_true[self.type_col].unique():
                mask = df_true[self.type_col] == wbt
                if mask.sum() == 0:
                    continue
                rows.append(
                    self._row(
                        target,
                        wbt,
                        y_true_all[mask.values],
                        y_pred_all[mask.values],
                        groups_all[mask.values] if groups_all is not None else None,
                    )
                )

        result = pd.DataFrame(rows).sort_values(["target", "water_body_type"])
        return result.reset_index(drop=True)

    def print_table(self, table: pd.DataFrame) -> None:
        """Pretty-print the metrics table to stdout."""
        print("\n" + "=" * 72)
        print("  AQUA-SENSE · Spatial-CV Metrics (per water-body type)")
        print("=" * 72)
        fmt = "{:<12} {:<14} {:>6} {:>8} {:>10} {:>10} {:>10}"
        header = fmt.format("Target", "Water body type", "N", "R²", "RMSE", "MAE", "Spearman")
        print(header)
        print("-" * 72)
        for _, row in table.iterrows():
            r2_str = f"{row['R2']:.4f}" if not np.isnan(row["R2"]) else "  N/A"
            sp_str = f"{row['spearman']:.4f}" if not np.isnan(row["spearman"]) else "  N/A"
            print(
                fmt.format(
                    row["target"],
                    row["water_body_type"],
                    int(row["n"]),
                    r2_str,
                    f"{row['RMSE']:.4f}",
                    f"{row['MAE']:.4f}",
                    sp_str,
                )
            )
        print("=" * 72 + "\n")

    def to_latex(self, table: pd.DataFrame) -> str:
        """Export metrics table as LaTeX (for reports)."""
        return table.to_latex(index=False, float_format="%.4f")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _row(
        self,
        target: str,
        wbt: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> dict:
        from src.models.schema import LOG_TARGETS

        row = {
            "target": target,
            "water_body_type": wbt,
            "n": int(np.sum(~np.isnan(y_true))),
            "R2": _safe_r2(y_true, y_pred),
            "RMSE": _rmse(y_true, y_pred),
            "MAE": _safe_mae(y_true, y_pred),
            "spearman": _safe_spearman(y_true, y_pred, groups),
        }
        # Heavy-tailed targets: raw-scale R2/RMSE are dominated by a few extreme values, so also report R2 on the
        # log1p scale (NaN for other targets).
        if target in LOG_TARGETS:
            row["R2_log"] = _safe_r2(np.log1p(np.maximum(y_true, 0.0)), np.log1p(np.maximum(y_pred, 0.0)))
        else:
            row["R2_log"] = float("nan")
        return row


# ---------------------------------------------------------------------------
# Standalone helper (used by Optuna objective)
# ---------------------------------------------------------------------------

def spatial_cv_rmse(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """Return RMSE — signature compatible with sklearn scoring callables."""
    return _rmse(y_true, y_pred)


# ---------------------------------------------------------------------------
# Classification scoring (used by src.models.tier_classifier.ScreeningClassifier)
# ---------------------------------------------------------------------------

def classification_scores(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """AUC and average precision (lift over the positive base rate) for a binary screening target.

    Accuracy is not reported: screening targets (e.g. BOD > 3 mg/L) are imbalanced (base rate often
    well under 50 %), so "predict everything negative" scores high accuracy while finding nothing.
    AUC and average-precision-vs-base-rate are the honest numbers for a ranked inspection shortlist.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    n = int(len(y_true))
    base_rate = float(y_true.mean()) if n else float("nan")
    if n < 2 or len(np.unique(y_true)) < 2:
        return {"n": n, "base_rate": base_rate, "auc": float("nan"), "ap": float("nan"), "lift": float("nan")}
    auc = float(roc_auc_score(y_true, y_score))
    ap = float(average_precision_score(y_true, y_score))
    return {"n": n, "base_rate": base_rate, "auc": auc, "ap": ap,
            "lift": float(ap / base_rate) if base_rate > 0 else float("nan")}


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int | float) -> float:
    """Precision among the top ``k`` highest-scored rows (``k`` a count if int, a fraction of n if float
    in (0, 1]) -- the number that matters for an "inspect these first" shortlist."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    n = len(y_true)
    if n == 0:
        return float("nan")
    n_k = int(round(k * n)) if isinstance(k, float) and 0 < k <= 1 else int(k)
    n_k = max(1, min(n_k, n))
    top = np.argsort(-y_score)[:n_k]
    return float(y_true[top].mean())
