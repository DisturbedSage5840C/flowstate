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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Optional


# Target parameters Navya's models predict
TARGET_COLS = ["chl_a", "turbidity", "do"]
WATER_BODY_TYPES = ["lake", "river", "reservoir", "lagoon"]


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² can be negative; return NaN if fewer than 2 samples."""
    if len(y_true) < 2:
        return float("nan")
    return float(r2_score(y_true, y_pred))


class MetricsReporter:
    """Compute and format spatial-CV evaluation metrics.

    Parameters
    ----------
    targets : list[str]
        Target column names to evaluate (default: chl_a, turbidity, do).
    type_col : str
        Column in df identifying water body type.
    """

    def __init__(
        self,
        targets: list[str] | None = None,
        type_col: str = "water_body_type",
    ):
        self.targets = targets or TARGET_COLS
        self.type_col = type_col

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

        for target in self.targets:
            if target not in df_true.columns:
                continue
            y_true_all = df_true[target].values
            y_pred_all = df_pred[target].values

            # ── Overall row ───────────────────────────────────────────
            rows.append(
                self._row(target, "overall", y_true_all, y_pred_all)
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
                    )
                )

        result = pd.DataFrame(rows).sort_values(["target", "water_body_type"])
        return result.reset_index(drop=True)

    def print_table(self, table: pd.DataFrame) -> None:
        """Pretty-print the metrics table to stdout."""
        print("\n" + "=" * 72)
        print("  AQUA-SENSE · Spatial-CV Metrics (per water-body type)")
        print("=" * 72)
        fmt = "{:<12} {:<14} {:>6} {:>8} {:>10} {:>10}"
        header = fmt.format("Target", "Water body type", "N", "R²", "RMSE", "MAE")
        print(header)
        print("-" * 72)
        for _, row in table.iterrows():
            r2_str = f"{row['R2']:.4f}" if not np.isnan(row["R2"]) else "  N/A"
            print(
                fmt.format(
                    row["target"],
                    row["water_body_type"],
                    int(row["n"]),
                    r2_str,
                    f"{row['RMSE']:.4f}",
                    f"{row['MAE']:.4f}",
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
    ) -> dict:
        return {
            "target": target,
            "water_body_type": wbt,
            "n": len(y_true),
            "R2": _safe_r2(y_true, y_pred),
            "RMSE": _rmse(y_true, y_pred),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
        }


# ---------------------------------------------------------------------------
# Standalone helper (used by Optuna objective)
# ---------------------------------------------------------------------------

def spatial_cv_rmse(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """Return RMSE — signature compatible with sklearn scoring callables."""
    return _rmse(y_true, y_pred)
