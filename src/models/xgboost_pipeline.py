"""
src/models/xgboost_pipeline.py
===============================
Optuna-tuned XGBoost pipeline for water quality parameter prediction.

Trains one model per target (chl_a, turbidity, do) using spatial K-fold CV
as the Optuna objective.  Saves best model artifacts to reports/models/.

Handoff contract (agreed with Jashan / P4)
------------------------------------------
    predict(df) -> pd.DataFrame with columns [chl_a, turbidity, do]

Usage — training
----------------
    from src.models.xgboost_pipeline import WaterQualityXGB
    xgb = WaterQualityXGB()
    xgb.train(df, n_trials=50)           # trains all three targets
    xgb.save("reports/models")

Usage — inference (dashboard)
------------------------------
    xgb = WaterQualityXGB.load("reports/models")
    preds = xgb.predict(df_new)          # → DataFrame[chl_a, turbidity, do]
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.metrics import mean_squared_error

from src.models.spatial_cv import SpatialKFold
from src.models.metrics import MetricsReporter, spatial_cv_rmse

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Feature columns (agree with Marutey's train.parquet schema) ──────────────
FEATURE_COLS = [
    "B2", "B3", "B4", "B5", "B8", "B11",
    "ndci", "bdm2", "bdm3", "red_green", "nir", "temp_surface",
]
TARGET_COLS = ["chl_a", "turbidity", "do"]
META_COLS   = ["site", "water_body_type", "lat", "lon", "date", "sensor"]

# ── Optuna search space ───────────────────────────────────────────────────────
SEARCH_SPACE = {
    "learning_rate":    (1e-2, 0.2),
    "max_depth":        (3, 6),
    "subsample":        (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "reg_lambda":       (1e-2, 10.0),
    "n_estimators":     (50, 200),
    "min_child_weight": (1, 6),
}


class WaterQualityXGB:
    """Train, tune, and serve XGBoost models for water quality parameters.

    Parameters
    ----------
    n_folds : int
        Spatial CV folds used inside the Optuna objective.
    random_state : int
        Global RNG seed for reproducibility.
    feature_cols : list[str]
        Input feature columns (default: FEATURE_COLS).
    targets : list[str]
        Target parameters to model (default: chl_a, turbidity, do).
    """

    def __init__(
        self,
        n_folds: int = 5,
        random_state: int = 42,
        feature_cols: list[str] | None = None,
        targets: list[str] | None = None,
    ):
        self.n_folds = n_folds
        self.random_state = random_state
        self.feature_cols = feature_cols or FEATURE_COLS
        self.targets = targets or TARGET_COLS
        self._models: dict[str, xgb.XGBRegressor] = {}
        self._best_params: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        n_trials: int = 50,
        verbose: bool = True,
    ) -> dict[str, float]:
        """Train one Optuna-tuned XGBoost model per target.

        Parameters
        ----------
        df : pd.DataFrame
            Full training table (must contain META_COLS + FEATURE_COLS + targets).
        n_trials : int
            Number of Optuna trials per target.
        verbose : bool
            Print per-target best RMSE.

        Returns
        -------
        dict[target → best spatial-CV RMSE]

        Note on selection bias: this RMSE is the minimum over `n_trials`
        Optuna trials, each scored on the same CV folds. Picking the best of
        many trials on the same folds is itself a (mild) form of overfitting
        to those folds, so this number is optimistically biased relative to
        performance on a truly held-out set. Use predict_oof() on a separate
        partition, or a nested CV, for an unbiased estimate.
        """
        results = {}
        for target in self.targets:
            if target not in df.columns:
                warnings.warn(f"Target '{target}' not in df; skipping.")
                continue
            if verbose:
                print(f"\n[XGB] Tuning for target: {target}  ({n_trials} trials)")
            best_rmse, best_params = self._tune(df, target, n_trials)
            # Refit on full data with best params (no early stopping without eval_set)
            model = self._build_model(best_params)
            X = df[self.feature_cols]
            y = df[target]
            model.fit(X, y)
            self._models[target] = model
            self._best_params[target] = best_params
            results[target] = best_rmse
            if verbose:
                print(f"  [OK] {target}: best spatial-CV RMSE = {best_rmse:.4f}")
        return results

    # ------------------------------------------------------------------
    # Inference — handoff interface with Jashan / dashboard
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return predictions for all targets.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain FEATURE_COLS.

        Returns
        -------
        pd.DataFrame with columns [chl_a, turbidity, do]
            Values are clipped to physically plausible ranges.
        """
        if not self._models:
            raise RuntimeError("No models trained yet.  Call .train() first.")
        preds = {}
        for target in self.targets:
            if target not in self._models:
                preds[target] = np.full(len(df), np.nan)
                continue
            raw = self._models[target].predict(df[self.feature_cols])
            preds[target] = self._clip(raw, target)
        return pd.DataFrame(preds, index=df.index)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model_dir: str | Path) -> None:
        """Save all trained models and best params to disk.

        Files saved:
            <model_dir>/<target>_xgb.json      ← XGBoost binary
            <model_dir>/<target>_params.json   ← Optuna best params
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        for target, model in self._models.items():
            model.save_model(model_dir / f"{target}_xgb.json")
            params_path = model_dir / f"{target}_params.json"
            with open(params_path, "w") as f:
                json.dump(self._best_params.get(target, {}), f, indent=2)
        print(f"[XGB] Models saved to {model_dir}")

    @classmethod
    def load(
        cls,
        model_dir: str | Path,
        targets: list[str] | None = None,
        feature_cols: list[str] | None = None,
    ) -> "WaterQualityXGB":
        """Load pre-trained models from disk.

        Parameters
        ----------
        model_dir : str | Path
            Directory containing *_xgb.json files.
        """
        model_dir = Path(model_dir)
        instance = cls(targets=targets, feature_cols=feature_cols)
        loaded_targets = targets or TARGET_COLS
        for target in loaded_targets:
            model_path = model_dir / f"{target}_xgb.json"
            if not model_path.exists():
                warnings.warn(f"Model file not found: {model_path}")
                continue
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            instance._models[target] = model
            params_path = model_dir / f"{target}_params.json"
            if params_path.exists():
                with open(params_path) as f:
                    instance._best_params[target] = json.load(f)
        print(f"[XGB] Loaded {len(instance._models)} model(s) from {model_dir}")
        return instance

    # ------------------------------------------------------------------
    # Out-of-fold prediction (honest per-type / per-fold evaluation)
    # ------------------------------------------------------------------

    def predict_oof(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return genuine out-of-fold predictions for every row in df.

        `predict()` uses the final model fit on the *entire* training set,
        so scoring it against the same df it was trained on is in-sample
        evaluation and inflates R²/RMSE/MAE. This instead refits a
        fold-local model per spatial-CV split (using the tuned params from
        `.train()`) and predicts only on that fold's held-out rows, so
        every prediction comes from a model that never saw that row during
        training. Use this for per-type/per-fold metrics on the training
        table; use `predict()` for genuinely new/unseen data.
        """
        if not self._best_params:
            raise RuntimeError("No tuned params available. Call .train() first.")
        skf = SpatialKFold(n_folds=self.n_folds, random_state=self.random_state)
        splits = list(skf.split(df))
        X = df[self.feature_cols]
        oof = {target: np.full(len(df), np.nan) for target in self.targets}
        for target in self.targets:
            if target not in self._best_params or target not in df.columns:
                continue
            y = df[target]
            params = self._best_params[target]
            for train_idx, val_idx in splits:
                model = self._build_model(params)
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                raw = model.predict(X.iloc[val_idx])
                oof[target][val_idx] = self._clip(raw, target)
        return pd.DataFrame(oof, index=df.index)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        df: pd.DataFrame,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Run per-type metrics on a held-out set and return the table.

        This is NOT re-running CV; call this on a separate test partition.
        """
        preds = self.predict(df)
        reporter = MetricsReporter(targets=self.targets)
        table = reporter.report(df, preds)
        if verbose:
            reporter.print_table(table)
        return table

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tune(
        self,
        df: pd.DataFrame,
        target: str,
        n_trials: int,
    ) -> tuple[float, dict]:
        """Run Optuna on spatial-CV RMSE for a single target."""
        skf = SpatialKFold(n_folds=self.n_folds, random_state=self.random_state)
        splits = list(skf.split(df))
        X = df[self.feature_cols]
        y = df[target]

        def objective(trial: optuna.Trial) -> float:
            params = {
                "learning_rate":    trial.suggest_float(
                    "learning_rate", *SEARCH_SPACE["learning_rate"], log=True
                ),
                "max_depth":        trial.suggest_int(
                    "max_depth", *SEARCH_SPACE["max_depth"]
                ),
                "subsample":        trial.suggest_float(
                    "subsample", *SEARCH_SPACE["subsample"]
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", *SEARCH_SPACE["colsample_bytree"]
                ),
                "reg_lambda":       trial.suggest_float(
                    "reg_lambda", *SEARCH_SPACE["reg_lambda"], log=True
                ),
                "n_estimators":     trial.suggest_int(
                    "n_estimators", *SEARCH_SPACE["n_estimators"], step=25
                ),
                "min_child_weight": trial.suggest_int(
                    "min_child_weight", *SEARCH_SPACE["min_child_weight"]
                ),
            }
            # No early stopping here: early_stopping_rounds watches the same
            # val fold this trial is scored against, so the number of trees
            # would be chosen using knowledge of the held-out fold -- a leak
            # into the CV score. n_estimators is already a tuned
            # hyperparameter (SEARCH_SPACE), so it alone controls tree count.
            model = self._build_model(params)
            fold_rmses = []
            for train_idx, val_idx in splits:
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                preds = model.predict(X.iloc[val_idx])
                fold_rmses.append(
                    float(np.sqrt(mean_squared_error(y.iloc[val_idx], preds)))
                )
            return float(np.mean(fold_rmses))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return study.best_value, study.best_params

    def _build_model(self, params: dict) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(
            tree_method="hist",
            random_state=self.random_state,
            n_jobs=2,
            verbosity=0,
            **params,
        )

    @staticmethod
    def _clip(values: np.ndarray, target: str) -> np.ndarray:
        """Clip predictions to physically plausible ranges."""
        bounds = {
            "chl_a":     (0.0,  500.0),   # µg/L
            "turbidity": (0.0, 2000.0),   # FNU
            "do":        (0.0,   20.0),   # mg/L
        }
        lo, hi = bounds.get(target, (0.0, 1e9))
        return np.clip(values, lo, hi)
