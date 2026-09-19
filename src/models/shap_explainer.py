"""
src/models/shap_explainer.py
============================
SHAP-based interpretability for the Aqua-Sense XGBoost models.

Generates:
  1. Summary bar / beeswarm plots — overall feature importance
  2. Dependence plot — B5 (red-edge) vs Chl-a (key differentiator for pitch)
  3. Force plot for individual predictions (optional, useful in the dashboard)

Saved outputs
-------------
  reports/shap_plots/summary_<target>.png    ← beeswarm
  reports/shap_plots/bar_<target>.png        ← bar chart
  reports/shap_plots/dep_B5_chl_a.png        ← B5 dependence

Usage
-----
    from src.models.shap_explainer import SHAPExplainer
    exp = SHAPExplainer(output_dir="reports/shap_plots")
    exp.explain_all(xgb_pipeline, X_train, targets=["chl_a", "turbidity", "do"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for servers & CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb


class SHAPExplainer:
    """Generate SHAP explanations for the water quality XGBoost models.

    Parameters
    ----------
    output_dir : str | Path
        Directory where PNG plots will be saved.
    dpi : int
        Resolution for saved figures.
    """

    def __init__(
        self,
        output_dir: str | Path = "reports/shap_plots",
        dpi: int = 150,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        # Cache: target → shap_values array
        self._shap_cache: dict[str, np.ndarray] = {}
        self._explainer_cache: dict[str, shap.TreeExplainer] = {}

    # ------------------------------------------------------------------
    # High-level: explain all targets in one call
    # ------------------------------------------------------------------

    def explain_all(
        self,
        pipeline,            # WaterQualityXGB instance
        X: pd.DataFrame,
        targets: list[str] | None = None,
    ) -> None:
        """Generate and save summary + dependence plots for all targets.

        Parameters
        ----------
        pipeline : WaterQualityXGB
            Trained pipeline exposing ``_models`` dict.
        X : pd.DataFrame
            Feature matrix (same as training features).
        targets : list[str] | None
            Subset of targets to explain (default: all trained models).
        """
        targets = targets or list(pipeline._models.keys())
        for target in targets:
            if target not in pipeline._models:
                print(f"[SHAP] No model for '{target}'; skipping.")
                continue
            print(f"[SHAP] Explaining: {target}")
            model = pipeline._models[target]
            explainer, shap_vals = self._compute(model, X)
            self._shap_cache[target] = shap_vals
            self._explainer_cache[target] = explainer

            self.plot_beeswarm(shap_vals, X, target)
            self.plot_bar(shap_vals, X.columns.tolist(), target)

        # B5 → Chl-a dependence (the specific plot mentioned in the pitch)
        if "chl_a" in self._shap_cache and "B5" in X.columns:
            self.plot_dependence("B5", "chl_a", X)

    # ------------------------------------------------------------------
    # Individual plot methods (also callable standalone)
    # ------------------------------------------------------------------

    def plot_beeswarm(
        self,
        shap_values: np.ndarray,
        X: pd.DataFrame,
        target: str,
        max_display: int = 15,
    ) -> Path:
        """Save a SHAP beeswarm (summary) plot.

        Parameters
        ----------
        shap_values : np.ndarray of shape (n_samples, n_features)
        X : pd.DataFrame — feature matrix (for feature names + colours)
        target : str — used in the title and filename
        max_display : int — number of top features to display

        Returns
        -------
        Path to the saved PNG.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X,
            max_display=max_display,
            show=False,
            plot_type="dot",
        )
        plt.title(f"SHAP Beeswarm — {target}", fontsize=14, pad=12)
        plt.tight_layout()
        out_path = self.output_dir / f"summary_{target}.png"
        plt.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close("all")
        print(f"  Saved: {out_path}")
        return out_path

    def plot_bar(
        self,
        shap_values: np.ndarray,
        feature_names: list[str],
        target: str,
        max_display: int = 15,
    ) -> Path:
        """Save a SHAP bar chart (mean |SHAP|) — good for slide decks."""
        fig, ax = plt.subplots(figsize=(9, 5))
        shap.summary_plot(
            shap_values,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
            plot_type="bar",
        )
        plt.title(f"SHAP Feature Importance — {target}", fontsize=14, pad=12)
        plt.tight_layout()
        out_path = self.output_dir / f"bar_{target}.png"
        plt.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close("all")
        print(f"  Saved: {out_path}")
        return out_path

    def plot_dependence(
        self,
        feature: str,
        target: str,
        X: pd.DataFrame,
        interaction_feature: str = "auto",
    ) -> Path:
        """Save a SHAP dependence plot showing how `feature` drives `target`.

        Key use-case: B5 (red-edge) influence on Chl-a — highlighted in pitch.

        Parameters
        ----------
        feature : str
            Primary feature on the x-axis (e.g. "B5").
        target : str
            Target whose SHAP values to use (e.g. "chl_a").
        X : pd.DataFrame
            Feature matrix.
        interaction_feature : str | "auto"
            Feature to colour the scatter by (auto → SHAP picks it).
        """
        if target not in self._shap_cache:
            raise RuntimeError(
                f"No SHAP values for '{target}'.  Call explain_all() first."
            )
        shap_vals = self._shap_cache[target]
        feature_names = X.columns.tolist()
        if feature not in feature_names:
            raise ValueError(f"Feature '{feature}' not in X columns.")

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(
            feature,
            shap_vals,
            X,
            interaction_index=interaction_feature,
            ax=ax,
            show=False,
        )
        ax.set_title(
            f"SHAP Dependence: {feature} → {target}", fontsize=13, pad=10
        )
        plt.tight_layout()
        safe_feat = feature.replace("/", "_")
        out_path = self.output_dir / f"dep_{safe_feat}_{target}.png"
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close("all")
        print(f"  Saved: {out_path}")
        return out_path

    def waterfall(
        self,
        model: xgb.XGBRegressor,
        X: pd.DataFrame,
        idx: int,
        target: str,
    ) -> Path:
        """Save a SHAP waterfall plot for a single prediction (idx-th sample).

        Useful in the Streamlit dashboard for explaining individual predictions.
        """
        explainer = shap.TreeExplainer(model)
        shap_explanation = explainer(X.iloc[[idx]])
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_explanation[0], show=False)
        plt.title(f"SHAP Waterfall — {target}, sample {idx}", fontsize=12)
        plt.tight_layout()
        out_path = self.output_dir / f"waterfall_{target}_{idx}.png"
        plt.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close("all")
        return out_path

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute(
        self,
        model: xgb.XGBRegressor,
        X: pd.DataFrame,
    ) -> tuple[shap.TreeExplainer, np.ndarray]:
        """Build TreeExplainer and compute SHAP values."""
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        return explainer, shap_values
