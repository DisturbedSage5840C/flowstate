"""
src/models/tier_classifier.py
==============================
Classify WQI tier / CPCB best-use class from satellite-only features.

``wqi_tier`` and ``cpcb_class`` (src/wqi/wqi_engine.py) are derived from measured chemistry
(do, bod, ph, turbidity, total_coliform, conductivity, sar, boron, free_ammonia), never from
satellite bands, so predicting them from REAL_FEATURE_COLS alone is leakage-free. This is a
genuinely different question from the regression targets: "can the satellite tell you the
water is broadly Good vs Very Poor" is more decision-useful (and an easier target) than an
exact BOD/turbidity value, even though the same weak optical signal underlies both.

Honest, out-of-fold evaluation via the same SpatialKFold used for the regressors: a model
never sees the stations it is scored on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

from src.models.spatial_cv import SpatialKFold


class WQITierClassifier:
    """XGBoost multi-class classifier for a WQI tier / CPCB class label, with spatial OOF eval."""

    def __init__(self, feature_cols: list[str], target: str = "wqi_tier",
                 n_folds: int = 5, random_state: int = 42):
        self.feature_cols = feature_cols
        self.target = target
        self.n_folds = n_folds
        self.random_state = random_state
        self.encoder = LabelEncoder()
        self.model: XGBClassifier | None = None

    def _xgb(self, n_classes: int) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", num_class=n_classes,
            random_state=self.random_state, eval_metric="mlogloss",
        )

    def fit(self, df: pd.DataFrame) -> "WQITierClassifier":
        y = self.encoder.fit_transform(df[self.target])
        self.model = self._xgb(len(self.encoder.classes_))
        self.model.fit(df[self.feature_cols], y)
        return self

    def predict_oof(self, df: pd.DataFrame) -> np.ndarray:
        """Honest out-of-fold predicted labels (strings), refitting per spatial fold."""
        y_all = self.encoder.fit_transform(df[self.target])
        oof = np.empty(len(df), dtype=object)
        skf = SpatialKFold(n_folds=self.n_folds, random_state=self.random_state)
        for train_idx, val_idx in skf.split(df):
            clf = self._xgb(len(self.encoder.classes_))
            clf.fit(df[self.feature_cols].iloc[train_idx], y_all[train_idx])
            pred = clf.predict(df[self.feature_cols].iloc[val_idx])
            oof[val_idx] = self.encoder.inverse_transform(pred)
        return oof

    def evaluate(self, df: pd.DataFrame) -> dict:
        """OOF accuracy/F1/confusion matrix, plus a per-fold majority-class baseline."""
        y_true = df[self.target].to_numpy()
        y_pred = self.predict_oof(df)

        skf = SpatialKFold(n_folds=self.n_folds, random_state=self.random_state)
        baseline = np.empty(len(df), dtype=object)
        for train_idx, val_idx in skf.split(df):
            majority = df[self.target].iloc[train_idx].mode().iloc[0]
            baseline[val_idx] = majority

        labels = sorted(df[self.target].dropna().unique())
        return {
            "target": self.target,
            "n": int(len(df)),
            "classes": labels,
            "model": {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels)),
                "report": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
                "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            },
            "majority_class_baseline": {
                "accuracy": float(accuracy_score(y_true, baseline)),
                "macro_f1": float(f1_score(y_true, baseline, average="macro", labels=labels, zero_division=0)),
            },
        }
