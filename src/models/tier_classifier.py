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
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
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


# ---------------------------------------------------------------------------
# Binary screening: "is this water body likely to breach a regulatory limit?"
# ---------------------------------------------------------------------------
# Concentration regression from reflectance is near-zero out-of-fold for every target (DO R2 ~ 0,
# turbidity ~ 0; see reports/real/metrics_table.csv), because 85-100% of each target's variance is
# *between* stations and site-blocked CV holds whole stations out. Ranking stations by the probability
# of breaching a limit is a different, easier and more decision-useful question, and it does work:
# BOD > 3 mg/L reaches AUC ~0.75 with ~2x the base-rate precision. Labels come from measured chemistry,
# never from bands, so this stays leakage-free.

SCREENING_TARGETS = {
    "bod_gt_3":      {"column": "bod", "op": ">", "threshold": 3.0,
                      "label": "BOD above 3 mg/L (CPCB Class B/C limit)"},
    "bod_gt_6":      {"column": "bod", "op": ">", "threshold": 6.0,
                      "label": "BOD above 6 mg/L (heavily polluted)"},
    "do_lt_4":       {"column": "do", "op": "<", "threshold": 4.0,
                      "label": "DO below 4 mg/L (fails CPCB Class D)"},
    "turbidity_gt_10": {"column": "turbidity", "op": ">", "threshold": 10.0,
                        "label": "Turbidity above 10 NTU"},
    "cpcb_polluted": {"column": "cpcb_class", "op": "in", "threshold": ("D", "E", "Below E"),
                      "label": "CPCB class D/E/Below E (unfit for bathing or drinking-water supply)"},
}


def screening_label(df: pd.DataFrame, name: str) -> pd.Series:
    """Binary label for a screening target; NaN where the underlying measurement is missing."""
    spec = SCREENING_TARGETS[name]
    col = df[spec["column"]]
    if spec["op"] == "in":
        y = col.isin(spec["threshold"])
    elif spec["op"] == ">":
        y = col > spec["threshold"]
    else:
        y = col < spec["threshold"]
    return y.astype(float).where(col.notna())


class ScreeningClassifier:
    """Out-of-fold probability that a visit breaches a limit, scored the way a screening tool is used.

    Accuracy is not reported: with a 7-29% base rate a model that always says "clean" looks excellent and
    is useless. AUC (ranking quality), average precision against the base rate (lift), and precision@k
    (how many of the top-k flagged sites are genuinely bad) are what matter for an inspection shortlist.
    """

    def __init__(self, feature_cols: list[str], target: str, n_folds: int = 5, random_state: int = 42):
        if target not in SCREENING_TARGETS:
            raise KeyError(f"unknown screening target {target!r}; have {sorted(SCREENING_TARGETS)}")
        self.feature_cols = feature_cols
        self.target = target
        self.spec = SCREENING_TARGETS[target]
        self.n_folds = n_folds
        self.random_state = random_state
        self.model: XGBClassifier | None = None

    def _xgb(self) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=2.0, random_state=self.random_state, eval_metric="logloss", n_jobs=4, verbosity=0,
        )

    def frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rows usable for this target: label present and every feature present."""
        out = df.copy()
        out["_y"] = screening_label(out, self.target)
        return out.dropna(subset=["_y"] + list(self.feature_cols)).reset_index(drop=True)

    def predict_oof(self, df: pd.DataFrame) -> np.ndarray:
        """Out-of-fold breach probability; a model never scores a station it trained on."""
        y = df["_y"].to_numpy().astype(int)
        p = np.full(len(df), np.nan)
        for train_idx, val_idx in SpatialKFold(n_folds=self.n_folds, random_state=self.random_state).split(df):
            if len(np.unique(y[train_idx])) < 2:
                continue
            clf = self._xgb()
            clf.fit(df[self.feature_cols].iloc[train_idx], y[train_idx])
            p[val_idx] = clf.predict_proba(df[self.feature_cols].iloc[val_idx])[:, 1]
        return p

    def fit(self, df: pd.DataFrame) -> "ScreeningClassifier":
        frame = self.frame(df)
        self.model = self._xgb().fit(frame[self.feature_cols], frame["_y"].astype(int))
        return self

    def evaluate(self, df: pd.DataFrame) -> dict:
        frame = self.frame(df)
        y = frame["_y"].to_numpy().astype(int)
        p = self.predict_oof(frame)
        ok = np.isfinite(p)
        y, p = y[ok], p[ok]
        base = float(y.mean())
        if len(np.unique(y)) < 2:
            return {"target": self.target, "n": int(len(y)), "note": "only one class present"}
        ap = float(average_precision_score(y, p))
        order = np.argsort(-p)
        prec_at_k = {f"top_{int(k * 100)}pct": float(y[order[:max(1, int(k * len(y)))]].mean())
                     for k in (0.05, 0.10, 0.20)}
        return {
            "target": self.target,
            "description": self.spec["label"],
            "n": int(len(y)),
            "positives": int(y.sum()),
            "base_rate": round(base, 4),
            "roc_auc": round(float(roc_auc_score(y, p)), 4),
            "average_precision": round(ap, 4),
            "lift_over_base_rate": round(ap / base, 3) if base else None,
            "precision_at_k": {k: round(v, 4) for k, v in prec_at_k.items()},
            "lift_at_k": {k: round(v / base, 3) for k, v in prec_at_k.items()} if base else None,
            "brier_score": round(float(np.mean((p - y) ** 2)), 4),
            "brier_score_base_rate_guess": round(float(np.mean((base - y) ** 2)), 4),
            "n_folds": self.n_folds,
            "features": list(self.feature_cols),
        }
