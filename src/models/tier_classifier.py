"""
src/models/tier_classifier.py
==============================
Classify WQI tier / CPCB best-use class, and binary pollution-screening flags, from
satellite-only features.

``wqi_tier`` and ``cpcb_class`` (src/wqi/wqi_engine.py) are derived from measured chemistry
(do, bod, ph, turbidity, total_coliform, conductivity, sar, boron, free_ammonia), never from
satellite bands, so predicting them from REAL_FEATURE_COLS alone is leakage-free. This is a
genuinely different question from the regression targets: "can the satellite tell you the
water is broadly Good vs Very Poor" is more decision-useful (and an easier target) than an
exact BOD/turbidity value, even though the same weak optical signal underlies both.

``ScreeningClassifier`` targets a single binary threshold (e.g. BOD > 3 mg/L, the CPCB Class C
limit) instead of a 5-class tier. This is the project's headline deliverable: clean ablations
on train_real_large.parquet found real, measurable skill here (BOD>3 AUC 0.755, average
precision 0.601 vs a 0.288 base rate) where the row-level regressions have none -- a ranked
"inspect these stations first" shortlist is a decision an inspector can act on even though the
exact BOD concentration cannot be recovered from reflectance alone.

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


# ---------------------------------------------------------------------------
# Binary pollution screening -- the primary deliverable
# ---------------------------------------------------------------------------

# CPCB Class C BOD limit (3 mg/L) and a coarser "clearly polluted" threshold (6 mg/L); below CPCB
# Class D's DO minimum (unsafe even for fisheries/wildlife); worse than Class C overall.
SCREENING_TARGETS = ("bod_gt_3", "bod_gt_6", "do_lt_4", "cpcb_polluted")


def add_screening_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Binary pollution-screening labels, derived only from measured chemistry (never satellite bands).

    NaN wherever the underlying measurement (bod / do / cpcb_class) is missing for that visit --
    never imputed, so a row without a label is simply excluded by the caller's dropna, same as the
    regression targets.
    """
    from src.wqi.wqi_engine import CPCB_BELOW_E

    out = df.copy()
    bod, do = out.get("bod"), out.get("do")
    cls = out.get("cpcb_class")
    out["bod_gt_3"] = (bod > 3.0).astype(float).where(bod.notna()) if bod is not None else np.nan
    out["bod_gt_6"] = (bod > 6.0).astype(float).where(bod.notna()) if bod is not None else np.nan
    out["do_lt_4"] = (do < 4.0).astype(float).where(do.notna()) if do is not None else np.nan
    out["cpcb_polluted"] = (
        cls.isin(["D", "E", CPCB_BELOW_E]).astype(float).where(cls.notna()) if cls is not None else np.nan
    )
    return out


class ScreeningClassifier:
    """XGBoost binary classifier for a single pollution-screening flag, with spatial OOF eval.

    Reports AUC, average precision vs. the positive base rate (lift), precision@k for a ranked
    inspection shortlist, and a calibration check -- not accuracy, which class imbalance (these
    targets' base rates are typically well under 50 %) makes meaningless.
    """

    def __init__(self, feature_cols: list[str], target: str, n_folds: int = 5, random_state: int = 42):
        self.feature_cols = feature_cols
        self.target = target
        self.n_folds = n_folds
        self.random_state = random_state
        self.model: XGBClassifier | None = None

    def _xgb(self) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="binary:logistic",
            random_state=self.random_state, eval_metric="logloss",
        )

    def fit(self, df: pd.DataFrame) -> "ScreeningClassifier":
        self.model = self._xgb()
        self.model.fit(df[self.feature_cols], df[self.target].astype(int))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("No model trained yet.  Call .fit() first.")
        return self.model.predict_proba(df[self.feature_cols])[:, 1]

    def predict_proba_oof(self, df: pd.DataFrame) -> np.ndarray:
        """Honest out-of-fold predicted probability of the positive class, refitting per spatial fold."""
        y = df[self.target].astype(int).to_numpy()
        oof = np.full(len(df), np.nan)
        skf = SpatialKFold(n_folds=self.n_folds, random_state=self.random_state)
        for train_idx, val_idx in skf.split(df):
            clf = self._xgb()
            clf.fit(df[self.feature_cols].iloc[train_idx], y[train_idx])
            oof[val_idx] = clf.predict_proba(df[self.feature_cols].iloc[val_idx])[:, 1]
        return oof

    def evaluate(self, df: pd.DataFrame, k_fracs: tuple[float, ...] = (0.1, 0.2)) -> dict:
        """OOF AUC / average precision / precision@k / calibration for this screening target."""
        from src.models.metrics import classification_scores, precision_at_k

        y_true = df[self.target].astype(int).to_numpy()
        y_score = self.predict_proba_oof(df)
        scores = classification_scores(y_true, y_score)
        return {
            "target": self.target,
            "n": scores["n"],
            "base_rate": scores["base_rate"],
            "auc": scores["auc"],
            "average_precision": scores["ap"],
            "lift_vs_base_rate": scores["lift"],
            "precision_at_k": {f"{int(round(k * 100))}%": precision_at_k(y_true, y_score, k) for k in k_fracs},
            "calibration": self._calibration(y_true, y_score),
        }

    @staticmethod
    def _calibration(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> list[dict]:
        """Reliability check: mean predicted probability vs. actual positive rate, per decile of score."""
        mask = np.isfinite(y_true) & np.isfinite(y_score)
        y_true, y_score = y_true[mask], y_score[mask]
        if len(y_true) == 0:
            return []
        n_bins = min(n_bins, len(np.unique(y_score)))
        if n_bins < 2:
            return [{"mean_predicted": float(y_score.mean()), "mean_actual": float(y_true.mean()), "n": int(len(y_true))}]
        bins = pd.qcut(y_score, q=n_bins, duplicates="drop")
        g = pd.DataFrame({"bin": bins, "y": y_true, "p": y_score}).groupby("bin", observed=True)
        return [{"mean_predicted": float(sub["p"].mean()), "mean_actual": float(sub["y"].mean()), "n": int(len(sub))}
                for _, sub in g]
