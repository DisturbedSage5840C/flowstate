"""Data access for the real-data dashboard (pure pandas; no Streamlit imports so it is unit-testable)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.wqi.wqi_engine import compute_wqi_dataframe

ROOT = Path(__file__).resolve().parents[2]
TABLE_PATH = ROOT / "data" / "processed" / "train_real.parquet"
REAL_DIR = ROOT / "reports" / "real"
TARGETS = ("do", "bod", "turbidity")
PARAM_LABELS = {
    "wqi": "WQI (0-100, satellite-adapted)",
    "do": "Dissolved oxygen (mg/L)",
    "bod": "BOD (mg/L)",
    "turbidity": "Turbidity (NTU)",
}


@dataclass
class RealBundle:
    table: pd.DataFrame                               # one row per (station, date), measured labels + features
    metrics: pd.DataFrame | None = None               # reports/real/metrics_table.csv
    summary: dict = field(default_factory=dict)       # reports/real/dataset_summary.json
    validation: dict = field(default_factory=dict)    # reports/real/empirical_formula_validation.json
    comparison: pd.DataFrame | None = None            # reports/real/model_comparison.csv
    summary_metrics: dict = field(default_factory=dict)  # reports/real/metrics_summary.json
    screening: dict = field(default_factory=dict)     # reports/real/screening_metrics.json
    shortlist: pd.DataFrame | None = None             # reports/real/screening_shortlist.csv
    dl_summary: dict = field(default_factory=dict)    # reports/real/dl_summary.json
    has_predictions: bool = False                     # XGBoost out-of-fold predictions ({target}_pred)
    has_dl_predictions: bool = False                  # DL out-of-fold predictions ({target}_pred_dl)


def load_real(root: Path = ROOT) -> RealBundle | None:
    """Load the real-data bundle, or None when the real table has not been built."""
    table_path = root / "data" / "processed" / "train_real.parquet"
    if not table_path.exists():
        return None
    table = pd.read_parquet(table_path)
    if table.empty:
        return None
    table["date"] = pd.to_datetime(table["date"])
    real_dir = root / "reports" / "real"

    has_preds = False
    oof_path = real_dir / "oof_predictions.parquet"
    if oof_path.exists():
        oof = pd.read_parquet(oof_path)
        oof["date"] = pd.to_datetime(oof["date"])
        table = table.merge(oof, on=["site", "date"], how="left")
        has_preds = any(f"{t}_pred" in table.columns for t in TARGETS)

    def read_json(name: str) -> dict:
        path = real_dir / name
        return json.loads(path.read_text()) if path.exists() else {}

    has_dl = False
    dl_oof_path = real_dir / "dl_oof_predictions.parquet"
    if dl_oof_path.exists():
        dl_oof = pd.read_parquet(dl_oof_path)
        dl_oof["date"] = pd.to_datetime(dl_oof["date"])
        dl_oof = dl_oof.rename(columns={f"{t}_pred": f"{t}_pred_dl" for t in TARGETS})
        table = table.merge(dl_oof, on=["site", "date"], how="left")
        has_dl = any(f"{t}_pred_dl" in table.columns for t in TARGETS)

    frames = []
    for name in ("metrics_table.csv", "dl_metrics_table.csv"):
        path = real_dir / name
        if path.exists():
            frames.append(pd.read_csv(path))
    metrics = pd.concat(frames, ignore_index=True) if frames else None
    if metrics is not None:
        metrics = metrics.drop_duplicates(["model", "target", "water_body_type"], keep="first")
    comparison_path = real_dir / "model_comparison.csv"
    shortlist_path = real_dir / "screening_shortlist.csv"
    return RealBundle(
        table=table,
        metrics=metrics,
        summary=read_json("dataset_summary.json"),
        validation=read_json("empirical_formula_validation.json"),
        comparison=pd.read_csv(comparison_path) if comparison_path.exists() else None,
        summary_metrics=read_json("metrics_summary.json"),
        screening=read_json("screening_metrics.json"),
        shortlist=pd.read_csv(shortlist_path) if shortlist_path.exists() else None,
        dl_summary=read_json("dl_summary.json"),
        has_predictions=has_preds,
        has_dl_predictions=has_dl,
    )


def filter_visits(df: pd.DataFrame, states=None, types=None, start=None, end=None) -> pd.DataFrame:
    out = df
    if states:
        out = out[out["state"].isin(states)]
    if types:
        out = out[out["water_body_type"].isin(types)]
    if start is not None:
        out = out[out["date"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["date"] <= pd.Timestamp(end)]
    return out


PRED_SUFFIX = {"xgboost": "_pred", "model": "_pred", "dl": "_pred_dl"}


def with_display_columns(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Return frame whose ``do/bod/turbidity/wqi`` columns come from ``source``.

    ``source``: 'measured', 'xgboost' (alias 'model') or 'dl'. For the model sources the WQI is recomputed from
    that model's out-of-fold predictions for the parameters it predicts (DO, BOD, turbidity); nothing else is imputed.
    """
    out = df.copy()
    if source in PRED_SUFFIX:
        suffix = PRED_SUFFIX[source]
        for t in TARGETS:
            if f"{t}{suffix}" in out.columns:
                out[t] = out[f"{t}{suffix}"]
        for c in ("ph", "conductivity", "total_coliform"):
            if c in out.columns:
                out[c] = np.nan            # only model-predicted parameters feed the model WQI
        out = compute_wqi_dataframe(out.drop(columns=[c for c in out.columns if c.startswith("qi_")]))
    return out


def latest_per_station(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("date").groupby("site", as_index=False).last()


def tier_counts(df: pd.DataFrame) -> pd.Series:
    return df["wqi_tier"].dropna().value_counts()


def share_meeting_class(df: pd.DataFrame, classes=("A", "B", "C"), complete_only: bool = True) -> float:
    """Share of classified visits meeting the given CPCB best-use classes.

    ``complete_only`` (default) counts only visits where every criterion of the assigned class was measured. Class A-C
    need coliform, which is missing for most visits, so partial assignments would flatter the water quality.
    """
    known = df[df["cpcb_class"].notna()]
    if complete_only and "cpcb_class_complete" in known.columns:
        known = known[known["cpcb_class_complete"].fillna(False).astype(bool)]
    return float("nan") if known.empty else float(known["cpcb_class"].isin(classes).mean())


def complete_class_count(df: pd.DataFrame) -> int:
    if "cpcb_class_complete" not in df.columns:
        return 0
    return int(df.loc[df["cpcb_class"].notna(), "cpcb_class_complete"].fillna(False).astype(bool).sum())


def metric_pivot(metrics: pd.DataFrame, water_type: str = "overall") -> pd.DataFrame:
    """Rows = target, columns = model, values = R2 / RMSE for one water-body type."""
    m = metrics[metrics["water_body_type"] == water_type]
    wide = m.pivot(index="target", columns="model", values=["R2", "RMSE", "n"])
    wide.columns = [f"{a} - {b}" for a, b in wide.columns]
    return wide
