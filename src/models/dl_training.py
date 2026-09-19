"""Shared DL experiment runner: out-of-fold metrics on the XGBoost spatial folds + a final saved artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.models.baselines import mean_baseline_oof, median_baseline_oof
from src.models.dl_model import AquaSenseTrainer, group_holdout, oof_predictions
from src.models.metrics import MetricsReporter


def run_dl_experiment(df: pd.DataFrame, feature_cols: Sequence[str], target_cols: Sequence[str], out_dir: str | Path,
                      n_folds: int = 5, seed: int = 42, epochs: int = 80, patience: int = 15,
                      window: int = 3, max_gap_days: float = 90.0, artifact_name: str = "dl_artifact.pt",
                      **trainer_kw) -> dict:
    """Train/evaluate the CNN-BiLSTM-attention model exactly like the XGBoost pipeline is evaluated.

    * same site-blocked spatial folds (``SpatialKFold`` with the same seed, same row order);
    * out-of-fold predictions only; metrics via the shared ``MetricsReporter``;
    * baselines ("predict the training mean / median") on the same folds;
    * a final model fitted on all rows is saved together with its scalers as one artifact.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.dropna(subset=list(feature_cols)).reset_index(drop=True)

    oof, fold_info = oof_predictions(df, feature_cols, target_cols, n_folds=n_folds, seed=seed, epochs=epochs,
                                     patience=patience, window=window, max_gap_days=max_gap_days, **trainer_kw)

    reporter = MetricsReporter(targets=list(target_cols))
    tables = []
    for name, preds in (("dl_oof", oof),
                        ("baseline_mean", mean_baseline_oof(df, list(target_cols), n_folds, seed)),
                        ("baseline_median", median_baseline_oof(df, list(target_cols), n_folds, seed))):
        tab = reporter.report(df, preds)
        tab.insert(0, "model", name)
        tables.append(tab)
    metrics = pd.concat(tables, ignore_index=True)
    metrics.to_csv(out_dir / "dl_metrics_table.csv", index=False)

    keep = [c for c in ("site", "date") if c in df.columns]
    pd.concat([df[keep], oof.add_suffix("_pred")], axis=1).to_parquet(out_dir / "dl_oof_predictions.parquet", index=False)

    train_part, stop_part = group_holdout(df, frac=0.15, seed=seed)
    trainer = AquaSenseTrainer(feature_cols, target_cols, window=window, max_gap_days=max_gap_days, seed=seed,
                               **trainer_kw)
    predictor = trainer.fit(train_part, stop_part, epochs=epochs, patience=patience)
    artifact = predictor.save(out_dir / artifact_name)

    # how often did a sample actually have history? (a temporal model is only exercised when it did)
    from src.models.dl_model import build_windows
    _, _, lengths = build_windows(df, window, max_gap_days)
    summary = {
        "model": "CNN-BiLSTM-masked-attention over per-station visit windows",
        "validation": f"{n_folds}-fold site-blocked spatial CV, out-of-fold predictions",
        "rows": int(len(df)), "stations": int(df["site"].nunique()) if "site" in df.columns else None,
        "window": window, "max_gap_days": max_gap_days,
        "share_of_samples_with_history": float((lengths > 1).mean()),
        "mean_window_length": float(lengths.mean()),
        "folds": fold_info, "artifact": str(artifact),
        "metrics": metrics.to_dict(orient="records"),
    }
    (out_dir / "dl_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
