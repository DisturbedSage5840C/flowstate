"""
scripts/train_models.py
=======================
Full training, evaluation, and SHAP pipeline runner for Aqua-Sense (Navya / P3).

Usage:
    python scripts/train_models.py [--trials 20] [--folds 5] [--data data/processed/train.parquet]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import (
    WaterQualityXGB,
    MetricsReporter,
    SHAPExplainer,
    SpatialKFold,
)
from src.models.xgboost_pipeline import FEATURE_COLS, TARGET_COLS


def parse_args():
    parser = argparse.ArgumentParser(description="Train Aqua-Sense XGBoost pipeline with Spatial CV.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/train.parquet",
        help="Path to training parquet file.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=25,
        help="Number of Optuna trials per target parameter.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of spatial-CV folds.",
    )
    parser.add_argument(
        "--output-models",
        type=str,
        default="reports/models",
        help="Directory to save model artifacts.",
    )
    parser.add_argument(
        "--output-shap",
        type=str,
        default="reports/shap_plots",
        help="Directory to save SHAP plots.",
    )
    parser.add_argument(
        "--output-reports",
        type=str,
        default="reports",
        help="Directory to save metrics reports.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    
    if not data_path.exists():
        print(f"[ERROR] Training data not found at: {data_path}")
        print("Please generate it first via: python scripts/generate_synthetic_data.py")
        sys.exit(1)

    print(f"\n=======================================================")
    print(f"   Aqua-Sense ML Training Pipeline (Navya / P3)")
    print(f"=======================================================")
    print(f"  Data:         {data_path}")
    print(f"  Optuna Trials:{args.trials}")
    print(f"  Spatial Folds:{args.folds}")
    print(f"  Random Seed:  {args.seed}")
    print(f"=======================================================\n")

    df = pd.read_parquet(data_path)
    print(f"[DATA] Loaded {len(df)} samples across {df['site'].nunique()} sites.")
    print(f"       Sites: {', '.join(sorted(df['site'].unique()))}\n")

    # 1. Inspect Spatial CV folds
    skf = SpatialKFold(n_folds=args.folds, random_state=args.seed)
    fold_summary = skf.fold_summary(df)
    print("[SPATIAL-CV] Geographic Fold Allocation:")
    print(fold_summary.to_string(index=False))
    print()

    # 2. Train and Tune XGBoost with Spatial-CV objective
    print(f"[TRAIN] Launching Optuna HPO ({args.trials} trials/target)...")
    xgb_pipe = WaterQualityXGB(
        n_folds=args.folds,
        random_state=args.seed,
    )
    cv_scores = xgb_pipe.train(df, n_trials=args.trials, verbose=True)

    # 3. Save model artifacts
    model_dir = Path(args.output_models)
    model_dir.mkdir(parents=True, exist_ok=True)
    xgb_pipe.save(model_dir)

    # 4. Generate Predictions and Metrics Report
    print("\n[EVAL] Computing Spatial Metrics Report (per water body type & overall)...")
    preds = xgb_pipe.predict(df)
    reporter = MetricsReporter(targets=TARGET_COLS)
    metrics_table = reporter.report(df, preds)
    
    print("\n" + "="*50)
    print("           SPATIAL CROSS-VALIDATION METRICS")
    print("="*50)
    reporter.print_table(metrics_table)
    print("="*50 + "\n")

    reports_dir = Path(args.output_reports)
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = reports_dir / "metrics_table.csv"
    metrics_table.to_csv(metrics_csv, index=False)
    print(f"[OK] Saved metrics table to: {metrics_csv}")

    # Save summary JSON
    summary_dict = {
        "spatial_cv_best_rmse": cv_scores,
        "n_samples": len(df),
        "n_sites": int(df["site"].nunique()),
        "metrics": metrics_table.to_dict(orient="records"),
    }
    with open(reports_dir / "metrics_summary.json", "w") as f:
        json.dump(summary_dict, f, indent=2)

    # 5. Generate SHAP interpretability plots
    print(f"\n[SHAP] Generating SHAP explainability plots in: {args.output_shap}...")
    shap_dir = Path(args.output_shap)
    shap_dir.mkdir(parents=True, exist_ok=True)
    explainer = SHAPExplainer(output_dir=shap_dir, dpi=150)
    X = df[FEATURE_COLS]
    explainer.explain_all(xgb_pipe, X, targets=TARGET_COLS)

    print("\n[OK] Pipeline completed successfully!")
    print(f"  Models:  {model_dir}")
    print(f"  Metrics: {reports_dir}")
    print(f"  SHAP:    {shap_dir}\n")


if __name__ == "__main__":
    main()
