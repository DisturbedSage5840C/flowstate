"""Compare XGBoost and the DL model (both out-of-fold on the same spatial folds) and record the production choice.

    python -m scripts.compare_models

Reads reports/real/metrics_table.csv (XGBoost + baselines) and reports/real/dl_metrics_table.csv, writes
reports/real/model_comparison.csv and .json.
"""
import json

import pandas as pd

from src.data import nwdp
from src.models.comparison import compare_models

REAL = nwdp.ROOT / "reports" / "real"


def main():
    xgb = pd.read_csv(REAL / "metrics_table.csv")
    dl_path = REAL / "dl_metrics_table.csv"
    dl = pd.read_csv(dl_path) if dl_path.exists() else None
    out = compare_models(xgb, dl)
    out.to_csv(REAL / "model_comparison.csv", index=False)
    (REAL / "model_comparison.json").write_text(json.dumps(out.to_dict(orient="records"), indent=2))
    cols = ["target", "n", "xgboost_oof_R2", "xgboost_oof_R2_log", "dl_oof_R2", "baseline_mean_R2", "production_model",
            "beats_baseline", "skill"]
    print(out[[c for c in cols if c in out.columns]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
