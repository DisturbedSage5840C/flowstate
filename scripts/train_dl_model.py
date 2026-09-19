"""DL model on the SYNTHETIC demo table (scripts/generate_synthetic_train.py). For real data use scripts/train_dl_real.py.

    python scripts/train_dl_model.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import pandas as pd

from src.models.dl_training import run_dl_experiment
from src.models.schema import SYNTH_FEATURE_COLS, SYNTH_TARGET_COLS

DATA_PATH = Path("data/processed/train.parquet")
REPORTS_DIR = Path("reports")


def main():
    df = pd.read_parquet(DATA_PATH)
    summary = run_dl_experiment(df, SYNTH_FEATURE_COLS, SYNTH_TARGET_COLS, REPORTS_DIR, epochs=60,
                                artifact_name="dl_artifact.pt")
    metrics = pd.DataFrame(summary["metrics"])
    dl = metrics[metrics["model"] == "dl_oof"]

    # legacy dashboard summary (reports/metrics.json), derived from the same table
    overall = {}
    for t in SYNTH_TARGET_COLS:
        row = dl[(dl["target"] == t) & (dl["water_body_type"] == "overall")].iloc[0]
        overall[f"{t}_r2"], overall[f"{t}_rmse"] = float(row["R2"]), float(row["RMSE"])
    per_type = {}
    for wtype in sorted(set(dl["water_body_type"]) - {"overall"}):
        per_type[wtype] = {
            f"{t}_r2": float(dl[(dl["target"] == t) & (dl["water_body_type"] == wtype)]["R2"].iloc[0])
            for t in SYNTH_TARGET_COLS
        }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps({
        "model": "CNN-BiLSTM-Attention (synthetic demo table, out-of-fold)",
        "overall": overall, "per_type": per_type, "n_rows": summary["rows"],
    }, indent=2))
    print(dl[dl["water_body_type"] == "overall"][["target", "n", "R2", "RMSE", "MAE"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
