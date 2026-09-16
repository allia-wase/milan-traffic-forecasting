"""Run one tuning experiment on the validation week and append it to the experiment log.

The test week is never used here; it is reserved for the final evaluation.

Usage (from the repository root):
    python -m scripts.run_experiment --model lstm \
        --params '{"lookback": 144, "model": {"hidden_size": 64}}' \
        --note "Why this run: what the previous result suggested"
"""
import argparse
import csv
import json
from datetime import datetime

from src import config
from src.data_loader import load_internet_matrix, square_series
from src.eda import area_totals
from src.forecasting.pipeline import MODELS, run

LOG_FIELDS = [
    "run_id", "model", "square", "split", "mae", "rmse", "mape", "mase",
    "train_seconds", "predict_seconds", "best_epoch", "epochs_run", "parameters", "aic",
    "params", "note",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score one model configuration on the validation week.")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--params", default="{}", help="JSON object of model/training parameters")
    parser.add_argument("--square", type=int, help="square ID (default: the busiest square)")
    parser.add_argument("--note", required=True, help="reasoning behind this experiment")
    args = parser.parse_args()

    matrix, timestamps = load_internet_matrix()
    square = args.square or int(area_totals(matrix).idxmax())
    result = run(args.model, square_series(matrix, timestamps, square), json.loads(args.params), "val")

    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{args.model}"
    row = {
        "run_id": run_id,
        "model": args.model,
        "square": square,
        "split": result.split,
        **{k: round(v, 4) for k, v in result.metrics.items()},
        "train_seconds": round(result.train_seconds, 2),
        "predict_seconds": round(result.predict_seconds, 3),
        "best_epoch": result.details.get("best_epoch", ""),
        "epochs_run": result.details.get("epochs_run", ""),
        "parameters": result.details.get("parameters", ""),
        "aic": round(result.details["aic"], 1) if "aic" in result.details else "",
        "params": json.dumps(result.params, sort_keys=True),
        "note": args.note,
    }

    config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not config.EXPERIMENT_LOG.exists()
    with config.EXPERIMENT_LOG.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)

    runs_dir = config.EXPERIMENTS_DIR / "runs"
    runs_dir.mkdir(exist_ok=True)
    (runs_dir / f"{run_id}.json").write_text(json.dumps({**row, "details": result.details}, indent=2))

    print(json.dumps({k: row[k] for k in LOG_FIELDS if k not in ("params", "note")}, indent=2))
    if result.details.get("converged") is False:
        print("WARNING: ARIMA optimiser did not converge")


if __name__ == "__main__":
    main()
