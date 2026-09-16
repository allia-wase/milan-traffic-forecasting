"""Run tuning experiments on the validation week and append them to the experiment log.

The test week is never used here; it is reserved for the final evaluation. A single run tests
one change chosen from the previous result. --grid runs every combination of the listed values
on top of --params, logging each combination as its own run.

Usage (from the repository root):
    python scripts/05_tuning_experiment.py --model lstm \
        --params '{"lookback": 144, "model": {"hidden_size": 64}}' \
        --note "Why this run: what the previous result suggested"
    python scripts/05_tuning_experiment.py --model arima \
        --params '{"d": 1, "daily_harmonics": 16, "weekly_harmonics": 8}' \
        --grid '{"p": [0, 1, 2], "q": [0, 1, 2]}' --note "Why this grid"
"""
import argparse
import csv
import json
from datetime import datetime

from milan_forecasting import config
from milan_forecasting.data.loader import load_internet_matrix, square_series
from milan_forecasting.analysis.statistics import area_totals
from milan_forecasting.forecasting.evaluation import grid_combinations
from milan_forecasting.forecasting.pipeline import MODELS, run

LOG_FIELDS = [
    "run_id", "model", "square", "split", "mae", "rmse", "mape", "mase",
    "train_seconds", "predict_seconds", "best_epoch", "epochs_run", "parameters", "aic", "converged",
    "params", "note",
]


def run_and_log(model: str, series, square: int, params: dict, note: str) -> dict:
    result = run(model, series, params, "val")
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{model}"
    row = {
        "run_id": run_id,
        "model": model,
        "square": square,
        "split": result.split,
        **{k: round(v, 4) for k, v in result.metrics.items()},
        "train_seconds": round(result.train_seconds, 2),
        "predict_seconds": round(result.predict_seconds, 3),
        "best_epoch": result.details.get("best_epoch", ""),
        "epochs_run": result.details.get("epochs_run", ""),
        "parameters": result.details.get("parameters", ""),
        "aic": round(result.details["aic"], 1) if "aic" in result.details else "",
        "converged": result.details.get("converged", ""),
        "params": json.dumps(result.params, sort_keys=True),
        "note": note,
    }

    config.TUNING_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not config.EXPERIMENT_LOG.exists()
    with config.EXPERIMENT_LOG.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)

    runs_dir = config.TUNING_DIR / "runs"
    runs_dir.mkdir(exist_ok=True)
    (runs_dir / f"{run_id}.json").write_text(json.dumps({**row, "details": result.details}, indent=2))

    print(json.dumps({k: row[k] for k in LOG_FIELDS if k not in ("params", "note")}, indent=2), flush=True)
    if result.details.get("converged") is False:
        print("WARNING: ARIMA optimiser did not converge", flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Score model configurations on the validation week.")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--params", default="{}", help="JSON object of model/training parameters")
    parser.add_argument("--grid", help='JSON object mapping parameter names to lists of values, e.g. {"p": [0, 1]}')
    parser.add_argument("--square", type=int, help="square ID (default: the busiest square)")
    parser.add_argument("--note", required=True, help="reasoning behind this experiment")
    args = parser.parse_args()

    matrix, timestamps = load_internet_matrix()
    square = args.square or int(area_totals(matrix).idxmax())
    series = square_series(matrix, timestamps, square)
    base = json.loads(args.params)

    if not args.grid:
        run_and_log(args.model, series, square, base, args.note)
        return
    grid = json.loads(args.grid)
    for params in grid_combinations(base, grid):
        label = ", ".join(f"{k}={params[k]}" for k in grid)
        run_and_log(args.model, series, square, params, f"Grid search ({label}): {args.note}")


if __name__ == "__main__":
    main()
