"""Final evaluation on the untouched test week (16-22 Dec) for the three busiest squares.

Model settings come from configs/final_models.json, chosen on the validation week by the tuning
experiments. Neural networks are trained once per seed; ARIMA and the baseline are deterministic,
so their repeats only serve to measure run time.

Usage (from the repository root):
    python scripts/06_evaluate_models.py --seeds 42 7 123 --timing-repeats 3
"""
import argparse
import json

import pandas as pd

from milan_forecasting import config
from milan_forecasting.data.loader import load_internet_matrix, square_series
from milan_forecasting.analysis.statistics import area_totals
from milan_forecasting.forecasting.evaluation import (
    NEURAL,
    TIMING_METHOD,
    repeat_params,
    results_tables,
    timing_summary,
)
from milan_forecasting.forecasting.figures import plot_forecast
from milan_forecasting.forecasting.pipeline import MODELS, run
from milan_forecasting.plotting import apply_style
from milan_forecasting.system_info import system_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the final models on the test week.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    parser.add_argument("--timing-repeats", type=int, default=3)
    args = parser.parse_args()

    final_configs = json.loads(config.FINAL_MODELS_CONFIG.read_text())
    apply_style()
    config.FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    matrix, timestamps = load_internet_matrix()
    squares = area_totals(matrix).nlargest(config.N_EVAL_SQUARES).index.tolist()

    rows = []
    for square in squares:
        series = square_series(matrix, timestamps, square)
        predictions = None
        for model in MODELS:
            repeats = repeat_params(model, final_configs.get(model, {}), args.seeds, args.timing_repeats)
            for index, params in enumerate(repeats):
                print(f"square {square} · {model} · run {index + 1}/{len(repeats)}", flush=True)
                result = run(model, series, params, "test")
                rows.append({
                    "square": square,
                    "model": model,
                    "run": index,
                    "seed": params.get("seed", ""),
                    **result.metrics,
                    "train_seconds": result.train_seconds,
                    "predict_seconds": result.predict_seconds,
                    "predict_ms_per_step": 1000 * result.predict_seconds / len(result.actual),
                    "best_epoch": result.details.get("best_epoch", ""),
                })
                if index > 0:
                    continue
                if predictions is None:
                    predictions = pd.DataFrame({"actual": result.actual}, index=result.timestamps)
                predictions[model] = result.predicted
                if model != "naive":
                    note = f"  ·  seed {params['seed']}" if model in NEURAL else ""
                    plot_forecast(result.timestamps, result.actual, result.predicted, model, square,
                                  result.metrics, note, config.FORECASTS_DIR / f"square_{square}_{model}.png")
        predictions.to_csv(config.FORECASTS_DIR / f"predictions_{square}.csv", index_label="timestamp")

    runs = pd.DataFrame(rows)
    runs.to_csv(config.EVALUATION_DIR / "metrics_all_runs.csv", index=False)
    (config.EVALUATION_DIR / "results_tables.md").write_text(results_tables(runs), encoding="utf-8")
    (config.EVALUATION_DIR / "timing.json").write_text(json.dumps({
        "hardware": system_info(),
        "method": TIMING_METHOD,
        "seeds": args.seeds,
        "timing_repeats": args.timing_repeats,
        "configs": final_configs,
        "by_model": timing_summary(runs),
    }, indent=2), encoding="utf-8")
    print(results_tables(runs))
    print(json.dumps(timing_summary(runs), indent=2))


if __name__ == "__main__":
    main()
