"""Final evaluation on the untouched test week (16-22 Dec) for the three busiest squares.

Model settings come from configs/final_models.json, chosen on the validation week by the tuning
experiments. Neural networks are trained once per seed; ARIMA and the baseline are deterministic,
so their repeats only serve to measure run time.

Usage (from the repository root):
    python -m scripts.evaluate --seeds 42 7 123 --timing-repeats 3
"""
import argparse
import json

import pandas as pd

from src import config
from src.data_loader import load_internet_matrix, square_series
from src.eda import area_totals
from src.forecasting.pipeline import MODELS, run
from src.forecasting.plots import MODEL_LABELS, plot_forecast
from src.plotting import apply_style
from src.system_info import system_info

FINAL_CONFIGS = config.ROOT / "configs" / "final_models.json"
FINAL_DIR = config.OUTPUT_DIR / "final"
NEURAL = ("lstm", "tcn")
METRIC_ORDER = ("mae", "mape", "rmse", "mase")

TIMING_METHOD = (
    "Wall-clock time from time.perf_counter. Training time covers fitting on the training split "
    "(for the networks: all epochs including early stopping on the validation split). Prediction "
    "time covers producing all 1,008 one-step-ahead forecasts of the test week. Networks: one "
    "measurement per seed per square; ARIMA and baseline: repeated runs per square. Reported as the "
    "median with min-max over all measurements for the model across the three squares. CPU only."
)


def repeat_params(model: str, params: dict, seeds: list[int], timing_repeats: int) -> list[dict]:
    if model in NEURAL:
        return [{**params, "seed": seed} for seed in seeds]
    return [params] * (1 if model == "naive" else timing_repeats)


def format_cell(values: pd.Series, digits: int) -> str:
    if len(values) > 1:
        return f"{values.mean():.{digits}f} ± {values.std(ddof=1):.{digits}f}"
    return f"{values.iloc[0]:.{digits}f}"


def results_tables(runs: pd.DataFrame) -> str:
    sections = []
    for square, group in runs.groupby("square", sort=False):
        lines = [
            f"### Square {square}",
            "",
            "| Model | MAE | MAPE (%) | RMSE | MASE |",
            "|---|---|---|---|---|",
        ]
        for model in MODELS:
            rows = group[group["model"] == model]
            # Deterministic repeats have identical errors; only seeds contribute spread.
            rows = rows if model in NEURAL else rows.head(1)
            cells = [format_cell(rows[m], 3 if m == "mase" else 2) for m in METRIC_ORDER]
            lines.append(f"| {MODEL_LABELS[model]} | " + " | ".join(cells) + " |")
        sections.append("\n".join(lines))
    note = "Networks: mean ± standard deviation over seeds. ARIMA and the baseline are deterministic."
    return "\n\n".join(sections) + f"\n\n{note}\n"


def timing_summary(runs: pd.DataFrame) -> dict:
    summary = {}
    for model, group in runs.groupby("model", sort=False):
        summary[model] = {
            "measurements": len(group),
            "train_seconds_median": round(group["train_seconds"].median(), 2),
            "train_seconds_min": round(group["train_seconds"].min(), 2),
            "train_seconds_max": round(group["train_seconds"].max(), 2),
            "predict_ms_per_step_median": round(group["predict_ms_per_step"].median(), 4),
            "predict_ms_per_step_min": round(group["predict_ms_per_step"].min(), 4),
            "predict_ms_per_step_max": round(group["predict_ms_per_step"].max(), 4),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the final models on the test week.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    parser.add_argument("--timing-repeats", type=int, default=3)
    args = parser.parse_args()

    final_configs = json.loads(FINAL_CONFIGS.read_text())
    apply_style()
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
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
                                  result.metrics, note, config.FIGURES_DIR / f"forecast_{square}_{model}.png")
        predictions.to_csv(FINAL_DIR / f"predictions_{square}.csv", index_label="timestamp")

    runs = pd.DataFrame(rows)
    runs.to_csv(FINAL_DIR / "metrics_all_runs.csv", index=False)
    (FINAL_DIR / "results_tables.md").write_text(results_tables(runs), encoding="utf-8")
    (FINAL_DIR / "timing.json").write_text(json.dumps({
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
