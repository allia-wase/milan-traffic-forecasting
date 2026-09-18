"""Tuning grids, repeated-run planning, results tables and timing summaries."""
import itertools

import pandas as pd

from milan_forecasting.forecasting.figures import MODEL_LABELS
from milan_forecasting.forecasting.pipeline import BASELINES, MODELS

NEURAL = ("lstm", "tcn")
METRIC_ORDER = ("mae", "mape", "rmse", "mase")

TIMING_METHOD = (
    "Wall-clock time from time.perf_counter. Training time covers fitting on the training split "
    "(for the networks: all epochs including early stopping on the validation split). Prediction "
    "time covers producing all 1,008 one-step-ahead forecasts of the test week. Networks: one "
    "measurement per seed per square; ARIMA: repeated runs per square; the two naive baselines: one "
    "run per square, since there is nothing to fit. Reported as the "
    "median with min-max over all measurements for the model across the three squares. CPU only."
)


def grid_combinations(base: dict, grid: dict) -> list[dict]:
    """Every combination of the grid values, each merged over the base parameters."""
    keys = list(grid)
    return [{**base, **dict(zip(keys, values))} for values in itertools.product(*(grid[k] for k in keys))]


def repeat_params(model: str, params: dict, seeds: list[int], timing_repeats: int) -> list[dict]:
    """One run per seed for the networks; deterministic models are repeated only to time them."""
    if model in NEURAL:
        return [{**params, "seed": seed} for seed in seeds]
    return [params] * (1 if model in BASELINES else timing_repeats)


def format_cell(values: pd.Series, digits: int) -> str:
    if len(values) > 1:
        return f"{values.mean():.{digits}f} ± {values.std(ddof=1):.{digits}f}"
    return f"{values.iloc[0]:.{digits}f}"


def results_tables(runs: pd.DataFrame) -> str:
    """One Markdown table per square with MAE, MAPE, RMSE and MASE for every model."""
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
    note = "Networks: mean ± standard deviation over seeds. ARIMA and the baselines are deterministic."
    return "\n\n".join(sections) + f"\n\n{note}\n"


def timing_summary(runs: pd.DataFrame) -> dict:
    """Median and range of training and per-forecast prediction time for each model."""
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
