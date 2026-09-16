import pandas as pd

from milan_forecasting.forecasting.evaluation import repeat_params, results_tables, timing_summary
from milan_forecasting.forecasting.pipeline import MODELS


def test_repeat_params_uses_seeds_only_for_networks():
    assert repeat_params("lstm", {"lookback": 144}, [1, 2], 5) == [
        {"lookback": 144, "seed": 1},
        {"lookback": 144, "seed": 2},
    ]
    assert repeat_params("arima", {"p": 2}, [1, 2], 3) == [{"p": 2}] * 3
    assert repeat_params("naive", {}, [1, 2], 3) == [{}]


def _runs() -> pd.DataFrame:
    rows = []
    for model in MODELS:
        repeats = 2
        for run in range(repeats):
            error = 100.0 + run if model in ("lstm", "tcn") else 100.0
            rows.append({
                "square": 5161, "model": model, "run": run,
                "mae": error, "mape": 7.5, "rmse": 150.0, "mase": 0.3,
                "train_seconds": 10.0 + run, "predict_ms_per_step": 0.1,
            })
    return pd.DataFrame(rows)


def test_results_tables_report_seed_spread_only_for_networks():
    table = results_tables(_runs())

    assert "### Square 5161" in table
    assert "| LSTM | 100.50 ± 0.71 |" in table
    assert "| ARIMA-Fourier | 100.00 |" in table
    assert table.count("\n| ") == 1 + len(MODELS)


def test_timing_summary_reports_median_and_range():
    summary = timing_summary(_runs())

    assert summary["tcn"]["measurements"] == 2
    assert summary["tcn"]["train_seconds_median"] == 10.5
    assert summary["tcn"]["train_seconds_min"] == 10.0
    assert summary["tcn"]["train_seconds_max"] == 11.0
