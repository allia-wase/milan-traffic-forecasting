"""Failure analysis of the test-week forecasts, using the predictions saved by
scripts/06_evaluate_models.py.

Usage (from the repository root, after `python scripts/06_evaluate_models.py`):
    python scripts/07_failure_analysis.py
"""
import json

import pandas as pd

from milan_forecasting import config
from milan_forecasting.forecasting.diagnostics import (
    error_by_day,
    error_by_move_size,
    largest_misses,
    load_predictions,
    reaction_summary,
    weekend_to_weekday_ratio,
)
from milan_forecasting.forecasting.figures import (
    TRAINED_MODELS,
    plot_error_by_day,
    plot_error_by_move,
    plot_failure_window,
)
from milan_forecasting.plotting import apply_style

# The busiest square's two worst test days by error (see mae_by_day in the output): the last
# weekend before Christmas, when its traffic peaked about 50% above weekday levels.
FAILURE_WINDOW = ("2013-12-21 00:00", "2013-12-22 23:50")
DETAIL_HALF_WIDTH = pd.Timedelta(hours=2)


def main() -> None:
    apply_style()
    runs = pd.read_csv(config.EVALUATION_DIR / "metrics_all_runs.csv")
    squares = list(dict.fromkeys(runs["square"]))  # evaluation order is busiest first
    predictions = {square: load_predictions(config.FORECASTS_DIR / f"predictions_{square}.csv") for square in squares}

    daily = {square: error_by_day(frame) for square, frame in predictions.items()}
    naive_peak = max(frame["naive"].max() for frame in daily.values())
    plot_error_by_day(
        daily,
        f"Networks: seed 42. Seasonal naive omitted for scale (its daily error reaches {naive_peak:,.0f}; "
        "see summary.json).",
        config.FAILURE_ANALYSIS_DIR / "error_by_day.png",
    )

    curves = {square: {m: error_by_move_size(frame, m) for m in TRAINED_MODELS} for square, frame in predictions.items()}
    plot_error_by_move(curves, config.FAILURE_ANALYSIS_DIR / "error_vs_change_size.png")

    busiest = squares[0]
    start, end = (pd.Timestamp(t, tz=config.TIMEZONE) for t in FAILURE_WINDOW)
    worst_in_window = largest_misses(predictions[busiest].loc[start:end], "arima", 1).index[0]
    plot_failure_window(
        predictions[busiest],
        window=(start, end),
        detail=(worst_in_window - DETAIL_HALF_WIDTH, worst_in_window + DETAIL_HALF_WIDTH),
        miss_time=worst_in_window,
        title=f"Poorest period: square {busiest} on the last weekend before Christmas (21–22 Dec 2013)",
        path=config.FAILURE_ANALYSIS_DIR / "poorest_period.png",
    )

    report = {"failure_window": {"square": busiest, "start": FAILURE_WINDOW[0], "end": FAILURE_WINDOW[1],
                                 "largest_arima_miss": worst_in_window.isoformat()}}
    for square, frame in predictions.items():
        misses = largest_misses(frame, "arima", 10)
        report[str(square)] = {
            "mae_by_day": daily[square].round(2).to_dict(),
            "weekend_to_weekday_mae": weekend_to_weekday_ratio(daily[square]),
            "reaction": {model: reaction_summary(frame, model) for model in TRAINED_MODELS},
            "largest_arima_misses": [
                {"time": ts.strftime("%a %d %H:%M"), **{k: round(float(v), 1) for k, v in row.items()}}
                for ts, row in misses[["actual", "previous_actual", "arima", "lstm", "tcn", "abs_error"]].iterrows()
            ],
        }
    (config.FAILURE_ANALYSIS_DIR / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    for square in squares:
        reaction = report[str(square)]["reaction"]
        print(f"square {square}: " + "  ".join(
            f"{m} under-reacting in {r['under_reaction_share_largest']:.0%} of largest errors "
            f"(corr {r['corr_abs_error_abs_move']:.2f})" for m, r in reaction.items()))


if __name__ == "__main__":
    main()
