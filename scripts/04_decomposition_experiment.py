"""Experiment: does robust STL fitting change the decomposition of the busiest square?

statsmodels fits MSTL without robustness weights by default, so outliers such as Christmas Day
influence the estimated trend and seasonal components — the decomposition bends towards the very
points it is meant to expose. This compares the default fit with a robust fit of log(1 + traffic).

Usage (from the repository root):
    python scripts/04_decomposition_experiment.py
"""
import json
import time

import pandas as pd

from milan_forecasting import config
from milan_forecasting.data.loader import load_internet_matrix, square_series
from milan_forecasting.analysis.statistics import area_totals, component_strengths, decompose, residual_anomalies

THRESHOLD = 3.5
CHRISTMAS = "2013-12-25"
HOLIDAY_WINDOW = ("2013-12-23", "2013-12-28")
# If anomalies concentrate in this festive stretch they mark a regime change; if they are spread
# evenly over all days and hours they mark systematic misfit instead.
HOLIDAY_REGIME = ("2013-12-23", "2014-01-01")


def summarise(components: pd.DataFrame, seconds: float) -> dict:
    residual = components["residual"]
    mask, bounds, z_scores = residual_anomalies(residual, THRESHOLD)
    flagged = z_scores[mask]
    by_day = flagged.groupby(flagged.index.strftime("%Y-%m-%d")).size().sort_values(ascending=False)
    christmas = residual[residual.index.strftime("%Y-%m-%d") == CHRISTMAS]
    std_by_hour = residual.groupby(residual.index.hour).std()

    def _share_between(start: str, end: str) -> float:
        start_ts = pd.Timestamp(start, tz=config.TIMEZONE)
        end_ts = pd.Timestamp(end, tz=config.TIMEZONE) + pd.Timedelta(days=1)
        return float(((flagged.index >= start_ts) & (flagged.index < end_ts)).mean())

    by_hour = flagged.groupby(flagged.index.hour).size().reindex(range(24), fill_value=0)
    return {
        "fit_seconds": round(seconds, 1),
        "residual_std": float(residual.std()),
        "residual_mad": float((residual - residual.median()).abs().median()),
        "residual_std_max_to_min_by_hour": float(std_by_hour.max() / std_by_hour.min()),
        "anomaly_points": int(mask.sum()),
        "anomaly_share": float(mask.mean()),
        "anomaly_bounds": [float(b) for b in bounds],
        "top_anomaly_days": {day: int(count) for day, count in by_day.head(10).items()},
        "days_with_anomalies": int(by_day.size),
        "share_in_holiday_regime": _share_between(*HOLIDAY_REGIME),
        "share_in_eval_week": _share_between(config.EVAL_START, config.EVAL_END),
        "busiest_anomaly_hour_count": int(by_hour.max()),
        "hours_with_no_anomalies": int((by_hour == 0).sum()),
        "christmas_mean_residual": float(christmas.mean()),
        "christmas_min_residual": float(christmas.min()),
        "component_strengths": component_strengths(components),
    }


def main() -> None:
    matrix, timestamps = load_internet_matrix()
    busiest = int(area_totals(matrix).idxmax())
    series = square_series(matrix, timestamps, busiest)

    fits = {}
    summaries = {}
    for name, robust in (("default", False), ("robust", True)):
        print(f"Fitting MSTL ({name}, robust={robust})...", flush=True)
        started = time.perf_counter()
        fits[name] = decompose(series, robust=robust)
        summaries[name] = summarise(fits[name], time.perf_counter() - started)

    holiday = slice(
        pd.Timestamp(HOLIDAY_WINDOW[0], tz=config.TIMEZONE),
        pd.Timestamp(HOLIDAY_WINDOW[1], tz=config.TIMEZONE),
    )
    trend_difference = (fits["robust"]["trend"] - fits["default"]["trend"]).abs()
    daily_difference = (fits["robust"]["seasonal_daily"] - fits["default"]["seasonal_daily"]).abs()
    christmas = fits["default"].index.strftime("%Y-%m-%d") == CHRISTMAS

    report = {
        "square_id": busiest,
        "note": "All values are on the log(1 + traffic) scale; a difference of 0.05 is about 5% of traffic.",
        "default": summaries["default"],
        "robust": summaries["robust"],
        "comparison": {
            "max_abs_trend_difference": float(trend_difference.max()),
            "max_abs_trend_difference_holiday_window": float(trend_difference.loc[holiday].max()),
            "mean_abs_daily_seasonal_difference": float(daily_difference.mean()),
            "mean_abs_daily_seasonal_difference_christmas": float(daily_difference[christmas].mean()),
            "residual_correlation": float(fits["robust"]["residual"].corr(fits["default"]["residual"])),
        },
    }
    config.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (config.ANALYSIS_DIR / "decomposition_experiment.json").write_text(json.dumps(report, indent=2))

    print(f"\nSquare {busiest}, log(1 + traffic) scale\n")
    rows = [
        ("fit seconds", "fit_seconds", "{:.1f}"),
        ("residual std", "residual_std", "{:.4f}"),
        ("residual MAD", "residual_mad", "{:.4f}"),
        ("residual std, busiest/quietest hour", "residual_std_max_to_min_by_hour", "{:.2f}"),
        ("anomaly points", "anomaly_points", "{:d}"),
        ("anomaly share", "anomaly_share", "{:.3%}"),
        ("days with anomalies (of 62)", "days_with_anomalies", "{:d}"),
        ("share in 23 Dec - 1 Jan", "share_in_holiday_regime", "{:.1%}"),
        ("share in evaluation week", "share_in_eval_week", "{:.1%}"),
        ("hours of day with no anomalies", "hours_with_no_anomalies", "{:d}"),
        ("Christmas mean residual", "christmas_mean_residual", "{:.3f}"),
        ("Christmas min residual", "christmas_min_residual", "{:.3f}"),
    ]
    print(f"{'metric':<38}{'default':>14}{'robust':>14}")
    for label, key, fmt in rows:
        print(f"{label:<38}{fmt.format(summaries['default'][key]):>14}{fmt.format(summaries['robust'][key]):>14}")
    for label, key in (("daily seasonality", "daily_seasonality"), ("weekly seasonality", "weekly_seasonality"), ("trend", "trend")):
        d = summaries["default"]["component_strengths"][key]
        r = summaries["robust"]["component_strengths"][key]
        print(f"{'strength: ' + label:<38}{d:>14.3f}{r:>14.3f}")
    print("\ntop anomaly days")
    print("  default:", summaries["default"]["top_anomaly_days"])
    print("  robust: ", summaries["robust"]["top_anomaly_days"])
    print("\ncomparison:", json.dumps(report["comparison"], indent=2))


if __name__ == "__main__":
    main()
