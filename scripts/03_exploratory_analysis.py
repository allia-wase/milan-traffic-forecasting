"""Exploratory analysis: traffic distribution, focus-area time series, and diagnostics of the busiest square.

Requires the processed matrix from `python scripts/01_build_dataset.py`.

Usage (from the repository root):
    python scripts/03_exploratory_analysis.py
"""
import json

import pandas as pd

from milan_forecasting import config
from milan_forecasting.data.loader import load_internet_matrix, square_series
from milan_forecasting.analysis.statistics import (
    DAILY_PERIOD,
    WEEKLY_PERIOD,
    area_totals,
    autocorrelation,
    component_strengths,
    daily_residual_anomalies,
    decompose,
    distribution_stats,
    multiplicative_seasonality_evidence,
    stationarity_tests,
    temporal_profile_stats,
)
from milan_forecasting.analysis.figures import (
    plot_area_series,
    plot_autocorrelation,
    plot_decomposition,
    plot_traffic_distribution,
)
from milan_forecasting.plotting import apply_style

ANOMALY_THRESHOLD = 3.5


def _json_default(obj):
    return obj.item() if hasattr(obj, "item") else str(obj)


def _holidays_in(index: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, str]]:
    days = {d.strftime("%Y-%m-%d") for d in index.normalize()}
    return [
        (pd.Timestamp(day, tz=config.TIMEZONE), name)
        for day, name in config.ITALIAN_HOLIDAYS.items()
        if day in days
    ]


def main() -> None:
    apply_style()
    matrix, timestamps = load_internet_matrix()

    totals = area_totals(matrix)
    ranks = totals.rank(ascending=False, method="min").astype(int)
    top3 = totals.nlargest(3).index.tolist()
    focus = list(config.FOCUS_SQUARES)
    plot_traffic_distribution(totals, top3, focus, config.ANALYSIS_DIR / "traffic_distribution.png")

    series = {sq: square_series(matrix, timestamps, sq) for sq in top3 + focus}
    two_weeks_end = timestamps[0] + pd.Timedelta(days=14) - pd.Timedelta(milliseconds=config.INTERVAL_MS)
    first_two_weeks = {sq: s.loc[timestamps[0]:two_weeks_end] for sq, s in series.items()}
    plot_area_series(
        {f"Square {sq}  ·  rank {ranks[sq]:,} of 10,000": s for sq, s in first_two_weeks.items()},
        _holidays_in(first_two_weeks[top3[0]].index),
        "Internet activity, 1–14 November 2013 (shaded: weekends)",
        config.ANALYSIS_DIR / "first_two_weeks.png",
    )

    busiest = top3[0]
    acf_values, pacf_values, band = autocorrelation(series[busiest])
    plot_autocorrelation(acf_values, pacf_values, band, busiest, config.ANALYSIS_DIR / "autocorrelation.png")

    # Robust fitting keeps holidays and one-off events out of the estimated components; see
    # scripts/04_decomposition_experiment.py for the comparison that motivated it.
    components = decompose(series[busiest], robust=True)
    daily_anomalies, daily_bounds = daily_residual_anomalies(components["residual"], ANOMALY_THRESHOLD)
    eval_window = (
        pd.Timestamp(config.EVAL_START, tz=config.TIMEZONE),
        pd.Timestamp(config.EVAL_END, tz=config.TIMEZONE) + pd.Timedelta(days=1),
    )
    plot_decomposition(components, daily_anomalies, daily_bounds, ANOMALY_THRESHOLD, eval_window, busiest,
                       config.ANALYSIS_DIR / "decomposition.png")

    flagged_days = daily_anomalies[daily_anomalies["flagged"]].copy()
    flagged_days["holiday"] = [config.ITALIAN_HOLIDAYS.get(day, "") for day in flagged_days.index]
    flagged_days = flagged_days.reindex(flagged_days["z_score"].abs().sort_values(ascending=False).index)
    flagged_days = flagged_days.drop(columns="flagged").reset_index()

    significant_pacf = [lag for lag in range(1, pacf_values.size) if abs(pacf_values[lag]) > band]
    report = {
        "top3_squares": top3,
        "ranks": {sq: int(ranks[sq]) for sq in top3 + focus},
        "totals": {sq: float(totals[sq]) for sq in top3 + focus},
        "distribution": distribution_stats(totals),
        "temporal_profiles": {
            sq: {
                "first_two_weeks": temporal_profile_stats(first_two_weeks[sq]),
                "full_period": temporal_profile_stats(series[sq]),
            }
            for sq in top3 + focus
        },
        "busiest_square": {
            "square_id": busiest,
            "acf": {f"lag_{lag}": float(acf_values[lag]) for lag in (1, 6, 36, 72, DAILY_PERIOD, 3 * DAILY_PERIOD, WEEKLY_PERIOD, 2 * WEEKLY_PERIOD)},
            "acf_min_first_day": {"lag": int(acf_values[:DAILY_PERIOD].argmin()), "value": float(acf_values[:DAILY_PERIOD].min())},
            "white_noise_band": band,
            "pacf_first_12": [float(v) for v in pacf_values[1:13]],
            "pacf_significant_lags": significant_pacf,
            "decomposition": {"series": "log(1 + traffic)", "periods": [DAILY_PERIOD, WEEKLY_PERIOD], "robust": True},
            "multiplicative_seasonality": multiplicative_seasonality_evidence(series[busiest], ANOMALY_THRESHOLD),
            "component_strengths": component_strengths(components),
            "stationarity": stationarity_tests(series[busiest]),
            "unusual_days": {
                "threshold": ANOMALY_THRESHOLD,
                "scored_on": "mean residual per day",
                "daily_mean_residual_bounds": list(daily_bounds),
                "flagged": flagged_days.to_dict(orient="records"),
                "flagged_in_eval_week": [
                    day for day in flagged_days["date"] if config.EVAL_START <= day <= config.EVAL_END
                ],
            },
        },
    }
    config.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (config.ANALYSIS_DIR / "statistics.json").write_text(json.dumps(report, indent=2, default=_json_default))
    print(json.dumps(report, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
