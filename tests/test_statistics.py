import numpy as np
import pandas as pd
import pytest

from milan_forecasting import config
from milan_forecasting.analysis.statistics import (
    daily_residual_anomalies,
    distribution_stats,
    residual_anomalies,
    square_to_grid,
    temporal_profile_stats,
)


def test_square_to_grid_is_row_major():
    assert square_to_grid(1) == (0, 0)
    assert square_to_grid(100) == (0, 99)
    assert square_to_grid(101) == (1, 0)
    assert square_to_grid(10000) == (99, 99)


@pytest.mark.filterwarnings("ignore:Precision loss occurred in moment calculation")
def test_gini_and_shares_at_the_extremes():
    equal = distribution_stats(pd.Series(np.ones(1000)))
    assert equal["gini"] == pytest.approx(0, abs=1e-9)
    assert equal["top_10pct_share"] == pytest.approx(0.10)

    concentrated = distribution_stats(pd.Series(np.r_[np.full(999, 1e-9), 1.0]))
    assert concentrated["gini"] == pytest.approx(0.999, abs=1e-3)
    assert concentrated["top_1pct_share"] == pytest.approx(1.0, abs=1e-6)


def test_temporal_profile_finds_evening_peak_and_weekend_drop():
    index = pd.date_range("2013-11-04", periods=14 * config.INTERVALS_PER_DAY, freq="10min", tz=config.TIMEZONE)
    evening_peak = np.where(index.hour == 20, 10.0, 1.0)
    weekend_factor = np.where(index.dayofweek >= 5, 0.5, 1.0)
    profile = temporal_profile_stats(pd.Series(evening_peak * weekend_factor, index=index))

    assert profile["peak_hour"] == 20
    assert profile["weekend_to_weekday_ratio"] == pytest.approx(0.5)
    assert profile["zero_share"] == 0


def test_daily_residual_anomalies_flag_one_unusual_day():
    index = pd.date_range("2013-11-01", periods=30 * config.INTERVALS_PER_DAY, freq="10min", tz=config.TIMEZONE)
    residual = pd.Series(np.random.default_rng(3).normal(0, 0.05, len(index)), index=index)
    residual[residual.index.strftime("%Y-%m-%d") == "2013-11-20"] -= 1.5

    frame, (low, high) = daily_residual_anomalies(residual, threshold=3.5)

    assert frame.loc["2013-11-20", "flagged"]
    assert int(frame["flagged"].sum()) == 1
    assert low < 0 < high


def test_residual_anomalies_flag_an_isolated_spike():
    residual = pd.Series(np.random.default_rng(0).normal(0, 1, 5000))
    residual.iloc[1234] = 25.0

    mask, (low, high), _ = residual_anomalies(residual, threshold=3.5)

    assert mask.iloc[1234]
    assert mask.sum() <= 10
    assert low < 0 < high
