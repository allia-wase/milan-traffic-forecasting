import math

import numpy as np
import pytest

from milan_forecasting.forecasting.significance import (
    diebold_mariano,
    holm_adjusted,
    long_run_variance,
    newey_west_lag,
)


def test_clearly_better_forecast_is_significant_and_negative():
    rng = np.random.default_rng(0)
    actual = rng.normal(100, 10, 1000)
    good = actual + rng.normal(0, 1, 1000)
    bad = actual + rng.normal(0, 5, 1000)

    result = diebold_mariano(actual, good, bad)

    assert result["statistic"] < 0
    assert result["p_value"] < 1e-6
    assert result["mean_loss_difference"] < 0


def test_forecasts_with_the_same_error_distribution_are_not_significant():
    rng = np.random.default_rng(1)
    actual = rng.normal(100, 10, 1000)
    a = actual + rng.normal(0, 2, 1000)
    b = actual + rng.normal(0, 2, 1000)

    assert diebold_mariano(actual, a, b, loss="squared")["p_value"] > 0.05


def test_identical_forecasts_give_no_statistic():
    actual = np.arange(10.0)
    result = diebold_mariano(actual, actual + 1, actual + 1)

    assert math.isnan(result["statistic"])


def test_long_run_variance_grows_with_positive_autocorrelation():
    rng = np.random.default_rng(2)
    noise = rng.normal(size=5000)
    persistent = np.convolve(noise, np.ones(5) / 5, mode="valid")

    assert long_run_variance(persistent, 10) > 3 * long_run_variance(persistent, 0)


def test_newey_west_lag_for_a_test_week():
    assert newey_west_lag(1008) == 6


def test_holm_adjusted_follows_the_step_down_rule():
    assert holm_adjusted([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_adjusted_is_monotone_and_capped_at_one():
    assert holm_adjusted([0.2, 0.3, 0.4]) == pytest.approx([0.6, 0.6, 0.6])
    assert holm_adjusted([0.5, 0.6]) == pytest.approx([1.0, 1.0])


def test_holm_adjusted_never_lowers_a_p_value():
    rng = np.random.default_rng(3)
    raw = list(rng.uniform(0, 1, 20))
    assert all(adjusted >= p for adjusted, p in zip(holm_adjusted(raw), raw))
