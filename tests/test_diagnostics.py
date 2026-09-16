import numpy as np
import pandas as pd
import pytest

from milan_forecasting import config
from milan_forecasting.forecasting.diagnostics import (
    error_by_day,
    error_by_move_size,
    largest_misses,
    load_predictions,
    reaction_summary,
    under_reaction,
    weekend_to_weekday_ratio,
)


def _frame(actual, **predictions):
    index = pd.date_range("2013-12-16", periods=len(actual), freq="10min", tz=config.TIMEZONE)
    columns = {"actual": actual, "naive": actual, "arima": actual, "lstm": actual, "tcn": actual}
    columns.update(predictions)
    return pd.DataFrame(columns, index=index, dtype=float)


def test_under_reaction_separates_falling_short_from_overshooting():
    actual = pd.Series([100.0, 200.0, 150.0, 160.0])
    predicted = pd.Series([100.0, 150.0, 180.0, 190.0])
    moves = actual.diff()

    flags = under_reaction(actual, predicted, moves)

    # +100 predicted as +50 (short), -50 predicted as -20 (short), +10 predicted as +40 (overshoot)
    assert flags.tolist() == [True, True, False]


def test_reaction_summary_on_a_lagging_forecast():
    actual = np.array([100, 300, 100, 400, 100, 350, 120, 380], dtype=float)
    lagged = np.r_[actual[0], actual[:-1]]
    summary = reaction_summary(_frame(actual, arima=lagged), "arima", top_share=0.25)

    assert summary["under_reaction_share_all"] == pytest.approx(1.0)
    assert summary["under_reaction_share_largest"] == pytest.approx(1.0)
    assert summary["corr_abs_error_abs_move"] == pytest.approx(1.0)


def test_error_by_day_labels_calendar_days():
    actual = np.full(2 * config.INTERVALS_PER_DAY, 100.0)
    arima = actual + np.r_[np.full(144, 10.0), np.full(144, -20.0)]
    daily = error_by_day(_frame(actual, arima=arima))

    assert list(daily.index) == ["Mon 16", "Tue 17"]
    assert daily.loc["Mon 16", "arima"] == pytest.approx(10)
    assert daily.loc["Tue 17", "arima"] == pytest.approx(20)


def test_error_by_move_size_increases_when_errors_track_moves():
    rng = np.random.default_rng(0)
    actual = np.cumsum(rng.normal(0, 50, 500)) + 5000
    lagged = np.r_[actual[0], actual[:-1]]
    curve = error_by_move_size(_frame(actual, arima=lagged), "arima", bins=5)

    assert len(curve) == 5
    assert curve.is_monotonic_increasing
    assert curve.index.is_monotonic_increasing


def test_largest_misses_and_prediction_round_trip(tmp_path):
    frame = _frame([100.0, 200.0, 300.0], arima=[100.0, 260.0, 290.0])
    top = largest_misses(frame, "arima", 1)
    assert top["abs_error"].iloc[0] == pytest.approx(60)
    assert top["previous_actual"].iloc[0] == pytest.approx(100)

    path = tmp_path / "predictions.csv"
    frame.to_csv(path, index_label="timestamp")
    loaded = load_predictions(path)
    assert str(loaded.index.tz) == config.TIMEZONE
    assert (loaded.index == frame.index).all()
    np.testing.assert_allclose(loaded["arima"], frame["arima"])


def test_weekend_to_weekday_ratio():
    daily = pd.DataFrame({"arima": [10.0, 10.0, 30.0, 20.0]}, index=["Mon 16", "Tue 17", "Sat 21", "Sun 22"])
    assert weekend_to_weekday_ratio(daily) == {"arima": pytest.approx(2.5)}
