import numpy as np
import pandas as pd
import pytest
import torch

from milan_forecasting import config
from milan_forecasting.forecasting.preprocessing import (
    LogStandardiser,
    chronological_splits,
    sliding_windows,
    split_targets,
)
from milan_forecasting.forecasting.metrics import evaluate, mape, seasonal_naive_scale
from milan_forecasting.forecasting.models.neural import (
    LSTMForecaster,
    TCNForecaster,
    TrainConfig,
    build_model,
    train_model,
)
from milan_forecasting.forecasting.models.statistical import (
    ArimaFourierConfig,
    ArimaFourierForecaster,
    fourier_terms,
    seasonal_naive,
)


def full_index():
    return pd.date_range("2013-11-01", periods=62 * config.INTERVALS_PER_DAY, freq="10min", tz=config.TIMEZONE)


def test_splits_cover_the_expected_weeks():
    index = full_index()
    splits = chronological_splits(index)

    assert splits.train == slice(0, 38 * 144)
    assert splits.val.stop - splits.val.start == 7 * 144
    assert splits.test.stop - splits.test.start == 7 * 144
    assert index[splits.val.start] == pd.Timestamp("2013-12-09", tz=config.TIMEZONE)
    assert index[splits.test.start] == pd.Timestamp("2013-12-16", tz=config.TIMEZONE)
    assert index[splits.test.stop - 1] == pd.Timestamp("2013-12-22 23:50", tz=config.TIMEZONE)


def test_log_standardiser_round_trips():
    values = np.array([0.0, 10.0, 250.0, 4000.0])
    scaler = LogStandardiser.fit(values)
    np.testing.assert_allclose(scaler.inverse(scaler.transform(values)), values, atol=1e-9)
    assert scaler.transform(values).mean() == pytest.approx(0)


def test_sliding_windows_use_only_past_values():
    scaled = np.arange(10, dtype=np.float64)
    inputs, targets = sliding_windows(scaled, np.array([5, 6]), lookback=3)

    np.testing.assert_array_equal(inputs, [[2, 3, 4], [3, 4, 5]])
    np.testing.assert_array_equal(targets, [5, 6])
    with pytest.raises(ValueError):
        sliding_windows(scaled, np.array([2]), lookback=3)
    np.testing.assert_array_equal(split_targets(slice(0, 6), lookback=3), [3, 4, 5])


def test_seasonal_naive_and_fourier_terms():
    np.testing.assert_array_equal(seasonal_naive(np.arange(300), np.array([150, 200])), [6, 56])
    terms = fourier_terms(10, period=5, harmonics=2)
    assert terms.shape == (10, 4)
    np.testing.assert_allclose(terms[5], terms[0], atol=1e-12)
    assert fourier_terms(10, period=5, harmonics=0).shape == (10, 0)


def test_arima_exog_drops_weekly_harmonics_that_repeat_daily_ones():
    cfg = ArimaFourierConfig(daily_harmonics=16, weekly_harmonics=8)
    exog = ArimaFourierForecaster(cfg)._exog(2016)

    assert exog.shape == (2016, 2 * 16 + 2 * 7)
    assert np.linalg.matrix_rank(exog) == exog.shape[1]


def test_metrics_on_known_values():
    actual, predicted = np.array([100.0, 200.0]), np.array([110.0, 180.0])
    result = evaluate(actual, predicted, mase_scale=5.0)

    assert result["mae"] == pytest.approx(15)
    assert result["rmse"] == pytest.approx(np.sqrt((100 + 400) / 2))
    assert result["mape"] == pytest.approx(10)
    assert result["mase"] == pytest.approx(3)
    assert seasonal_naive_scale(np.arange(10.0), season=2) == pytest.approx(2)
    with pytest.raises(ValueError):
        mape(np.array([0.0]), np.array([1.0]))


def test_network_output_shapes_and_receptive_field():
    batch = torch.randn(4, 144)
    assert LSTMForecaster(hidden_size=8)(batch).shape == (4,)

    tcn = TCNForecaster(channels=8, levels=6, kernel_size=3)
    assert tcn(batch).shape == (4,)
    assert tcn.receptive_field == 253
    with pytest.raises(ValueError, match="receptive field"):
        build_model("tcn", TrainConfig(lookback=1008, model={"channels": 8, "levels": 6}))


def test_tcn_is_causal():
    tcn = TCNForecaster(channels=4, levels=3, kernel_size=2, dropout=0.0).eval()
    x = torch.randn(1, 20)
    features = tcn.network(x.unsqueeze(1))
    changed = x.clone()
    changed[0, 10] += 5.0
    changed_features = tcn.network(changed.unsqueeze(1))

    torch.testing.assert_close(changed_features[..., :10], features[..., :10])
    assert not torch.allclose(changed_features[..., 10], features[..., 10])


def test_training_reduces_validation_loss_on_a_sine_wave():
    t = np.arange(600)
    scaled = np.sin(2 * np.pi * t / 24).astype(np.float64)
    cfg = TrainConfig(lookback=24, batch_size=32, max_epochs=8, patience=8, model={"hidden_size": 16})
    model = build_model("lstm", cfg)
    train = sliding_windows(scaled, np.arange(24, 450), 24)
    val = sliding_windows(scaled, np.arange(450, 600), 24)

    result = train_model(model, train, val, cfg)

    assert result.best_val_loss < result.val_losses[0]
    assert result.best_val_loss < 0.05


def test_arima_fourier_fits_a_seasonal_series_without_lookahead():
    t = np.arange(400)
    scaled = np.sin(2 * np.pi * t / 24) + np.random.default_rng(0).normal(0, 0.05, t.size)
    cfg = ArimaFourierConfig(p=1, d=0, q=0, daily_harmonics=2, weekly_harmonics=0, periods=(24, 168))
    forecaster = ArimaFourierForecaster(cfg).fit(scaled, slice(0, 300))

    targets = np.arange(300, 400)
    predictions = forecaster.predict(scaled, targets)
    assert np.mean(np.abs(predictions - scaled[targets])) < 0.1

    future_changed = scaled.copy()
    future_changed[311:] += 10.0
    assert forecaster.predict(future_changed, np.array([310]))[0] == pytest.approx(predictions[10])
