"""Forecast error metrics, all computed on the original traffic scale."""
import numpy as np

from milan_forecasting import config


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute percentage error, in percent. Undefined when any actual value is zero."""
    if np.any(actual == 0):
        raise ValueError("MAPE is undefined for zero actual values")
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


def seasonal_naive_scale(train_values: np.ndarray, season: int = config.INTERVALS_PER_DAY) -> float:
    """In-sample MAE of the same-time-yesterday forecast, the denominator of MASE."""
    return float(np.mean(np.abs(train_values[season:] - train_values[:-season])))


def evaluate(actual: np.ndarray, predicted: np.ndarray, mase_scale: float) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    error = mae(actual, predicted)
    return {
        "mae": error,
        "rmse": rmse(actual, predicted),
        "mape": mape(actual, predicted),
        "mase": error / mase_scale,
    }
