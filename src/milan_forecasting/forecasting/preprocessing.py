"""Chronological splits, log-standardisation and sliding windows for one-step-ahead forecasting.

Index convention: position t in a series is the interval being predicted, so the model input for
target t is the `lookback` observed values at positions t - lookback ... t - 1. This matches the
brief's formulation of predicting x(t + 1) from a history ending at t.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from milan_forecasting import config


@dataclass(frozen=True)
class Splits:
    train: slice
    val: slice
    test: slice


def chronological_splits(index: pd.DatetimeIndex) -> Splits:
    """Train up to 8 Dec, validate on 9-15 Dec, test on 16-22 Dec (all Milan local time)."""

    def day_start(day: str) -> int:
        return int(index.searchsorted(pd.Timestamp(day, tz=config.TIMEZONE)))

    val_start = day_start(config.VAL_START)
    test_start = day_start(config.EVAL_START)
    test_stop = day_start(config.EVAL_END) + config.INTERVALS_PER_DAY
    return Splits(slice(0, val_start), slice(val_start, test_start), slice(test_start, test_stop))


def split_targets(split: slice, lookback: int = 0) -> np.ndarray:
    """Target positions in a split, skipping any whose input window would start before the series."""
    return np.arange(max(split.start, lookback), split.stop)


@dataclass(frozen=True)
class LogStandardiser:
    """log(1 + x), then standardised with statistics from the training split only."""

    mean: float
    std: float

    @classmethod
    def fit(cls, values: np.ndarray) -> LogStandardiser:
        logged = np.log1p(np.asarray(values, dtype=np.float64))
        return cls(float(logged.mean()), float(logged.std()))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.log1p(np.asarray(values, dtype=np.float64)) - self.mean) / self.std

    def inverse(self, scaled: np.ndarray) -> np.ndarray:
        return np.expm1(np.asarray(scaled, dtype=np.float64) * self.std + self.mean)


def sliding_windows(scaled: np.ndarray, targets: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Inputs of shape (n_targets, lookback) and the matching target values."""
    targets = np.asarray(targets)
    if targets.min() < lookback:
        raise ValueError(f"target {targets.min()} has fewer than {lookback} past observations")
    inputs = scaled[targets[:, None] + np.arange(-lookback, 0)]
    return inputs.astype(np.float32), scaled[targets].astype(np.float32)
