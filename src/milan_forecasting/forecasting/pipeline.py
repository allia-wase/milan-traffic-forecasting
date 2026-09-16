"""Fit any of the compared forecasters on one area and score it on one split.

Every model sees the same splits, the same training-only scaling and the same target positions,
so differences in the results come from the models rather than from the data handling.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from milan_forecasting.forecasting.preprocessing import (
    LogStandardiser,
    chronological_splits,
    sliding_windows,
    split_targets,
)
from milan_forecasting.forecasting.metrics import evaluate, seasonal_naive_scale
from milan_forecasting.forecasting.models.neural import TrainConfig, build_model, predict, train_model
from milan_forecasting.forecasting.models.statistical import ArimaFourierConfig, ArimaFourierForecaster, seasonal_naive

MODELS = ("naive", "arima", "lstm", "tcn")


@dataclass
class RunResult:
    model: str
    params: dict
    split: str
    timestamps: pd.DatetimeIndex
    actual: np.ndarray
    predicted: np.ndarray
    metrics: dict[str, float]
    train_seconds: float
    predict_seconds: float
    details: dict = field(default_factory=dict)


def run(model_name: str, series: pd.Series, params: dict, split_name: str) -> RunResult:
    if model_name not in MODELS:
        raise ValueError(f"unknown model {model_name!r}; choose from {MODELS}")
    values = series.to_numpy(dtype=np.float64)
    splits = chronological_splits(series.index)
    split = getattr(splits, split_name)
    targets = split_targets(split)

    scaler = LogStandardiser.fit(values[splits.train])
    scaled = scaler.transform(values)
    details: dict = {}

    if model_name == "naive":
        train_seconds = 0.0
        started = time.perf_counter()
        predicted = seasonal_naive(values, targets)
        predict_seconds = time.perf_counter() - started

    elif model_name == "arima":
        if "periods" in params:
            params = {**params, "periods": tuple(params["periods"])}
        cfg = ArimaFourierConfig(**params)
        started = time.perf_counter()
        forecaster = ArimaFourierForecaster(cfg).fit(scaled, splits.train)
        train_seconds = time.perf_counter() - started
        started = time.perf_counter()
        predicted = scaler.inverse(forecaster.predict(scaled, targets))
        predict_seconds = time.perf_counter() - started
        params = cfg.as_dict()
        details = {"aic": forecaster.aic, "converged": forecaster.converged, "parameters": len(forecaster.result.params)}

    else:
        cfg = TrainConfig(**params)
        model = build_model(model_name, cfg)
        train_windows = sliding_windows(scaled, split_targets(splits.train, cfg.lookback), cfg.lookback)
        val_windows = sliding_windows(scaled, split_targets(splits.val, cfg.lookback), cfg.lookback)
        result = train_model(model, train_windows, val_windows, cfg)
        train_seconds = result.seconds
        inputs, _ = sliding_windows(scaled, targets, cfg.lookback)
        started = time.perf_counter()
        predicted = scaler.inverse(predict(model, inputs))
        predict_seconds = time.perf_counter() - started
        params = cfg.as_dict()
        details = {
            "best_epoch": result.best_epoch,
            "epochs_run": result.epochs_run,
            "best_val_loss": result.best_val_loss,
            "parameters": sum(p.numel() for p in model.parameters()),
            "train_losses": result.train_losses,
            "val_losses": result.val_losses,
        }

    actual = values[targets]
    return RunResult(
        model=model_name,
        params=params,
        split=split_name,
        timestamps=series.index[targets],
        actual=actual,
        predicted=predicted,
        metrics=evaluate(actual, predicted, seasonal_naive_scale(values[splits.train])),
        train_seconds=train_seconds,
        predict_seconds=predict_seconds,
        details=details,
    )
