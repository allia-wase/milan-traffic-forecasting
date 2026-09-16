"""Error diagnostics computed from saved test-week predictions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config

PREDICTED_MODELS = ("naive", "arima", "lstm", "tcn")


def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("timestamp"), utc=True)).tz_convert(config.TIMEZONE)
    return frame


def error_by_day(predictions: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute error per model for each calendar day, labelled like 'Mon 16'."""
    labels = predictions.index.strftime("%a %d")
    errors = predictions[list(PREDICTED_MODELS)].sub(predictions["actual"], axis=0).abs()
    return errors.groupby(labels, sort=False).mean()


def actual_moves(predictions: pd.DataFrame) -> pd.Series:
    """Observed 10-minute change at each target; the first target has no earlier value in the file."""
    return predictions["actual"].diff()


def under_reaction(actual: pd.Series, predicted: pd.Series, moves: pd.Series) -> pd.Series:
    """True where the forecast fell short of the actual move or pointed the other way.

    With error = predicted - actual, the forecast under-reacts when the error has the opposite sign
    to the move: a rise it under-predicted or a drop it over-predicted.
    """
    valid = moves.notna() & (moves != 0)
    error = (predicted - actual)[valid]
    return np.sign(error) == -np.sign(moves[valid])


def reaction_summary(predictions: pd.DataFrame, model: str, top_share: float = 0.05) -> dict[str, float]:
    moves = actual_moves(predictions)
    under = under_reaction(predictions["actual"], predictions[model], moves)
    abs_error = (predictions[model] - predictions["actual"]).abs()[under.index]
    largest = abs_error >= abs_error.quantile(1 - top_share)
    return {
        "under_reaction_share_all": float(under.mean()),
        "under_reaction_share_largest": float(under[largest].mean()),
        "largest_share": top_share,
        "corr_abs_error_abs_move": float(np.corrcoef(abs_error, moves[under.index].abs())[0, 1]),
    }


def error_by_move_size(predictions: pd.DataFrame, model: str, bins: int = 10) -> pd.Series:
    """Mean absolute error within equal-count bins of the absolute 10-minute move, indexed by bin median."""
    moves = actual_moves(predictions).abs()
    frame = pd.DataFrame({
        "move": moves,
        "error": (predictions[model] - predictions["actual"]).abs(),
    }).dropna()
    frame["bin"] = pd.qcut(frame["move"], bins, labels=False, duplicates="drop")
    grouped = frame.groupby("bin")
    return pd.Series(grouped["error"].mean().to_numpy(), index=grouped["move"].median().to_numpy(), name=model)


def largest_misses(predictions: pd.DataFrame, model: str, count: int = 10) -> pd.DataFrame:
    frame = predictions.assign(
        previous_actual=predictions["actual"].shift(1),
        abs_error=(predictions[model] - predictions["actual"]).abs(),
    )
    return frame.nlargest(count, "abs_error")
