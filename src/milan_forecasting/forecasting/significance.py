"""Diebold-Mariano test: are two models' errors actually different, or is it noise?

Errors 10 minutes apart are correlated, so the variance of the loss difference
d_t = L(e_a,t) - L(e_b,t) uses a Newey-West estimator instead of the plain sample variance.
p-values come from t(n - 1), with the Harvey, Leybourne and Newbold (1997) small-sample fix.

Every pair of models is tested, so the p-values also need a family-wise adjustment before any
of them is read as evidence; `holm_adjusted` provides it.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy import stats

LOSSES = {
    "absolute": np.abs,
    "squared": np.square,
}


def newey_west_lag(n: int) -> int:
    """Newey and West's (1994) rule of thumb for the truncation lag."""
    return int(math.floor(4 * (n / 100) ** (2 / 9)))


def long_run_variance(d: np.ndarray, max_lag: int) -> float:
    centred = d - d.mean()
    n = len(d)
    variance = centred @ centred / n
    for lag in range(1, max_lag + 1):
        weight = 1 - lag / (max_lag + 1)
        variance += 2 * weight * (centred[lag:] @ centred[:-lag]) / n
    return float(variance)


def diebold_mariano(actual, forecast_a, forecast_b, loss: str = "absolute", max_lag: int | None = None) -> dict:
    """Test H0: equal expected loss. A negative statistic means forecast_a has the lower loss."""
    actual = np.asarray(actual, dtype=float)
    loss_fn = LOSSES[loss]
    d = loss_fn(actual - np.asarray(forecast_a, dtype=float)) - loss_fn(actual - np.asarray(forecast_b, dtype=float))
    n = len(d)
    lag = newey_west_lag(n) if max_lag is None else max_lag
    variance = long_run_variance(d, lag)
    if variance <= 0:
        return {"loss": loss, "n": n, "lag": lag, "mean_loss_difference": float(d.mean()),
                "statistic": float("nan"), "p_value": float("nan")}
    # One-step-ahead forecasts: the HLN factor sqrt((n + 1 - 2h + h(h - 1)/n) / n) reduces to sqrt((n - 1)/n).
    statistic = d.mean() / math.sqrt(variance / n) * math.sqrt((n - 1) / n)
    p_value = 2 * stats.t.sf(abs(statistic), df=n - 1)
    return {"loss": loss, "n": n, "lag": lag, "mean_loss_difference": float(d.mean()),
            "statistic": float(statistic), "p_value": float(p_value)}


def holm_adjusted(p_values: Sequence[float]) -> list[float]:
    """Return Holm-Bonferroni adjusted p-values in the original input order."""
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted
