"""Seasonal-naive baseline and ARIMA with Fourier seasonal regressors."""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

from milan_forecasting import config

DAILY = config.INTERVALS_PER_DAY
WEEKLY = 7 * config.INTERVALS_PER_DAY


def seasonal_naive(values: np.ndarray, targets: np.ndarray, season: int = DAILY) -> np.ndarray:
    """Predict each interval with the value observed one season earlier."""
    return np.asarray(values)[np.asarray(targets) - season]


def fourier_terms(n_obs: int, period: int, harmonics: int) -> np.ndarray:
    """Sine/cosine pairs for the first `harmonics` frequencies of a period, shape (n_obs, 2 * harmonics)."""
    if harmonics == 0:
        return np.empty((n_obs, 0))
    angles = 2 * np.pi * np.arange(n_obs)[:, None] * np.arange(1, harmonics + 1) / period
    return np.hstack([np.sin(angles), np.cos(angles)])


@dataclass(frozen=True)
class ArimaFourierConfig:
    p: int = 2
    d: int = 0
    q: int = 1
    daily_harmonics: int = 8
    weekly_harmonics: int = 4
    periods: tuple[int, int] = (DAILY, WEEKLY)

    def as_dict(self) -> dict:
        return asdict(self)


class ArimaFourierForecaster:
    """ARIMA errors around a Fourier representation of the daily and weekly cycles.

    Seasonal ARIMA would need a 144-step seasonal polynomial, which is impractical; Fourier terms
    describe the same cycles with 2 * K parameters per period (dynamic harmonic regression).
    """

    def __init__(self, cfg: ArimaFourierConfig):
        self.cfg = cfg
        self.result = None

    def _exog(self, n_obs: int) -> np.ndarray:
        daily_period, weekly_period = self.cfg.periods
        daily = fourier_terms(n_obs, daily_period, self.cfg.daily_harmonics)
        weekly = fourier_terms(n_obs, weekly_period, self.cfg.weekly_harmonics)
        return np.hstack([daily, weekly[:, self._distinct_weekly_columns()]])

    def _distinct_weekly_columns(self) -> list[int]:
        """Drop weekly harmonics that repeat a daily one.

        With a weekly period of 7 days, weekly harmonic 7k has the same frequency as daily harmonic k;
        keeping both makes the regressors perfectly collinear and the fit fails.
        """
        daily_period, weekly_period = self.cfg.periods
        k_weekly = self.cfg.weekly_harmonics
        keep = []
        for k in range(1, k_weekly + 1):
            daily_equivalent, remainder = divmod(k * daily_period, weekly_period)
            if remainder == 0 and daily_equivalent <= self.cfg.daily_harmonics:
                continue
            keep.append(k)
        return [k - 1 for k in keep] + [k_weekly + k - 1 for k in keep]

    def fit(self, scaled: np.ndarray, train: slice) -> ArimaFourierForecaster:
        if train.start != 0:
            raise ValueError("training must start at the beginning of the series so Fourier phases align")
        exog = self._exog(train.stop)
        model = SARIMAX(
            scaled[train], exog=exog if exog.size else None,
            order=(self.cfg.p, self.cfg.d, self.cfg.q), trend="c" if self.cfg.d == 0 else "n",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self.result = model.fit(disp=False, maxiter=200)
        self.converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        return self

    def predict(self, scaled: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """One-step-ahead predictions: each uses observations strictly before its target.

        The fitted parameters are re-applied to the longer series without re-estimation, and the
        Kalman filter's one-step predictions are read off at the target positions.
        """
        targets = np.asarray(targets)
        stop = int(targets.max()) + 1
        exog = self._exog(stop)
        extended = self.result.apply(scaled[:stop], exog=exog if exog.size else None, refit=False)
        predictions = extended.predict(start=int(targets.min()), end=stop - 1)
        return np.asarray(predictions)[targets - targets.min()]

    @property
    def aic(self) -> float:
        return float(self.result.aic)
