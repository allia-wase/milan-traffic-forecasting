"""Statistics for the exploratory analysis of the Internet-activity matrix."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from src import config

GRID_SIDE = 100
DAILY_PERIOD = config.INTERVALS_PER_DAY
WEEKLY_PERIOD = 7 * config.INTERVALS_PER_DAY
ROBUST_SIGMA = 1.4826  # scales the median absolute deviation to a standard deviation for normal data


def area_totals(matrix: np.ndarray) -> pd.Series:
    totals = np.asarray(matrix.sum(axis=1, dtype=np.float64))
    index = pd.RangeIndex(1, config.N_SQUARES + 1, name="square_id")
    return pd.Series(totals, index=index, name="total_internet")


def square_to_grid(square_id: int) -> tuple[int, int]:
    """(row, column) of a square, assuming IDs run row by row from the grid's first corner."""
    return divmod(square_id - 1, GRID_SIDE)


def totals_grid(totals: pd.Series) -> np.ndarray:
    return totals.sort_index().to_numpy().reshape(GRID_SIDE, GRID_SIDE)


def distribution_stats(totals: pd.Series) -> dict:
    values = np.sort(totals.to_numpy(dtype=np.float64))
    n = values.size
    grand_total = values.sum()
    cumulative_share = np.cumsum(values) / grand_total

    def top_share(fraction: float) -> float:
        return float(values[-int(n * fraction):].sum() / grand_total)

    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "min": float(values[0]),
        "max": float(values[-1]),
        "p01": float(np.percentile(values, 1)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p99": float(np.percentile(values, 99)),
        "mean_to_median": float(values.mean() / np.median(values)),
        "max_to_median": float(values[-1] / np.median(values)),
        "skewness": float(stats.skew(values)),
        "skewness_log10": float(stats.skew(np.log10(values))),
        "gini": float((n + 1 - 2 * cumulative_share.sum()) / n),
        "top_1pct_share": top_share(0.01),
        "top_10pct_share": top_share(0.10),
        "bottom_50pct_share": float(cumulative_share[n // 2 - 1]),
    }


def temporal_profile_stats(series: pd.Series) -> dict:
    values = series.astype(np.float64)
    hourly = values.groupby(values.index.hour).mean()
    is_weekend = values.index.dayofweek >= 5
    return {
        "mean": float(values.mean()),
        "coefficient_of_variation": float(values.std() / values.mean()),
        "peak_hour": int(hourly.idxmax()),
        "trough_hour": int(hourly.idxmin()),
        "peak_to_trough_ratio": float(hourly.max() / hourly.min()),
        "weekend_to_weekday_ratio": float(values[is_weekend].mean() / values[~is_weekend].mean()),
        "lag1_autocorrelation": float(values.autocorr(1)),
        "lag144_autocorrelation": float(values.autocorr(DAILY_PERIOD)),
        "max_step_change_to_mean": float(values.diff().abs().max() / values.mean()),
        "zero_share": float((values == 0).mean()),
    }


def autocorrelation(
    series: pd.Series, acf_lags: int = 2 * WEEKLY_PERIOD, pacf_lags: int = 48
) -> tuple[np.ndarray, np.ndarray, float]:
    """ACF and PACF values plus the approximate 95% band for white noise (±1.96/√n)."""
    values = series.to_numpy(dtype=np.float64)
    band = 1.96 / np.sqrt(values.size)
    return acf(values, nlags=acf_lags, fft=True), pacf(values, nlags=pacf_lags, method="ywm"), float(band)


def decompose(series: pd.Series, log_transform: bool = True, robust: bool = False) -> pd.DataFrame:
    """MSTL decomposition with daily and weekly seasonal periods.

    With log_transform the series is modelled as log(1 + x), which turns multiplicative seasonality
    (daily swings that grow with the traffic level) into additive components. With robust, STL
    down-weights outlying points when estimating the components, at a noticeable cost in run time.
    """
    values = series.to_numpy(dtype=np.float64)
    if log_transform:
        values = np.log1p(values)
    fit = MSTL(values, periods=(DAILY_PERIOD, WEEKLY_PERIOD), stl_kwargs={"robust": robust}).fit()
    return pd.DataFrame(
        {
            "observed": fit.observed,
            "trend": fit.trend,
            "seasonal_daily": fit.seasonal[:, 0],
            "seasonal_weekly": fit.seasonal[:, 1],
            "residual": fit.resid,
        },
        index=series.index,
    )


def component_strengths(components: pd.DataFrame) -> dict:
    """Strength of trend and seasonality, F = max(0, 1 - Var(R) / Var(X + R))."""
    residual_var = components["residual"].var()

    def strength(column: str) -> float:
        return float(max(0.0, 1 - residual_var / (components[column] + components["residual"]).var()))

    return {
        "trend": strength("trend"),
        "daily_seasonality": strength("seasonal_daily"),
        "weekly_seasonality": strength("seasonal_weekly"),
    }


def multiplicative_seasonality_evidence(series: pd.Series, threshold: float = 3.5) -> dict:
    """Evidence that seasonal swings scale with the level, comparing raw and log decompositions.

    Both decompositions are fitted without robustness weights so the two are compared on equal terms.
    """
    daily = series.groupby(series.index.strftime("%Y-%m-%d")).agg(["mean", "min", "max"])
    daily_range = daily["max"] - daily["min"]
    evidence = {
        "corr_daily_mean_vs_daily_range": float(np.corrcoef(daily["mean"], daily_range)[0, 1]),
        "corr_daily_mean_vs_relative_range": float(np.corrcoef(daily["mean"], daily_range / daily["mean"])[0, 1]),
    }
    fits = (("raw", decompose(series, log_transform=False)), ("log1p", decompose(series, log_transform=True)))
    for name, components in fits:
        residual = components["residual"]
        fitted = components["trend"] + components["seasonal_daily"] + components["seasonal_weekly"]
        std_by_hour = residual.groupby(residual.index.hour).std()
        mask, _, _ = residual_anomalies(residual, threshold)
        evidence[name] = {
            "residual_std_max_to_min_by_hour": float(std_by_hour.max() / std_by_hour.min()),
            "corr_fitted_level_vs_abs_residual": float(residual.abs().corr(fitted)),
            "anomaly_share": float(mask.mean()),
        }
    return evidence


def stationarity_tests(series: pd.Series) -> dict:
    """ADF (null: unit root) and KPSS (null: level-stationary) on the level and on differenced series."""
    variants = {
        "level": series,
        "first_difference": series.diff(),
        "daily_seasonal_difference": series.diff(DAILY_PERIOD),
    }
    results = {}
    for name, variant in variants.items():
        values = variant.dropna().to_numpy(dtype=np.float64)
        adf_stat, adf_p, adf_lags, *_ = adfuller(values, autolag="AIC", result_object=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", InterpolationWarning)
            kpss_stat, kpss_p, kpss_lags, _ = kpss(values, regression="c", nlags="auto", result_object=False)
        results[name] = {
            "adf_statistic": float(adf_stat),
            "adf_p_value": float(adf_p),
            "adf_lags_used": int(adf_lags),
            "kpss_statistic": float(kpss_stat),
            "kpss_p_value": float(kpss_p),
            # KPSS p-values are interpolated from a table and clipped to [0.01, 0.10].
            "kpss_p_value_is_table_bound": any(issubclass(w.category, InterpolationWarning) for w in caught),
            "kpss_lags_used": int(kpss_lags),
        }
    return results


def residual_anomalies(
    residual: pd.Series, threshold: float = 3.5
) -> tuple[pd.Series, tuple[float, float], pd.Series]:
    """Flag residuals whose modified z-score (median/MAD based) exceeds the threshold."""
    median = residual.median()
    scale = ROBUST_SIGMA * (residual - median).abs().median()
    z_scores = (residual - median) / scale
    bounds = (float(median - threshold * scale), float(median + threshold * scale))
    return z_scores.abs() > threshold, bounds, z_scores


def daily_residual_anomalies(
    residual: pd.Series, threshold: float = 3.5
) -> tuple[pd.DataFrame, tuple[float, float]]:
    """Score whole days by their average residual and flag the unusual ones.

    Scoring days rather than single 10-minute points keeps the result comparable between
    decompositions: the point-level scale depends on how closely the fit tracks ordinary variation.
    """
    daily_mean = residual.groupby(residual.index.strftime("%Y-%m-%d")).mean()
    median = daily_mean.median()
    scale = ROBUST_SIGMA * (daily_mean - median).abs().median()
    z_scores = (daily_mean - median) / scale
    frame = pd.DataFrame({
        "mean_residual": daily_mean,
        "z_score": z_scores,
        "flagged": z_scores.abs() > threshold,
    })
    frame.index.name = "date"
    return frame, (float(median - threshold * scale), float(median + threshold * scale))
