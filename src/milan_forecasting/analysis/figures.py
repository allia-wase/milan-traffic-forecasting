"""Figures for the exploratory analysis."""
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from milan_forecasting.analysis.statistics import DAILY_PERIOD, WEEKLY_PERIOD, square_to_grid, totals_grid
from milan_forecasting.plotting import (
    CRITICAL,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SERIES,
    SHADE,
    SURFACE,
    save,
    sequential_cmap,
)

LABEL_BOX = {"boxstyle": "round,pad=0.25", "facecolor": SURFACE, "edgecolor": "none", "alpha": 0.9}


def _date_numbers(index: pd.DatetimeIndex) -> np.ndarray:
    return mdates.date2num(index.to_pydatetime())


def _date_number(timestamp: pd.Timestamp) -> float:
    return mdates.date2num(timestamp.to_pydatetime())


def _shade_weekends(ax: plt.Axes, index: pd.DatetimeIndex) -> None:
    for day in pd.date_range(index[0].normalize(), index[-1].normalize(), freq="D"):
        if day.dayofweek == 5:
            ax.axvspan(_date_number(day), _date_number(day + pd.Timedelta(days=2)), color=SHADE, linewidth=0, zorder=0)


def plot_traffic_distribution(totals: pd.Series, top: list[int], focus: list[int], path: Path) -> None:
    fig, (ax_hist, ax_map) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.5, 1]})

    values = totals.to_numpy()
    bins = np.logspace(np.log10(values.min()), np.log10(values.max()), 60)
    counts, _, _ = ax_hist.hist(values, bins=bins, color=SERIES[0], edgecolor=SURFACE, linewidth=0.5)
    ax_hist.set_xscale("log")
    ax_hist.set_ylim(0, counts.max() * 1.4)
    ax_hist.grid(axis="x", visible=False)
    ax_hist.set_xlabel("Total Internet activity per square, 1 Nov 2013 – 1 Jan 2014 (log scale)")
    ax_hist.set_ylabel("Number of squares")
    ax_hist.set_title("Distribution of total traffic across the 10,000 squares")

    for square in top + focus:
        ax_hist.axvline(totals[square], color=INK_SECONDARY, linewidth=0.8)
    median = totals.median()
    ax_hist.axvline(median, color=INK_MUTED, linewidth=0.8)

    label_rows = [
        (totals[top].min(), "Top 3: " + ", ".join(map(str, top)) + " ", "right", 0.985),
        (median, " Median", "left", 0.805),
    ]
    label_rows += [(totals[sq], f"Square {sq} ", "right", y) for sq, y in zip(focus, (0.865, 0.925))]
    for value, text, ha, y in label_rows:
        ax_hist.text(value, y, text, transform=ax_hist.get_xaxis_transform(), ha=ha, va="top",
                     color=INK_SECONDARY, fontsize=8.5)

    image = ax_map.imshow(np.log10(totals_grid(totals)), origin="lower", cmap=sequential_cmap(), interpolation="nearest")
    ax_map.grid(False)
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for spine in ax_map.spines.values():
        spine.set_visible(False)
    for square in top + focus:
        row, col = square_to_grid(square)
        ax_map.scatter(col, row, s=70, facecolors="none", edgecolors=INK_PRIMARY, linewidths=1.1)

    offsets = [((10, -10), "left", "top"), ((-10, 10), "right", "bottom")]
    for square, (offset, ha, va) in zip(focus, offsets):
        row, col = square_to_grid(square)
        ax_map.annotate(str(square), (col, row), xytext=offset, textcoords="offset points", ha=ha, va=va,
                        fontsize=8.5, color=INK_PRIMARY, bbox=LABEL_BOX)
    row, col = square_to_grid(top[0])
    ax_map.annotate("Top 3", (col, row), xytext=(10, 10), textcoords="offset points", ha="left", va="bottom",
                    fontsize=8.5, color=INK_PRIMARY, bbox=LABEL_BOX)

    colorbar = fig.colorbar(image, ax=ax_map, fraction=0.046, pad=0.03)
    colorbar.set_label("log10 of total Internet activity", color=INK_SECONDARY)
    colorbar.outline.set_visible(False)
    ax_map.set_title("Spatial pattern on the 100 × 100 grid")

    save(fig, path)


def plot_area_series(
    series_by_title: dict[str, pd.Series], holidays: list[tuple[pd.Timestamp, str]], title: str, path: Path
) -> None:
    n_panels = len(series_by_title)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12.5, 1.95 * n_panels + 0.8), sharex=True)
    index = next(iter(series_by_title.values())).index
    x = _date_numbers(index)

    for ax, (panel_title, series) in zip(axes, series_by_title.items()):
        _shade_weekends(ax, index)
        ax.plot(x, series.to_numpy(), color=SERIES[0], linewidth=1.0)
        ax.set_title(panel_title, fontsize=10.5)
        ax.set_ylim(bottom=0)
        ax.grid(axis="x", visible=False)

    for day, label in holidays:
        noon = _date_number(day + pd.Timedelta(hours=12))
        axes[0].text(noon, 0.97, label, transform=axes[0].get_xaxis_transform(), ha="center", va="top",
                     fontsize=8, color=INK_SECONDARY, bbox=LABEL_BOX)

    axes[-1].set_xlim(x[0], x[-1])
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(tz=index.tz))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b", tz=index.tz))
    fig.supylabel("Internet activity per 10 minutes", color=INK_SECONDARY, fontsize=10)
    fig.suptitle(title, x=0.01, ha="left", color=INK_PRIMARY, fontsize=12.5)
    fig.tight_layout()
    save(fig, path)


def plot_autocorrelation(acf_values: np.ndarray, pacf_values: np.ndarray, band: float, square_id: int, path: Path) -> None:
    fig, (ax_acf, ax_pacf) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})

    lag_days = np.arange(acf_values.size) / DAILY_PERIOD
    ax_acf.axhspan(-band, band, color=SHADE, linewidth=0)
    ax_acf.axhline(0, color=INK_MUTED, linewidth=0.8)
    ax_acf.plot(lag_days, acf_values, color=SERIES[0], linewidth=1.2)
    for lag, name in ((DAILY_PERIOD, "1 day"), (WEEKLY_PERIOD, "1 week")):
        ax_acf.scatter(lag / DAILY_PERIOD, acf_values[lag], s=40, color=SERIES[0], edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax_acf.annotate(f"lag {lag} ({name})\nr = {acf_values[lag]:.2f}", (lag / DAILY_PERIOD, acf_values[lag]),
                        xytext=(8, 6), textcoords="offset points", fontsize=8.5, color=INK_SECONDARY, bbox=LABEL_BOX)
    ax_acf.set_xlim(0, lag_days[-1])
    ax_acf.set_xticks(range(0, int(lag_days[-1]) + 1))
    ax_acf.set_xlabel("Lag (days)")
    ax_acf.set_ylabel("Correlation")
    ax_acf.set_title(f"Autocorrelation (ACF), square {square_id}, lags up to 14 days")

    lags = np.arange(1, pacf_values.size)
    ax_pacf.axhspan(-band, band, color=SHADE, linewidth=0)
    ax_pacf.axhline(0, color=INK_MUTED, linewidth=0.8)
    ax_pacf.vlines(lags, 0, pacf_values[1:], color=SERIES[0], linewidth=1.5)
    ax_pacf.scatter(lags, pacf_values[1:], s=16, color=SERIES[0], zorder=3)
    ax_pacf.set_xlim(0, lags[-1] + 1)
    ax_pacf.set_xticks(range(0, lags[-1] + 1, 6))
    ax_pacf.set_xlabel("Lag (10-minute steps)")
    ax_pacf.set_title("Partial autocorrelation (PACF), first 8 hours")

    fig.text(0.01, -0.02, "Shaded band: approximate 95% interval for white noise (±1.96/√n).",
             color=INK_MUTED, fontsize=8.5, ha="left")
    fig.tight_layout()
    save(fig, path)


def plot_decomposition(
    components: pd.DataFrame,
    daily_anomalies: pd.DataFrame,
    bounds: tuple[float, float],
    threshold: float,
    eval_window: tuple[pd.Timestamp, pd.Timestamp],
    square_id: int,
    path: Path,
) -> None:
    panels = [
        ("observed", "Observed series, log(1 + Internet activity)", 0.7),
        ("trend", "Trend", 1.4),
        ("seasonal_daily", "Daily seasonal component (period 144)", 0.7),
        ("seasonal_weekly", "Weekly seasonal component (period 1008)", 1.0),
        ("residual", "Residual", 0.7),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 11), sharex=True)
    index = components.index
    x = _date_numbers(index)
    window = (_date_number(eval_window[0]), _date_number(eval_window[1]))

    for ax, (column, title, width) in zip(axes, panels):
        ax.axvspan(*window, color=SHADE, linewidth=0, zorder=0)
        ax.plot(x, components[column].to_numpy(), color=SERIES[0], linewidth=width)
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="x", visible=False)

    axes[0].text(sum(window) / 2, 0.97, "Evaluation week\n16–22 Dec", transform=axes[0].get_xaxis_transform(),
                 ha="center", va="top", fontsize=8.5, color=INK_SECONDARY, bbox=LABEL_BOX)

    residual_ax = axes[-1]
    for bound in bounds:
        residual_ax.axhline(bound, color=INK_MUTED, linewidth=0.8)

    flagged = daily_anomalies[daily_anomalies["flagged"]]
    noon = [_date_number(pd.Timestamp(day, tz=index.tz) + pd.Timedelta(hours=12)) for day in flagged.index]
    residual_ax.scatter(noon, flagged["mean_residual"].to_numpy(), s=45, color=CRITICAL, zorder=3,
                        label=f"Unusual day (daily mean residual, |modified z| > {threshold}): {len(flagged)} days")
    ranked = flagged.reindex(flagged["z_score"].abs().sort_values(ascending=False).index).head(4)
    for position, (day, row) in enumerate(ranked.iterrows()):
        day_x = _date_number(pd.Timestamp(day, tz=index.tz) + pd.Timedelta(hours=12))
        # Flagged days can be adjacent, so alternate the label side to keep them apart.
        side = 1 if position % 2 == 0 else -1
        residual_ax.annotate(pd.Timestamp(day).strftime("%d %b"), (day_x, row["mean_residual"]),
                             xytext=(13 * side, 7 if row["mean_residual"] > 0 else -9), textcoords="offset points",
                             ha="left" if side > 0 else "right", fontsize=8.5, color=INK_PRIMARY, bbox=LABEL_BOX)
    residual_ax.legend(loc="upper left", fontsize=8.5)

    residual_ax.set_xlim(x[0], x[-1])
    residual_ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, tz=index.tz))
    residual_ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b", tz=index.tz))
    fig.suptitle(f"MSTL decomposition of log(1 + Internet activity), square {square_id} (ticks on Mondays)", x=0.01, ha="left",
                 color=INK_PRIMARY, fontsize=12.5)
    fig.tight_layout()
    save(fig, path)
