"""Figures comparing forecasts with the observed series."""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from milan_forecasting.plotting import INK_MUTED, INK_PRIMARY, INK_SECONDARY, SERIES, SHADE, SURFACE, save

MODEL_LABELS = {"naive": "Seasonal naive", "arima": "ARIMA-Fourier", "lstm": "LSTM", "tcn": "TCN"}
MODEL_COLORS = {"naive": INK_MUTED, "arima": SERIES[0], "lstm": SERIES[1], "tcn": SERIES[2]}
TRAINED_MODELS = ("arima", "lstm", "tcn")
LABEL_BOX = {"boxstyle": "round,pad=0.3", "facecolor": SURFACE, "edgecolor": "none", "alpha": 0.92}


def _date_numbers(index: pd.DatetimeIndex) -> np.ndarray:
    return mdates.date2num(index.to_pydatetime())


def plot_forecast(
    timestamps: pd.DatetimeIndex,
    actual: np.ndarray,
    predicted: np.ndarray,
    model: str,
    square: int,
    metrics: dict[str, float],
    note: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 4.4))
    x = mdates.date2num(timestamps.to_pydatetime())

    # Actual drawn wider and underneath so the prediction stays readable where the two coincide.
    ax.plot(x, actual, color=INK_SECONDARY, linewidth=1.8, label="Actual", zorder=2)
    ax.plot(x, predicted, color=MODEL_COLORS[model], linewidth=1.0, label=f"{MODEL_LABELS[model]} prediction", zorder=3)

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mdates.DayLocator(tz=timestamps.tz))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b", tz=timestamps.tz))
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Internet activity per 10 minutes")
    ax.legend(loc="upper right", ncol=2, fontsize=9)

    ax.set_title(f"{MODEL_LABELS[model]}: one-step-ahead forecast vs actual, square {square}, 16–22 Dec 2013",
                 pad=24, fontsize=11.5)
    summary = f"MAE {metrics['mae']:.1f}  ·  MAPE {metrics['mape']:.1f}%  ·  RMSE {metrics['rmse']:.1f}"
    ax.text(0, 1.02, f"{summary}{note}", transform=ax.transAxes, color=INK_PRIMARY, fontsize=9.5, va="bottom")

    fig.tight_layout()
    save(fig, path)


def plot_failure_window(
    predictions: pd.DataFrame,
    window: tuple[pd.Timestamp, pd.Timestamp],
    detail: tuple[pd.Timestamp, pd.Timestamp],
    miss_time: pd.Timestamp,
    title: str,
    path: Path,
) -> None:
    fig, (ax_all, ax_detail) = plt.subplots(2, 1, figsize=(12.5, 8.2), gridspec_kw={"height_ratios": [1.15, 1]})
    tz = predictions.index.tz

    overview = predictions.loc[window[0]:window[1]]
    x = _date_numbers(overview.index)
    ax_all.axvspan(*mdates.date2num([detail[0].to_pydatetime(), detail[1].to_pydatetime()]),
                   color=SHADE, linewidth=0, zorder=0)
    ax_all.plot(x, overview["actual"], color=INK_SECONDARY, linewidth=1.8, label="Actual", zorder=2)
    for model in TRAINED_MODELS:
        ax_all.plot(x, overview[model], color=MODEL_COLORS[model], linewidth=0.9, label=MODEL_LABELS[model], zorder=3)
    ax_all.set_xlim(x[0], x[-1])
    ax_all.set_ylim(bottom=0)
    ax_all.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18], tz=tz))
    ax_all.xaxis.set_major_formatter(mdates.DateFormatter("%a %d\n%H:%M", tz=tz))
    ax_all.grid(axis="x", visible=False)
    ax_all.set_title(title, fontsize=11.5)
    ax_all.text(mdates.date2num(detail[1].to_pydatetime()), 0.97, "shaded: enlarged below",
                transform=ax_all.get_xaxis_transform(), color=INK_SECONDARY, fontsize=8.5, va="top",
                bbox=LABEL_BOX)
    ax_all.legend(loc="upper left", ncol=2, fontsize=9)

    zoom = predictions.loc[detail[0]:detail[1]]
    xz = _date_numbers(zoom.index)
    ax_detail.plot(xz, zoom["actual"], color=INK_SECONDARY, linewidth=1.8, marker="o", markersize=5, label="Actual", zorder=2)
    for model in TRAINED_MODELS:
        ax_detail.plot(xz, zoom[model], color=MODEL_COLORS[model], linewidth=1.0, marker="o", markersize=3.5,
                       label=MODEL_LABELS[model], zorder=3)
    miss = predictions.loc[miss_time]
    previous = predictions["actual"].shift(1).loc[miss_time]
    ax_detail.annotate(
        f"{miss_time:%a %H:%M}: actual moved {miss['actual'] - previous:+,.0f} in 10 minutes\n"
        f"actual {miss['actual']:,.0f}  ·  ARIMA {miss['arima']:,.0f}  ·  LSTM {miss['lstm']:,.0f}  ·  TCN {miss['tcn']:,.0f}",
        (mdates.date2num(miss_time.to_pydatetime()), miss["actual"]),
        xytext=(0, -46), textcoords="offset points", ha="center", va="top", fontsize=9, color=INK_PRIMARY,
        bbox=LABEL_BOX, arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.8},
    )
    ax_detail.set_xlim(xz[0], xz[-1])
    ax_detail.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30], tz=tz))
    ax_detail.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    ax_detail.grid(axis="x", visible=False)
    ax_detail.set_title("Detail, one marker per 10-minute forecast: the forecasts follow a smooth climb "
                        "and miss the sudden dips", fontsize=10.5)
    fig.supylabel("Internet activity per 10 minutes", color=INK_SECONDARY, fontsize=10)
    fig.tight_layout()
    save(fig, path)


def plot_error_by_day(daily_by_square: dict[int, pd.DataFrame], note: str, path: Path) -> None:
    fig, axes = plt.subplots(1, len(daily_by_square), figsize=(13.5, 4.4))
    width = 0.27
    for ax, (square, daily) in zip(axes, daily_by_square.items()):
        positions = np.arange(len(daily))
        for offset, model in zip((-width, 0, width), TRAINED_MODELS):
            ax.bar(positions + offset, daily[model], width, color=MODEL_COLORS[model],
                   edgecolor=SURFACE, linewidth=1.0, label=MODEL_LABELS[model])
        ax.set_xticks(positions, [day.replace(" ", "\n") for day in daily.index])
        ax.grid(axis="x", visible=False)
        ax.set_title(f"Square {square}", fontsize=10.5)
    axes[0].set_ylabel("Mean absolute error per forecast")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, fontsize=9, bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Test-week error by day", x=0.01, ha="left", color=INK_PRIMARY, fontsize=12.5)
    fig.text(0.01, -0.02, note, color=INK_MUTED, fontsize=8.5, ha="left")
    fig.tight_layout()
    save(fig, path)


def plot_error_by_move(curves_by_square: dict[int, dict[str, pd.Series]], path: Path) -> None:
    fig, axes = plt.subplots(1, len(curves_by_square), figsize=(13.5, 4.4))
    for ax, (square, curves) in zip(axes, curves_by_square.items()):
        for model in TRAINED_MODELS:
            curve = curves[model]
            ax.plot(curve.index, curve.to_numpy(), color=MODEL_COLORS[model], linewidth=2, marker="o",
                    markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5, label=MODEL_LABELS[model])
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
        ax.set_title(f"Square {square}", fontsize=10.5)
        ax.set_xlabel("Size of the actual 10-minute change")
    axes[0].set_ylabel("Mean absolute error per forecast")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, fontsize=9, bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Forecast error grows with the size of the actual 10-minute change",
                 x=0.01, ha="left", color=INK_PRIMARY, fontsize=12.5)
    fig.text(0.01, -0.02, "Each point: one tenth of the test-week forecasts, grouped by the size of the actual "
             "change (x = group median). Networks: seed 42.", color=INK_MUTED, fontsize=8.5, ha="left")
    fig.tight_layout()
    save(fig, path)
