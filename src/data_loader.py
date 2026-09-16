"""Streaming loader that turns the raw Milan telecom text files into a dense Internet-activity matrix.

Rows of the matrix are grid squares (square_id - 1) and columns are 10-minute intervals, so a
single area's time series is one row and the whole two-month dataset fits in ~357 MB of float32.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv
from tqdm import tqdm

from src import config

INTERNET_SCHEMA = {
    "square_id": pa.int16(),
    "time_interval": pa.int64(),
    "internet": pa.float32(),
}


def list_raw_files(raw_dir: Path | str = config.RAW_DIR) -> list[Path]:
    files = sorted(Path(raw_dir).glob(config.FILE_GLOB))
    if not files:
        raise FileNotFoundError(
            f"No '{config.FILE_GLOB}' files found in {raw_dir}. "
            "Set the MILAN_RAW_DIR environment variable or pass --raw-dir."
        )
    return files


def file_date(path: Path) -> pd.Timestamp:
    return pd.Timestamp(path.stem.removeprefix("sms-call-internet-mi-"), tz=config.TIMEZONE)


def read_internet_day(path: Path) -> pa.Table:
    """Read only the three columns needed for Internet forecasting, with compact types."""
    return pv.read_csv(
        path,
        read_options=pv.ReadOptions(column_names=config.RAW_COLUMNS),
        parse_options=pv.ParseOptions(delimiter="\t"),
        convert_options=pv.ConvertOptions(
            include_columns=list(INTERNET_SCHEMA),
            column_types=INTERNET_SCHEMA,
        ),
    )


def aggregate_day(table: pa.Table) -> pa.Table:
    # Raw rows are split by the caller's country code; a square's traffic is the sum over all codes.
    return table.group_by(["square_id", "time_interval"]).aggregate([("internet", "sum")])


def build_internet_matrix(
    files: list[Path], progress: bool = True
) -> tuple[np.ndarray, np.ndarray, float]:
    """Stream the daily files into a (N_SQUARES, n_intervals) float32 matrix.

    Only one day's table is held in memory at a time. Intervals with no Internet record are left
    at 0, which matches the dataset convention of omitting rows with no activity.
    Returns the matrix, the interval start times (epoch ms) and the share of grid cells observed.
    """
    start = file_date(files[0])
    end = file_date(files[-1]) + pd.Timedelta(days=1)
    start_ms = start.value // 1_000_000
    n_intervals = int((end - start) / pd.Timedelta(milliseconds=config.INTERVAL_MS))

    matrix = np.zeros((config.N_SQUARES, n_intervals), dtype=np.float32)
    observed_cells = 0

    for path in tqdm(files, desc="Aggregating days", unit="file", disable=not progress):
        day = aggregate_day(read_internet_day(path))
        squares = day["square_id"].to_numpy().astype(np.int64) - 1
        slots = (day["time_interval"].to_numpy() - start_ms) // config.INTERVAL_MS

        if squares.min() < 0 or squares.max() >= config.N_SQUARES or slots.min() < 0 or slots.max() >= n_intervals:
            raise ValueError(f"{path.name} contains records outside the expected square/time grid")

        matrix[squares, slots] = pc.fill_null(day["internet_sum"], 0).to_numpy()
        observed_cells += day.num_rows

    timestamps_ms = start_ms + np.arange(n_intervals, dtype=np.int64) * config.INTERVAL_MS
    return matrix, timestamps_ms, observed_cells / matrix.size


def save_internet_matrix(
    matrix: np.ndarray,
    timestamps_ms: np.ndarray,
    matrix_path: Path = config.MATRIX_PATH,
    timestamps_path: Path = config.TIMESTAMPS_PATH,
) -> None:
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(matrix_path, matrix)
    np.save(timestamps_path, timestamps_ms)


def load_internet_matrix(
    mmap: bool = True,
    matrix_path: Path = config.MATRIX_PATH,
    timestamps_path: Path = config.TIMESTAMPS_PATH,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Load the processed matrix; memory-mapped by default so only the rows used are paged in."""
    matrix = np.load(matrix_path, mmap_mode="r" if mmap else None)
    timestamps = pd.to_datetime(np.load(timestamps_path), unit="ms", utc=True).tz_convert(config.TIMEZONE)
    return matrix, timestamps


def square_series(matrix: np.ndarray, timestamps: pd.DatetimeIndex, square_id: int) -> pd.Series:
    return pd.Series(
        np.asarray(matrix[square_id - 1], dtype=np.float32),
        index=timestamps,
        name=f"square_{square_id}",
    )
