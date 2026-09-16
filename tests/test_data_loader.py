import numpy as np
import pytest

from src import config
from src.data_loader import build_internet_matrix, load_internet_matrix, save_internet_matrix, square_series

NOV_1_MIDNIGHT_ROME_MS = 1383260400000
DAY_MS = 86_400_000
TEN_MIN_MS = 600_000


def _write_day(path, rows):
    lines = ["\t".join(str(v) for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def two_days(tmp_path):
    t0 = NOV_1_MIDNIGHT_ROME_MS
    day1 = tmp_path / "sms-call-internet-mi-2013-11-01.txt"
    day2 = tmp_path / "sms-call-internet-mi-2013-11-02.txt"
    _write_day(day1, [
        (1, t0, 39, "", "", "", "", 2.0),
        (1, t0, 0, "", "", "", "", 1.5),
        (1, t0 + TEN_MIN_MS, 39, 0.1, "", "", "", ""),
        (10000, t0, 39, "", "", "", "", 4.0),
    ])
    _write_day(day2, [(2, t0 + DAY_MS, 39, "", "", "", "", 7.0)])
    return [day1, day2]


def test_matrix_sums_country_codes_and_zero_fills(two_days):
    matrix, timestamps_ms, coverage = build_internet_matrix(two_days, progress=False)

    assert matrix.shape == (config.N_SQUARES, 2 * config.INTERVALS_PER_DAY)
    assert matrix.dtype == np.float32
    assert matrix[0, 0] == pytest.approx(3.5)
    assert matrix[0, 1] == 0
    assert matrix[9999, 0] == pytest.approx(4.0)
    assert matrix[1, config.INTERVALS_PER_DAY] == pytest.approx(7.0)
    assert matrix.sum() == pytest.approx(14.5)
    assert timestamps_ms[0] == NOV_1_MIDNIGHT_ROME_MS
    assert timestamps_ms[config.INTERVALS_PER_DAY] == NOV_1_MIDNIGHT_ROME_MS + DAY_MS
    assert coverage == pytest.approx(4 / matrix.size)


def test_records_outside_grid_are_rejected(tmp_path):
    day = tmp_path / "sms-call-internet-mi-2013-11-01.txt"
    _write_day(day, [(1, NOV_1_MIDNIGHT_ROME_MS - TEN_MIN_MS, 39, "", "", "", "", 1.0)])

    with pytest.raises(ValueError, match="outside"):
        build_internet_matrix([day], progress=False)


def test_memory_mapped_round_trip(two_days, tmp_path):
    matrix, timestamps_ms, _ = build_internet_matrix(two_days, progress=False)
    matrix_path, ts_path = tmp_path / "m.npy", tmp_path / "t.npy"
    save_internet_matrix(matrix, timestamps_ms, matrix_path, ts_path)

    loaded, timestamps = load_internet_matrix(mmap=True, matrix_path=matrix_path, timestamps_path=ts_path)
    series = square_series(loaded, timestamps, 1)

    assert isinstance(loaded, np.memmap)
    assert str(timestamps.tz) == config.TIMEZONE
    assert timestamps[0].hour == 0 and timestamps[0].day == 1
    assert series.iloc[0] == pytest.approx(3.5)
