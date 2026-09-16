"""Project-wide paths and dataset constants."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = Path(os.environ.get("MILAN_RAW_DIR", ROOT / "data" / "raw"))
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"

MATRIX_PATH = PROCESSED_DIR / "internet_matrix.npy"
TIMESTAMPS_PATH = PROCESSED_DIR / "timestamps_ms.npy"

FILE_GLOB = "sms-call-internet-mi-*.txt"
RAW_COLUMNS = [
    "square_id",
    "time_interval",
    "country_code",
    "sms_in",
    "sms_out",
    "call_in",
    "call_out",
    "internet",
]

TIMEZONE = "Europe/Rome"
N_SQUARES = 10_000
INTERVAL_MS = 10 * 60 * 1000
INTERVALS_PER_DAY = 144

FOCUS_SQUARES = (4159, 4556)
N_EVAL_SQUARES = 3
VAL_START = "2013-12-09"
EVAL_START = "2013-12-16"
EVAL_END = "2013-12-22"

EXPERIMENTS_DIR = OUTPUT_DIR / "experiments"
EXPERIMENT_LOG = EXPERIMENTS_DIR / "experiment_log.csv"

ITALIAN_HOLIDAYS = {
    "2013-11-01": "All Saints' Day",
    "2013-12-07": "Sant'Ambrogio (Milan patron saint)",
    "2013-12-08": "Immaculate Conception",
    "2013-12-25": "Christmas Day",
    "2013-12-26": "St Stephen's Day",
    "2014-01-01": "New Year's Day",
}
