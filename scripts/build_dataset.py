"""Build the dense 10,000 x T Internet-activity matrix from the raw daily files.

Usage (from the repository root):
    python -m scripts.build_dataset --raw-dir "path/to/Dataverse"
"""
import argparse
import json
import time
from pathlib import Path

from src import config
from src.data_loader import build_internet_matrix, list_raw_files, save_internet_matrix
from src.memory import current_rss_mb, peak_rss_mb


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the processed Internet-activity matrix.")
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    args = parser.parse_args()

    files = list_raw_files(args.raw_dir)
    baseline_mb = current_rss_mb()

    started = time.perf_counter()
    matrix, timestamps_ms, coverage = build_internet_matrix(files)
    build_seconds = time.perf_counter() - started
    save_internet_matrix(matrix, timestamps_ms)

    report = {
        "files": len(files),
        "first_file": files[0].name,
        "last_file": files[-1].name,
        "raw_size_mb": round(sum(f.stat().st_size for f in files) / 1e6, 1),
        "matrix_shape": list(matrix.shape),
        "matrix_mb": round(matrix.nbytes / 1e6, 1),
        "observed_cell_share": round(coverage, 5),
        "build_seconds": round(build_seconds, 1),
        "baseline_rss_mb": round(baseline_mb, 1),
        "peak_rss_mb": round(peak_rss_mb(), 1),
    }
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (config.METRICS_DIR / "build_dataset.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
