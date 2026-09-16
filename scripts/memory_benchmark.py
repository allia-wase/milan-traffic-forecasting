"""Compare the memory footprint of naive and optimised loading strategies.

Each strategy runs on a sample of days inside a fresh child process, so its peak memory is measured
in isolation. Results are then projected to the full dataset, because the naive strategy cannot be
run on all 62 days on a 16 GB machine.

Usage (from the repository root):
    python -m scripts.memory_benchmark --raw-dir "path/to/Dataverse" --sample-days 3
"""
import argparse
import json
import multiprocessing as mp
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa

from src import config
from src.data_loader import (
    build_internet_matrix,
    list_raw_files,
    load_internet_matrix,
    read_internet_day,
    square_series,
)
from src.memory import MB, current_rss_mb, peak_rss_mb
from src.plotting import BASELINE, GRIDLINE, INK_MUTED, INK_PRIMARY, INK_SECONDARY, SERIES, SURFACE


def naive_pandas(files: list[Path]) -> int:
    df = pd.concat(
        [pd.read_csv(f, sep="\t", header=None, names=config.RAW_COLUMNS) for f in files],
        ignore_index=True,
    )
    return int(df.memory_usage(deep=True).sum())


def pandas_selected_columns(files: list[Path]) -> int:
    df = pd.concat(
        [
            pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 7], dtype={0: "int16", 1: "int64", 7: "float32"})
            for f in files
        ],
        ignore_index=True,
    )
    return int(df.memory_usage(deep=True).sum())


def pyarrow_selected_columns(files: list[Path]) -> int:
    return pa.concat_tables([read_internet_day(f) for f in files]).nbytes


def streaming_dense_matrix(files: list[Path]) -> int:
    matrix, _, _ = build_internet_matrix(files, progress=False)
    return matrix.nbytes


def load_series_in_memory(_: list[Path]) -> int:
    matrix, timestamps = load_internet_matrix(mmap=False)
    return square_series(matrix, timestamps, 5161).nbytes + matrix.nbytes


def load_series_memory_mapped(_: list[Path]) -> int:
    matrix, timestamps = load_internet_matrix(mmap=True)
    return square_series(matrix, timestamps, 5161).nbytes


LOADING_STRATEGIES = [
    ("naive_pandas", "pandas, all 8 columns,\ndefault 64-bit types", naive_pandas),
    ("pandas_selected_columns", "pandas, 3 columns,\ndowncast types", pandas_selected_columns),
    ("pyarrow_selected_columns", "pyarrow, 3 columns,\ndowncast types", pyarrow_selected_columns),
    ("streaming_dense_matrix", "Streaming aggregate\ninto float32 matrix", streaming_dense_matrix),
]

ACCESS_STRATEGIES = [
    ("load_series_in_memory", load_series_in_memory),
    ("load_series_memory_mapped", load_series_memory_mapped),
]


def _measure(fn, files: list[Path]) -> dict:
    baseline = current_rss_mb()
    started = time.perf_counter()
    data_bytes = fn(files)
    seconds = time.perf_counter() - started
    return {
        "data_mb": data_bytes / MB,
        "baseline_rss_mb": baseline,
        "peak_rss_mb": peak_rss_mb(),
        "final_rss_mb": current_rss_mb(),
        "seconds": seconds,
    }


def run_isolated(fn, files: list[Path]) -> dict:
    with mp.get_context("spawn").Pool(processes=1) as pool:
        return pool.apply(_measure, (fn, files))


def project_to_full(name: str, result: dict, sample_days: int, total_days: int) -> dict:
    scale = total_days / sample_days
    working_set = result["peak_rss_mb"] - result["baseline_rss_mb"]
    if name == "streaming_dense_matrix":
        full_matrix_mb = config.N_SQUARES * config.INTERVALS_PER_DAY * total_days * 4 / MB
        return {
            "projected_data_mb": full_matrix_mb,
            "projected_peak_rss_mb": result["baseline_rss_mb"] + working_set - result["data_mb"] + full_matrix_mb,
        }
    return {
        "projected_data_mb": result["data_mb"] * scale,
        "projected_peak_rss_mb": result["baseline_rss_mb"] + working_set * scale,
    }


def plot_results(results: list[dict], total_ram_gb: float, total_days: int, out_path: Path) -> None:
    labels = [r["label"] for r in results][::-1]
    data_gb = [r["projected_data_mb"] / 1000 for r in results][::-1]
    peak_gb = [r["projected_peak_rss_mb"] / 1000 for r in results][::-1]

    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, facecolor=SURFACE)
    panels = [
        (axes[0], data_gb, "Data held in memory (GB)"),
        (axes[1], peak_gb, "Peak process memory (GB)"),
    ]
    x_max = max(max(data_gb), max(peak_gb), total_ram_gb) * 1.18

    for ax, values, title in panels:
        ax.set_facecolor(SURFACE)
        y = np.arange(len(values))
        ax.barh(y, values, height=0.55, color=SERIES[0])
        for yi, value in zip(y, values):
            ax.text(value + x_max * 0.01, yi, f"{value:.2f}", va="center", ha="left", color=INK_PRIMARY, fontsize=9)
        ax.axvline(total_ram_gb, color=INK_MUTED, linewidth=1)
        ax.text(total_ram_gb + x_max * 0.01, -0.5, f"Installed RAM\n{total_ram_gb:.1f} GB",
                color=INK_SECONDARY, fontsize=8.5, va="bottom", ha="left")
        ax.set_xlim(0, x_max)
        ax.set_title(title, loc="left", color=INK_PRIMARY, fontsize=11, pad=10)
        ax.grid(axis="x", color=GRIDLINE, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(BASELINE)
        ax.tick_params(colors=INK_SECONDARY, length=0)

    axes[0].set_yticks(np.arange(len(labels)), labels)
    fig.suptitle(f"Projected memory for all {total_days} days of Internet activity, by loading strategy",
                 x=0.01, ha="left", color=INK_PRIMARY, fontsize=12.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark memory use of loading strategies.")
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--sample-days", type=int, default=3)
    args = parser.parse_args()

    files = list_raw_files(args.raw_dir)
    sample = files[: args.sample_days]
    total_ram_gb = psutil.virtual_memory().total / 1e9

    loading = []
    for name, label, fn in LOADING_STRATEGIES:
        print(f"Running {name} on {len(sample)} day(s)...", flush=True)
        result = {"strategy": name, "label": label, **run_isolated(fn, sample)}
        result.update(project_to_full(name, result, len(sample), len(files)))
        loading.append(result)

    access = []
    if config.MATRIX_PATH.exists():
        for name, fn in ACCESS_STRATEGIES:
            print(f"Running {name}...", flush=True)
            access.append({"strategy": name, **run_isolated(fn, [])})

    report = {
        "system": {
            "cpu": platform.processor(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "total_ram_gb": round(total_ram_gb, 1),
            "os": platform.platform(),
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
        },
        "sample_files": [f.name for f in sample],
        "total_files": len(files),
        "loading_strategies": loading,
        "series_access": access,
    }
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (config.METRICS_DIR / "memory_benchmark.json").write_text(json.dumps(report, indent=2))
    plot_results(loading, total_ram_gb, len(files), config.FIGURES_DIR / "memory_benchmark.png")

    header = f"{'strategy':<28}{'sample data MB':>16}{'sample peak MB':>16}{'seconds':>10}{'full data GB':>14}{'full peak GB':>14}"
    print(header)
    for r in loading:
        print(f"{r['strategy']:<28}{r['data_mb']:>16.1f}{r['peak_rss_mb']:>16.1f}{r['seconds']:>10.1f}"
              f"{r['projected_data_mb'] / 1000:>14.2f}{r['projected_peak_rss_mb'] / 1000:>14.2f}")
    for r in access:
        print(f"{r['strategy']:<28} peak RSS {r['peak_rss_mb']:.1f} MB (baseline {r['baseline_rss_mb']:.1f} MB), {r['seconds']:.2f}s")


if __name__ == "__main__":
    main()
