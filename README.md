# Milan Mobile Internet Traffic Forecasting

An empirical comparison of sequential models for **one-step-ahead forecasting of mobile Internet
activity** on the Telecom Italia *Big Data Challenge* grid of Milan (10,000 squares, 10-minute
intervals, 1 Nov 2013 – 1 Jan 2014).

> Research question: *How do different sequential models compare for one-step-ahead mobile network
> traffic forecasting, and how does their performance vary across geographical areas with different
> traffic characteristics?*

## Setup

Tested with Python 3.14 on Windows 10 (CPU only).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Data

1. Download the *Telecommunications – SMS, Call, Internet – MI* files from Harvard Dataverse
   (doi:10.7910/DVN/EGZHFV).
2. Extract the 62 daily files (`sms-call-internet-mi-YYYY-MM-DD.txt`, ~20.8 GB in total) anywhere
   on disk. They do **not** need to be copied into this repository.
3. Point the code at that folder, either with `--raw-dir` or an environment variable:

```powershell
$env:MILAN_RAW_DIR = "C:\path\to\Dataverse"      # PowerShell
```
```bash
export MILAN_RAW_DIR=/path/to/Dataverse           # bash
```

## Running

All commands run from the repository root.

| Step | Command | Output |
|---|---|---|
| Build the processed dataset | `python -m scripts.build_dataset --raw-dir <dir>` | `data/processed/internet_matrix.npy`, `outputs/metrics/build_dataset.json` |
| Memory benchmark | `python -m scripts.memory_benchmark --raw-dir <dir> --sample-days 3` | `outputs/metrics/memory_benchmark.json`, `outputs/figures/memory_benchmark.png` |
| Exploratory analysis | `python -m scripts.run_eda` | `outputs/figures/eda_*.png`, `outputs/metrics/eda.json` |
| Robust vs default decomposition | `python -m scripts.experiment_robust_stl` | `outputs/metrics/experiment_robust_stl.json` |
| Unit tests | `python -m pytest` | – |

`run_eda` and `experiment_robust_stl` need the processed dataset. Robust MSTL fitting is slow on
CPU; the exploratory analysis takes about 6 minutes on the hardware below.

## Repository layout

```
src/
  config.py          paths and dataset constants
  data_loader.py     streaming raw-file reader, dense matrix builder, memory-mapped loader
  memory.py          cross-platform process-memory measurement
  eda.py             distribution, profile, ACF/PACF, MSTL, stationarity and anomaly statistics
  eda_plots.py       exploratory-analysis figures
  plotting.py        shared figure palette and style
scripts/
  build_dataset.py   builds the 10,000 x 8,928 Internet-activity matrix
  memory_benchmark.py  naive vs optimised loading comparison
  run_eda.py         runs the exploratory analysis
  experiment_robust_stl.py  compares default and robust MSTL on the busiest square
tests/               unit tests for the data pipeline and analysis statistics
outputs/
  figures/           figures used in the report
  metrics/           JSON results of every run
```

## Data handling in brief

The raw files hold one row per *(square, interval, country code)* with eight columns. Only the
Internet column is needed, and a square's traffic is the sum over country codes. The pipeline
therefore reads three columns with compact types through pyarrow, aggregates one day at a time, and
writes the result into a preallocated `float32` matrix saved as `.npy`, which later stages open with
memory mapping.

Measured on an Intel i5-7200U (2 cores / 4 threads) with 17.0 GB (15.8 GiB) RAM. All sizes use
decimal units (1 GB = 10^9 bytes). Projections scale a 3-day sample to 62 days:

| Strategy | Full-dataset data in memory | Full-dataset peak process memory |
|---|---|---|
| pandas, all 8 columns, default types | 18.9 GB (projected) | 38.0 GB (projected) |
| pandas, 3 columns, downcast types | 4.1 GB (projected) | 8.4 GB (projected) |
| pyarrow, 3 columns, downcast types | 4.2 GB (projected) | 7.0 GB (projected) |
| Streaming aggregation into float32 matrix | 0.36 GB | 0.98 GB (full build measured: 0.96 GB) |

Reading one area's series from the saved matrix peaks at 478 MB when the whole file is loaded, versus
121 MB (0.26 s vs 0.02 s) when it is memory-mapped. Peak-memory figures vary by a few percent between
runs; the table reflects the run saved in `outputs/metrics/memory_benchmark.json`.

See `outputs/metrics/` for the full numbers and the method used to obtain them.
