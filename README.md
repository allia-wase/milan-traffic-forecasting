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
| One tuning experiment (validation week) | `python -m scripts.run_experiment --model tcn --params '<json>' --note "<reason>"` | a row in `outputs/experiments/experiment_log.csv`, details in `outputs/experiments/runs/` |
| Final evaluation (test week) | `python -m scripts.evaluate --seeds 42 7 123 --timing-repeats 3` | `outputs/final/`, `outputs/figures/forecast_*.png` |
| Failure analysis | `python -m scripts.failure_analysis` | `outputs/final/failure_analysis.json`, `outputs/figures/failure_*.png` |
| Unit tests | `python -m pytest` | – |

Every step after the first needs the processed dataset. Approximate run times on the hardware
below: exploratory analysis 6 min (robust MSTL is slow on CPU), final evaluation about 70 min,
failure analysis a few seconds (it reuses the saved predictions).

## Repository layout

```
src/
  config.py          paths and dataset constants
  data_loader.py     streaming raw-file reader, dense matrix builder, memory-mapped loader
  memory.py          cross-platform process-memory measurement
  eda.py             distribution, profile, ACF/PACF, MSTL, stationarity and anomaly statistics
  eda_plots.py       exploratory-analysis figures
  plotting.py        shared figure palette and style
  system_info.py     hardware/software description recorded with timings
  forecasting/
    data.py          chronological splits, log-standardisation, sliding windows
    metrics.py       MAE, RMSE, MAPE, MASE
    statistical.py   seasonal-naive baseline, ARIMA with Fourier seasonal terms
    neural.py        LSTM, TCN and the shared early-stopping trainer
    pipeline.py      fits any model on one area and scores it on one split
    diagnostics.py   error-by-day, under-reaction and error-vs-change analysis
    plots.py         forecast and failure-analysis figures
scripts/
  build_dataset.py   builds the 10,000 x 8,928 Internet-activity matrix
  memory_benchmark.py  naive vs optimised loading comparison
  run_eda.py         runs the exploratory analysis
  experiment_robust_stl.py  compares default and robust MSTL on the busiest square
  run_experiment.py  scores one configuration on the validation week and logs it
  evaluate.py        final test-week evaluation, tables, plots and timing
  failure_analysis.py  diagnoses where and why the forecasts fail
configs/
  final_models.json  configurations chosen by the tuning experiments
tests/               unit tests for the pipeline, statistics, models and evaluation
outputs/
  figures/           figures used in the report
  metrics/           JSON results of the data and analysis steps
  experiments/       tuning log (one row per experiment, with the reasoning) and per-run details
  final/             test-week predictions, results tables, timing and failure analysis
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

## Forecasting in brief

**Task.** One-step-ahead forecasting of Internet activity (the next 10 minutes) from an area's own
history, for the three busiest squares (5161, 5059, 5259).

**Splits.** Train 1 Nov – 8 Dec, validation 9–15 Dec, test 16–22 Dec 2013 (Milan time). All tuning
used the validation week of square 5161; the test week was used only by `scripts/evaluate.py`.

**Preprocessing.** `log(1 + x)`, then standardised with training-split statistics. Error metrics are
computed after transforming back to the original scale.

| Model | Final configuration (`configs/final_models.json`) |
|---|---|
| Seasonal naive (baseline) | value at the same time on the previous day |
| ARIMA-Fourier | ARIMA(2,1,1) errors around 16 daily and 8 weekly Fourier pairs (weekly harmonics that repeat a daily one are dropped) |
| LSTM | 1 layer of 64 units, one-day (144-step) input window, Adam 5e-4, early stopping (patience 10) |
| TCN | 6 residual levels of dilated causal convolutions, 16 channels, kernel 3, dropout 0.1, one-day window, Adam 1e-3, early stopping (patience 6) |

The tuning history, including one failed ARIMA fit and its diagnosis, is in
`outputs/experiments/experiment_log.csv`.

**Test-week results** (networks: mean ± standard deviation over seeds 42, 7 and 123; full tables in
`outputs/final/results_tables.md`):

| Square | Model | MAE | MAPE (%) | RMSE |
|---|---|---|---|---|
| 5161 | Seasonal naive | 338.59 | 25.94 | 619.04 |
| 5161 | ARIMA-Fourier | **83.79** | **7.74** | **131.54** |
| 5161 | LSTM | 92.53 ± 1.94 | 8.48 ± 0.04 | 144.31 ± 3.12 |
| 5161 | TCN | 87.74 ± 0.69 | 8.60 ± 0.63 | 132.10 ± 3.52 |
| 5059 | Seasonal naive | 171.74 | 18.02 | 245.87 |
| 5059 | ARIMA-Fourier | **66.56** | **6.43** | **97.25** |
| 5059 | LSTM | 69.58 ± 2.41 | 6.72 ± 0.24 | 102.17 ± 3.76 |
| 5059 | TCN | 73.31 ± 4.75 | 7.50 ± 0.99 | 104.32 ± 4.68 |
| 5259 | Seasonal naive | 470.32 | 71.62 | 861.62 |
| 5259 | ARIMA-Fourier | **62.91** | **6.79** | **92.23** |
| 5259 | LSTM | 66.63 ± 0.33 | 7.24 ± 0.11 | 96.00 ± 1.44 |
| 5259 | TCN | 68.86 ± 9.80 | 7.24 ± 0.39 | 103.54 ± 20.31 |

**Timing** (Intel i5-7200U, 2 cores / 4 threads, 17.0 GB RAM, no GPU, PyTorch on 2 threads; median
and range over 9 measurements across the three squares; method in `outputs/final/timing.json`):

| Model | Training time | Prediction time per forecast |
|---|---|---|
| ARIMA-Fourier | 53 s (31–61 s) | 0.13 ms |
| LSTM | 190 s (112–282 s) | 0.19 ms |
| TCN | 222 s (129–497 s) | 0.14 ms |

**Failure analysis.** The weakest period is the last weekend before Christmas in square 5161
(`outputs/figures/failure_weekend.png`). Across all areas and models, 94–100% of the largest 5% of
errors are under-reactions to a sharp 10-minute change, and error size correlates 0.58–0.72 with
the size of that change (`outputs/figures/failure_error_vs_change.png`).
