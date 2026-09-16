# Milan Mobile Internet Traffic Forecasting

This project predicts how much mobile Internet traffic each part of Milan will carry in the next ten
minutes, and compares how well three different kinds of sequential model do the job. It uses the
Telecom Italia *Big Data Challenge* data: the city split into 10,000 squares, with activity recorded
every ten minutes from 1 November 2013 to 1 January 2014.

| Deliverable | Where |
|---|---|
| Report | `reports/report/report.pdf` (built with `make report`) |
| Video | link added on submission |
| Dataset | [Harvard Dataverse - Milan telecommunications activity](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV) |

## The question

How do different sequential models compare for one-step-ahead mobile network traffic forecasting,
and how does their performance change between areas whose traffic behaves differently?

## What came out of it

A statistical model, ARIMA with Fourier terms for the daily and weekly cycles, was the most accurate
in all three areas tested, and it trained roughly four times faster than the two neural networks.
On the validation week it was practically tied with the TCN, but on the unseen test week it pulled
clearly ahead, and the TCN's results changed a lot depending on the random seed. The traffic turned
out to be mostly regular daily and weekly rhythm plus short-term memory, which is exactly what the
ARIMA model is built to describe.

All the models struggled in the same situation: when traffic jumped or dropped sharply within ten
minutes. Almost every one of the largest errors was a forecast that fell short of a sudden move.

## Repository structure

```
milan-traffic-forecasting/
├── configs/
│   └── final_models.json        settings chosen by the tuning experiments
├── notebooks/
│   └── milan-forecasting.ipynb  walkthrough of every section using the saved results
├── reports/
│   └── report/report.pdf        the written report (built with make report)
├── scripts/                     the pipeline, run in numbered order
│   ├── 01_build_dataset.py
│   ├── 02_memory_benchmark.py
│   ├── 03_exploratory_analysis.py
│   ├── 04_decomposition_experiment.py
│   ├── 05_tuning_experiment.py
│   ├── 06_evaluate_models.py
│   └── 07_failure_analysis.py
├── src/milan_forecasting/       the installable package
│   ├── config.py                paths, dataset constants, study design
│   ├── plotting.py              shared figure style
│   ├── system_info.py           hardware description for timing reports
│   ├── data/                    streaming loader and memory measurement
│   ├── analysis/                exploratory statistics and figures
│   └── forecasting/
│       ├── preprocessing.py     splits, log-standardisation, sliding windows
│       ├── metrics.py           MAE, RMSE, MAPE, MASE
│       ├── models/              baseline and ARIMA-Fourier (statistical.py), LSTM and TCN (neural.py)
│       ├── pipeline.py          fits and scores any model on identical data
│       ├── evaluation.py        repeated runs, results tables, timing summaries
│       ├── diagnostics.py       failure-analysis statistics
│       └── figures.py           forecast and failure-analysis figures
├── tests/                       27 unit tests
├── results/                     every output, grouped by report section
│   ├── 1_data_handling/
│   ├── 2_exploratory_analysis/
│   ├── 3_hyperparameter_tuning/
│   └── 4_model_evaluation/
│       ├── forecasts/           9 forecast plots and the test-week predictions
│       └── failure_analysis/
├── pyproject.toml
└── requirements.txt
```

## Getting started

You need Python 3.11 or newer; the project was developed with Python 3.14 on a Windows laptop
without a GPU. Create a virtual environment and install the project together with its development
tools:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest
```

Install it in editable mode (`-e`) as shown, because the code finds the `data`, `configs` and
`results` folders relative to the repository. The tests take about fifteen seconds.

Next, download the *Telecommunications - SMS, Call, Internet - MI* files from the dataset link above
and unzip the 62 daily files somewhere on your machine. Together they are about 20.8 GB, so there is
no need to copy them into the repository. Point the scripts at that folder with `--raw-dir`, or set
the `MILAN_RAW_DIR` environment variable once.

If you only want to see the results, open `notebooks/milan-forecasting.ipynb`. It reads the saved
outputs, so it runs in a few seconds and needs neither the raw data nor any training.

## Running the pipeline

The scripts are numbered in the order they are meant to run. Each one writes its outputs into the
matching folder under `results/`, and the times below are what they took on the development laptop.

| Step | Command | Time |
|---|---|---|
| 1 | `python scripts/01_build_dataset.py --raw-dir <dir>` | 1.5 min |
| 2 | `python scripts/02_memory_benchmark.py --raw-dir <dir>` | 2 min |
| 3 | `python scripts/03_exploratory_analysis.py` | 6 min |
| 4 | `python scripts/04_decomposition_experiment.py` | 5 min |
| 5 | `python scripts/05_tuning_experiment.py --model <m> --params '<json>' --note "<why>"` | up to an hour per run |
| 6 | `python scripts/06_evaluate_models.py` | 70 min |
| 7 | `python scripts/07_failure_analysis.py` | a few seconds |

Only the first two steps read the raw files. Everything after that works from the processed matrix
that step 1 saves in `data/processed/`. Step 5 runs one tuning experiment at a time and only ever
looks at the validation week, and step 7 reuses the predictions saved by step 6.

## How the study was done

### Handling the data

Each daily file has about 4.8 million rows, one for every combination of square, ten-minute interval
and caller country, spread over eight columns. Altogether the files are larger than the laptop's
memory, so loading them the usual way was not possible.

Forecasting only needs the Internet activity per square and interval. The loader therefore reads
just those three columns with compact number types, adds up the rows for each square one day at a
time, and writes the totals into a fixed 10,000 × 8,928 grid of 32-bit numbers. That grid takes
357 MB, and later steps open it memory-mapped so that only the rows they need are loaded. Reading a
single square this way peaks at 121 MB of process memory, compared with 478 MB when the whole grid
is loaded.

The memory benchmark tried four strategies on three days of data and scaled the results up to the
full two months:

| Strategy | Data in memory | Peak process memory |
|---|---|---|
| pandas, all 8 columns, default types | 18.9 GB | 38.0 GB |
| pandas, 3 columns, compact types | 4.1 GB | 8.4 GB |
| pyarrow, 3 columns, compact types | 4.2 GB | 7.0 GB |
| Streaming into a 32-bit grid (used) | 0.36 GB | 0.98 GB |

Building the full grid really did peak at 0.96 GB, which matches the estimate.

### Exploring the traffic

Traffic is very unevenly spread. The busiest tenth of the city carries almost half of all traffic,
and the busiest squares sit together in the centre. The three busiest squares also behave
differently: 5161 gets busier at weekends, 5259 empties out like an office district, and 5059 sits in
between. Square 4556 peaks late in the evening.

For the busiest square, the value ten minutes ago is an excellent guide to the value now, and the
same time yesterday and the same time last week are strong guides too. Busy days swing much harder
than quiet ones, so the analysis and all the models work on the logarithm of traffic, which turns
those proportional swings into steady ones. A decomposition of the series shows a strong daily
cycle, a weaker weekly one, a drop in traffic from mid-December into Christmas, and four unusual
days: Christmas Day and Boxing Day far below normal, and 11 and 18 December above it. A side
experiment (`scripts/04_decomposition_experiment.py`) showed that a robust decomposition was needed,
because the standard one quietly absorbed part of the Christmas dip.

### Choosing and tuning the models

The three models come from different families, so the comparison is not just between versions of
one architecture: ARIMA with Fourier seasonal terms, a recurrent LSTM network and a convolutional
TCN. A same-time-yesterday forecast serves as the baseline that every model has to beat.

The data is split in time. The models learn from 1 November to 8 December, are tuned on 9 to 15
December, and are finally tested on 16 to 22 December. Tuning used the validation week of the
busiest square only, and changed one setting at a time with the reason written down before each run.
The fifteen runs are in `results/3_hyperparameter_tuning/experiment_log.csv`. Tuning a model stopped
once a change improved its validation error by less than one percent.

A few things stood out. Smaller networks did better than bigger ones, and giving the networks a
week of history instead of a day did not help, even though it cost far more time. One ARIMA run
failed outright: a week is exactly seven days, so one of the weekly Fourier waves was identical to
one of the daily waves, and the model could not tell them apart. Removing the duplicate waves fixed
it, and both the failed run and the fixed one are kept in the log.

The final settings are in `configs/final_models.json`:

| Model | Final settings |
|---|---|
| Seasonal naive | the value at the same time on the previous day |
| ARIMA-Fourier | ARIMA(2,1,1) errors around 16 daily and 8 weekly Fourier pairs |
| LSTM | one layer of 64 units, one day of input history |
| TCN | six dilated convolution levels, 16 channels, one day of input history |

### Testing the models

The test week was used only once, after tuning had finished. Each network was trained with three
different random seeds, so its results are shown as a mean and a spread. The full tables are in
`results/4_model_evaluation/results_tables.md`.

| Square | Model | MAE | MAPE (%) | RMSE |
|---|---|---|---|---|
| 5161 | Seasonal naive | 338.59 | 25.94 | 619.04 |
| | ARIMA-Fourier | 83.79 | 7.74 | 131.54 |
| | LSTM | 92.53 ± 1.94 | 8.48 ± 0.04 | 144.31 ± 3.12 |
| | TCN | 87.74 ± 0.69 | 8.60 ± 0.63 | 132.10 ± 3.52 |
| 5059 | Seasonal naive | 171.74 | 18.02 | 245.87 |
| | ARIMA-Fourier | 66.56 | 6.43 | 97.25 |
| | LSTM | 69.58 ± 2.41 | 6.72 ± 0.24 | 102.17 ± 3.76 |
| | TCN | 73.31 ± 4.75 | 7.50 ± 0.99 | 104.32 ± 4.68 |
| 5259 | Seasonal naive | 470.32 | 71.62 | 861.62 |
| | ARIMA-Fourier | 62.91 | 6.79 | 92.23 |
| | LSTM | 66.63 ± 0.33 | 7.24 ± 0.11 | 96.00 ± 1.44 |
| | TCN | 68.86 ± 9.80 | 7.24 ± 0.39 | 103.54 ± 20.31 |

Timings were measured on an Intel Core i5-7200U with 2 cores, 4 threads and 17 GB of memory, with no
GPU. The figures are medians over nine runs across the three squares, and
`results/4_model_evaluation/timing.json` records exactly how they were taken.

| Model | Training time | Time per forecast |
|---|---|---|
| ARIMA-Fourier | 53 s (31 to 61 s) | 0.13 ms |
| LSTM | 190 s (112 to 282 s) | 0.19 ms |
| TCN | 222 s (129 to 497 s) | 0.14 ms |

Every model makes a forecast in well under a millisecond, so the real difference between them is how
long they take to train.

### Where the models struggle

The worst stretch for the busiest square was the last weekend before Christmas, when its afternoon
peaks were about half as high again as on weekdays. Looking across all three squares and all three models, 94
to 100 percent of the largest errors happened when traffic moved sharply within ten minutes and the
forecast did not move far enough. The bigger the sudden change, the bigger the error. The figures
for this are in `results/4_model_evaluation/failure_analysis/`.

## Reproducing the results

The neural networks use fixed random seeds, so running them again with the same seed and data gives
the same numbers. ARIMA and the baseline have no randomness at all. Timings are the exception: on a
laptop the same run can take noticeably longer from one attempt to the next (one identical TCN run
took 300 seconds once and 493 the next time), which is why the timing table reports medians over
repeated runs.

## References

1. G. Barlacchi *et al.*, "A multi-source dataset of urban life in the city of Milan and the Province
   of Trentino," *Scientific Data*, vol. 2, 150055, 2015, doi:10.1038/sdata.2015.55.
2. Telecom Italia, "Telecommunications - SMS, Call, Internet - MI," Harvard Dataverse, 2015,
   doi:10.7910/DVN/EGZHFV.
3. A. Azari, P. Papapetrou, S. Denic and G. Peters, "Cellular traffic prediction and classification:
   A comparative evaluation of LSTM and ARIMA," in *Discovery Science*, 2019, arXiv:1906.00939.
4. S. Bai, J. Z. Kolter and V. Koltun, "An empirical evaluation of generic convolutional and
   recurrent networks for sequence modeling," 2018, arXiv:1803.01271.
5. A. Zeng, M. Chen, L. Zhang and Q. Xu, "Are Transformers effective for time series forecasting?"
   in *Proc. AAAI Conf. Artificial Intelligence*, 2023, arXiv:2205.13504.
6. K. Bandara, R. J. Hyndman and C. Bergmeir, "MSTL: A seasonal-trend decomposition algorithm for
   time series with multiple seasonal patterns," 2021, arXiv:2107.13462.
