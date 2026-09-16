# Milan Mobile Internet Traffic Forecasting

[![tests](https://github.com/allia-wase/milan-traffic-forecasting/actions/workflows/tests.yml/badge.svg)](https://github.com/allia-wase/milan-traffic-forecasting/actions/workflows/tests.yml)

In this project I try to predict how much mobile Internet traffic a part of Milan will carry in the
next ten minutes, and I compare three kinds of sequential model on that task. The data is the
Telecom Italia *Big Data Challenge* set: Milan is split into 10,000 squares, and activity is
recorded every ten minutes from 1 November 2013 to 1 January 2014.

| Deliverable | Where |
|---|---|
| Report | `reports/report/report.pdf`, built from `report.md` with `python scripts/09_build_report.py` |
| Video | link added on submission |
| Dataset | [Harvard Dataverse: Milan telecommunications activity](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV) |

## Research question

How do different sequential models compare for one-step-ahead mobile network traffic forecasting,
and how does their performance change between areas whose traffic behaves differently?

## Main findings

- ARIMA with Fourier terms for the daily and weekly cycles had the lowest average error in all
  three squares I tested. It also trained about ten times faster than the LSTM and the TCN.
- Diebold-Mariano tests say ARIMA's lead is significant in five of the six comparisons with the
  networks. The one exception is square 5259, where ARIMA and the TCN come out level.
- The TCN was the least stable model. Changing only the random seed moved its error a lot.
- The traffic is mostly a regular daily and weekly rhythm with short-term memory on top, which is
  the structure ARIMA-Fourier assumes. I think that is the main reason it did so well.
- Every model failed in the same way: when traffic jumped or dropped sharply within ten minutes,
  the forecast did not move far enough.

## Repository layout

```
milan-traffic-forecasting/
├── configs/
│   └── final_models.json        settings picked during tuning
├── notebooks/
│   └── milan-forecasting.ipynb  walkthrough of the results (no training needed)
├── reports/
│   ├── report/                  report.md, style.css and the built report.pdf
│   └── video_plan.md            outline for the video
├── scripts/                     the pipeline, run in order
│   ├── 01_build_dataset.py
│   ├── 02_memory_benchmark.py
│   ├── 03_exploratory_analysis.py
│   ├── 03b_locate_squares.py    optional: finds the studied squares on a map
│   ├── 04_decomposition_experiment.py
│   ├── 05_tuning_experiment.py
│   ├── 06_evaluate_models.py
│   ├── 07_failure_analysis.py
│   ├── 08_significance_tests.py
│   └── 09_build_report.py
├── src/milan_forecasting/       the package the scripts import
│   ├── config.py                paths, dataset constants, split dates
│   ├── plotting.py              shared plot style
│   ├── system_info.py           hardware details saved with the timings
│   ├── data/                    streaming loader, memory measurement
│   ├── analysis/                EDA statistics and figures
│   └── forecasting/
│       ├── preprocessing.py     splits, log + standardisation, sliding windows
│       ├── metrics.py           MAE, RMSE, MAPE, MASE
│       ├── significance.py      Diebold-Mariano test
│       ├── models/              naive and ARIMA-Fourier (statistical.py), LSTM and TCN (neural.py)
│       ├── pipeline.py          fits and scores any model the same way
│       ├── evaluation.py        tuning grids, repeated runs, result tables, timing
│       ├── diagnostics.py       failure-analysis statistics
│       └── figures.py           forecast and failure plots
├── tests/                       33 unit tests (also run on GitHub Actions)
├── results/                     all outputs, one folder per report section
│   ├── 1_data_handling/
│   ├── 2_exploratory_analysis/
│   ├── 3_hyperparameter_tuning/
│   └── 4_model_evaluation/
│       ├── forecasts/           the 9 forecast plots and test-week predictions
│       ├── first_run_arima_211/ test results from before the ARIMA grid search
│       └── failure_analysis/
├── pyproject.toml
└── requirements.txt
```

## Setup

I developed and tested this only with Python 3.14 on Windows 10, on a laptop with no GPU. The
pinned packages should also work on Python 3.11 to 3.13, but I haven't tried them.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest
```

Use the editable install (`-e`). The code finds `data/`, `configs/` and `results/` relative to
the repository, so it needs to run from the checkout. The tests take about 15 seconds.

For the raw data, open the dataset link above (*Telecommunications - SMS, Call, Internet - MI*)
and download the 62 daily files, `sms-call-internet-mi-2013-11-01.txt` to
`sms-call-internet-mi-2014-01-01.txt`. They come zipped, so unzip them into one folder. They take
about 20.8 GB, so keep them outside the repository. Then either pass that folder to the scripts
with `--raw-dir` or set `MILAN_RAW_DIR` once.

The optional step 3b also needs `milano-grid.geojson` from the
[Milano Grid dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QJWLFU).
Dataverse asks you to fill in a short guestbook form before either download.

If you just want to look at the results, open `notebooks/milan-forecasting.ipynb`. It only reads
the saved outputs, so it runs in seconds without the raw data.

## Running the pipeline

Run the scripts in number order. Each one writes to its own folder under `results/`. The times
are what each step took on my laptop.

| Step | Command | Time |
|---|---|---|
| 1 | `python scripts/01_build_dataset.py --raw-dir <dir>` | 1.5 min |
| 2 | `python scripts/02_memory_benchmark.py --raw-dir <dir>` | 2 min |
| 3 | `python scripts/03_exploratory_analysis.py` | 6 min |
| 3b | `python scripts/03b_locate_squares.py --grid-file <geojson>` | 1 s |
| 4 | `python scripts/04_decomposition_experiment.py` | 5 min |
| 5 | `python scripts/05_tuning_experiment.py --model <m> --params '<json>' --note "<why>"` | up to 1 h per run |
| 5 (grid) | same, plus `--grid '{"p": [0, 1, 2], "q": [0, 1, 2]}'` | ~15 min for ARIMA |
| 6 | `python scripts/06_evaluate_models.py` (add `--models arima` to re-run one model) | 70 min |
| 7 | `python scripts/07_failure_analysis.py` | seconds |
| 8 | `python scripts/08_significance_tests.py` | seconds |
| 9 | `python scripts/09_build_report.py` | seconds |

Notes on the steps:

- **Raw files:** only steps 1 and 2 read them. Everything after step 1 uses the matrix it saves
  in `data/processed/`.
- **Step 5:** only ever looks at the validation week.
- **Steps 7 and 8:** reuse the predictions saved by step 6.
- **Step 9:** needs `pip install -e ".[report]"` and Chrome or Edge, which prints the HTML to PDF,
  so LaTeX isn't needed.

## How I did it

### Handling the data

Each daily file has about 4.8 million rows, one per square, ten-minute interval and caller
country, with eight columns. All 62 files together are bigger than my laptop's memory, so
`pd.read_csv` on everything was never going to work.

For forecasting I only need Internet activity per square and interval. So the loader:

- reads just three columns (`square_id`, `time_interval`, `internet`) with small number types;
- sums over country codes one day at a time;
- writes the totals into a fixed 10,000 × 8,928 `float32` matrix, which takes 357 MB.

Later steps memory-map the matrix, so only the rows they touch get loaded. Reading one square this
way peaks at 121 MB of process memory, against 478 MB when I load the whole matrix.

I benchmarked four loading strategies on three days of data and scaled the results up to two
months:

| Strategy | Data in memory | Peak process memory |
|---|---|---|
| pandas, all 8 columns, default types | 18.9 GB | 38.0 GB |
| pandas, 3 columns, compact types | 4.1 GB | 8.4 GB |
| pyarrow, 3 columns, compact types | 4.2 GB | 7.0 GB |
| Streaming into a `float32` matrix (used) | 0.36 GB | 0.98 GB |

The real build of the full matrix peaked at 0.96 GB, close to the estimate.

### Exploring the traffic

- **Spread across the city:** traffic is very uneven. The busiest 10% of squares carry almost
  half of all traffic, and the busiest squares are close together in the centre.
- **The three busiest squares differ:** 5161 gets busier at weekends, 5259 empties out at weekends
  like an office area, and 5059 is somewhere in between. Square 4556 peaks late in the evening.
- **Memory:** for the busiest square, the value ten minutes ago is a very good guide to the value
  now. The same time yesterday and the same time last week are also strong guides.
- **Log scale:** busy days swing much more than quiet ones, so the analysis and all the models
  use log traffic, which turns those proportional swings into roughly constant ones.
- **Decomposition:** a strong daily cycle, a weaker weekly one, and a drop from mid-December into
  Christmas. Four days stand out: Christmas Day and Boxing Day are far below normal, and 11 and
  18 December are above it.
- **Robust fitting:** `scripts/04_decomposition_experiment.py` shows why I used a robust
  decomposition. The default one absorbed part of the Christmas dip into the seasonal pattern.

### Choosing and tuning the models

I picked three models from different families, so the comparison isn't just versions of one
architecture:

- ARIMA with Fourier seasonal terms (statistical);
- an LSTM (recurrent network);
- a TCN (convolutional network).

As a baseline, every model has to beat "same time yesterday".

The data is split by date:

- **Training:** 1 November to 8 December.
- **Validation (tuning):** 9 to 15 December.
- **Test:** 16 to 22 December.

I tuned on the validation week of the busiest square only, changing one setting per run and
writing down the reason before running it. I stopped tuning a model once a change gained less than
1% on validation error. The 15 manual runs are in `results/3_hyperparameter_tuning/experiment_log.csv`,
followed by 9 grid-search runs over the ARIMA orders p and q (0 to 2 each).

What I noticed during tuning:

- **Smaller networks worked better than bigger ones.**
- **A week of input history didn't help the networks** compared with a day, and it made them far
  slower.
- **One ARIMA run failed completely.** A week is exactly 7 days, so the 7th weekly Fourier wave is
  the same as the 1st daily wave. The two columns were identical and the optimiser couldn't
  start. Dropping the duplicate waves fixed it. Both the failed run and the fixed one are in the
  log.
- **The grid search disagreed with my manual choice.** AIC ranked my hand-picked ARIMA(2,1,1)
  near the top, but ARIMA(2,1,0) had a 3.5% lower validation error. My own rule says to keep
  changes worth more than 1%, so the final model uses (2,1,0).

**Timing caveat:** I made that switch *after* I had already run the test week once with (2,1,1).
The choice itself used only validation data, but I had seen the test results, so I report both
(see below).

Final settings (from `configs/final_models.json`):

| Model | Settings |
|---|---|
| Seasonal naive | value at the same time on the previous day |
| ARIMA-Fourier | ARIMA(2,1,0) errors around 16 daily and 8 weekly Fourier pairs |
| LSTM | one layer of 64 units, one day of input |
| TCN | six dilated convolution levels, 16 channels, one day of input |

### Test results

I used the test week only after tuning was finished. I trained each network with three random
seeds and report the mean ± standard deviation. The full tables are in
`results/4_model_evaluation/results_tables.md`.

| Square | Model | MAE | MAPE (%) | RMSE |
|---|---|---|---|---|
| 5161 | Seasonal naive | 338.59 | 25.94 | 619.04 |
| | ARIMA-Fourier | 76.86 | 7.33 | 117.12 |
| | LSTM | 92.53 ± 1.94 | 8.48 ± 0.04 | 144.31 ± 3.12 |
| | TCN | 87.74 ± 0.69 | 8.60 ± 0.63 | 132.10 ± 3.52 |
| 5059 | Seasonal naive | 171.74 | 18.02 | 245.87 |
| | ARIMA-Fourier | 63.86 | 6.11 | 93.33 |
| | LSTM | 69.58 ± 2.41 | 6.72 ± 0.24 | 102.17 ± 3.76 |
| | TCN | 73.31 ± 4.75 | 7.50 ± 0.99 | 104.32 ± 4.68 |
| 5259 | Seasonal naive | 470.32 | 71.62 | 861.62 |
| | ARIMA-Fourier | 63.20 | 6.83 | 92.70 |
| | LSTM | 66.63 ± 0.33 | 7.24 ± 0.11 | 96.00 ± 1.44 |
| | TCN | 68.86 ± 9.80 | 7.24 ± 0.39 | 103.54 ± 20.31 |

**Timing.** All timings come from the same laptop: Intel Core i5-7200U (2 cores, 4 threads),
15.8 GiB RAM (saved as 17.0 GB in the result files), no GPU. The numbers are medians over nine runs
across the three squares, and `results/4_model_evaluation/timing.json` describes how I measured
them.

| Model | Training time | Time per forecast |
|---|---|---|
| ARIMA-Fourier | 19 s (12 to 25 s) | 0.12 ms |
| LSTM | 190 s (112 to 282 s) | 0.19 ms |
| TCN | 222 s (129 to 497 s) | 0.14 ms |

All three models predict in well under a millisecond, so in practice what matters is training
time.

**Significance.** To check whether the gaps in the error table are more than chance, I ran
Diebold-Mariano tests (`results/4_model_evaluation/significance.md`). They compare ARIMA with the
seed-42 networks over the 1,008 forecasts in each square. Using absolute errors:

- ARIMA is significantly better than both networks in 5161 and 5059.
- In 5259, ARIMA is significantly better than the LSTM but level with the TCN.

**First run with ARIMA(2,1,1).** Before the grid search, the test week was run with ARIMA(2,1,1):

- MAE was 83.79 / 66.56 / 62.91 on squares 5161 / 5059 / 5259, and training took a median of 53 s.
- With that model, ARIMA's lead over the TCN wasn't significant in any square.
- The (2,1,0) model is clearly better in 5161 and 5059, and about the same in 5259.

The full first-run results are in `results/4_model_evaluation/first_run_arima_211/`.

### Where the models struggle

- **Worst period:** for the busiest square it was the last weekend before Christmas, when the
  afternoon peaks were about 50% higher than on weekdays.
- **Common failure:** across all squares and models, 94 to 100% of the largest errors came from
  sharp ten-minute moves that the forecast didn't follow far enough. The bigger the jump, the
  bigger the error.

The plots are in `results/4_model_evaluation/failure_analysis/`.

## Reproducibility

The networks use fixed seeds, so rerunning with the same seed and data gives the same numbers.
ARIMA and the baseline have no randomness.

Timings are the exception, because a laptop isn't a stable benchmark machine. One identical TCN
run took 300 s once and 493 s the next time, which is why I report medians over repeated runs.

## Use of AI

I used an AI coding assistant while building this project. The report's AI declaration explains
where and how.

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
7. F. X. Diebold and R. S. Mariano, "Comparing predictive accuracy," *J. Business & Economic
   Statistics*, vol. 13, no. 3, pp. 253–263, 1995.

## License

The code is under the MIT License (see `LICENSE`). The Telecom Italia data isn't included here and
stays under the terms of its Dataverse record.
