# Milan Mobile Internet Traffic Forecasting

This project forecasts mobile Internet traffic in Milan ten minutes ahead using the Telecom
Italia Big Data Challenge dataset. The dataset covers 10,000 squares at ten-minute intervals
from 1 November 2013 to 1 January 2014.

| Deliverable | Location |
|---|---|
| Report | PDF submitted separately |
| Video | Link added on submission |
| Dataset | [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV) |

## Research question

How do different sequential models compare for one-step-ahead mobile network traffic forecasting,
and how does their performance change between areas whose traffic behaves differently?

## Findings

- ARIMA-Fourier has the lowest mean error in all three evaluated squares and trains about ten times faster than the neural models.
- Lag-1 persistence is a stronger one-step baseline than same-time-yesterday. On square 5161 its MAE is `92.80`, close to the LSTM mean of `92.53`.
- The TCN is the least stable model across random seeds. All models struggle with sharp ten-minute traffic changes.
- The traffic has strong daily and weekly structure, which matches the ARIMA-Fourier design.
- The three evaluated squares are adjacent, forming one 3 x 3 grid block about 470 to 525 m
	across. The geographic comparison is therefore within one central district, not across Milan.
- The saved significance results use seed-42 networks and, after Holm-Bonferroni correction, show ARIMA significantly ahead in three of six comparisons under absolute loss.

## Repository structure

```
scripts/        pipeline stages 01-08, run in order (see below)
src/             package code (data loading, analysis, forecasting, plotting)
configs/         final_models.json and other saved configuration
notebooks/       report.ipynb - reads results/ and renders every figure and table
results/         committed outputs from every stage, incl. results/4_model_evaluation/
tests/           unit tests, run with `python -m pytest`
```

## Viewing the results without rerunning anything

`notebooks/report.ipynb` reads the saved JSON/CSV/PNG outputs under `results/` and reproduces
every figure and table in the report. It runs in a few seconds and does not retrain or
reprocess anything; the one cell that touches the processed dataset matrix skips itself if that
matrix has not been built locally. This is the fastest way to check the submitted results
without rerunning `scripts/01`-`08`.

## Setup

Use Python 3.14 on Windows, or a compatible Python version supported by the pinned dependencies
(developed and timed against `torch==2.14.0+cpu` and `statsmodels==0.15.0`; see
`pyproject.toml` for the full pinned set):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m pytest
```

The raw dataset is about 20.8 GB. Keep it outside the repository and either pass its directory
to the data-building scripts with `--raw-dir` or set `MILAN_RAW_DIR`.

The split is chronological: training through 8 December, validation from 9 to 15 December, and test from 16 to 22 December.
EDA describes the full 62-day record, while model selection uses the validation week.
ARIMA was evaluated on the test week twice because the final order changed from `(2,1,1)` to `(2,1,0)`; both runs are archived.
Neural models are trained with three fixed seeds — `42`, `7`, `123` (the default for
`scripts/06_evaluate_models.py --seeds`); ARIMA and both baselines are deterministic, so their
repeated runs only measure timing, not variability.

Run the scripts in order from the repository root:

```bash
python scripts/01_build_dataset.py --raw-dir <raw-data-directory>
python scripts/02_memory_benchmark.py --raw-dir <raw-data-directory>
python scripts/03_exploratory_analysis.py
python scripts/04_decomposition_experiment.py
python scripts/05_tuning_experiment.py --model <model> --params '<json>' --note "<reason>"
python scripts/06_evaluate_models.py
python scripts/07_failure_analysis.py
python scripts/08_significance_tests.py
```

The optional geography step is:

```bash
python scripts/03b_locate_squares.py --grid-file <geojson>
```

Use `python scripts/06_evaluate_models.py --models <model>` for a partial evaluation. Saved
outputs are under `results/`, especially `results/4_model_evaluation/` for metrics, forecasts,
significance tests, timing, and failure analysis.

**Runtime.** On the two-core Intel i5-7200U laptop used for the reported timings,
`scripts/06_evaluate_models.py` takes roughly an hour end to end across all three squares.
Training is dominated by the LSTM and TCN across three seeds each (about 55 minutes combined);
ARIMA's three deterministic timing repeats and both baselines finish in under three minutes.
Earlier stages (`01`-`04`) are much faster: the dataset build itself takes under two minutes
once the raw files are available locally.

## Models and evaluation

The comparison includes seasonal-naive and lag-1 persistence baselines, ARIMA-Fourier, LSTM,
and TCN. The split is chronological: training through 8 December, validation from 9 to 15
December, and test from 16 to 22 December. Neural models use three fixed seeds; ARIMA and both
baselines are deterministic.

The processed matrix uses only Internet activity and is stored as a memory-mapped `float32`
array. The repository does not include the raw dataset or processed data.

## Reproducibility

Results and experiment logs are committed under `results/`. Timings are hardware-dependent and
were measured on a two-core Intel i5 laptop without a GPU. The tests run in about 15 seconds.

The Telecom Italia data remains under the terms of its Dataverse record.
