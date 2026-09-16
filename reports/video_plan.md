# Video plan (7–10 minutes)

These are talking points, not a script. Say them in your own words, and show the real repo and
figures while you talk. The rubric rewards explanations that are *specific to this
implementation*, so name the actual numbers, files and decisions.

| Time | Section | Show on screen | Points to cover |
|---|---|---|---|
| 0:00–0:45 | Problem | Report title page | Why 10-minute traffic forecasts matter to an operator. The research question. The three squares and the test week (16–22 Dec). |
| 0:45–2:00 | Handling the data | `results/1_data_handling/memory_benchmark.png`, `src/milan_forecasting/data/loader.py` | 20.8 GB of files against 7 GB of free RAM. Plain pandas would peak around 38 GB. Keep 3 columns, sum one day at a time into a float32 grid of 357 MB, measured peak 0.96 GB. Memory-mapping: 121 MB against 478 MB. One trade-off (the dropped columns would force a rebuild for multivariate models). |
| 2:00–3:30 | What the data looks like | `traffic_distribution.png`, `first_two_weeks.png`, `autocorrelation.png`, `decomposition.png` | Skewed, centre-heavy city. The three squares behave differently (5161 busier at weekends, 5259 like an office district). ACF 0.99 at lag 1, 0.88 at one day, 0.84 at one week. Why the log transform (daily range correlates 0.92 with daily mean). Christmas and 18 Dec flagged as anomalies. |
| 3:30–5:00 | Models and why | Table 2 in the report, `models/statistical.py`, `models/neural.py` | ARIMA-Fourier: several long seasonalities with few parameters. LSTM: recurrent, nonlinear. TCN: dilated causal convolutions, receptive field 253. Why these three are different families. The seasonal-naive baseline. |
| 5:00–6:15 | Tuning | `experiment_log.csv`, Tables 3–4 | One change per run, validation week only, stop below a 1% gain. **Technical decision to explain:** the ARIMA run that failed (weekly harmonic 7 = daily harmonic 1, because 1008 = 7 × 144), and how dropping the duplicate harmonics fixed it. Also: the week-long LSTM window cost 28 times the time for a 3% gain, so it was dropped. |
| 6:15–7:45 | Results | Tables 5–9, one forecast plot | ARIMA(2,1,0) had the lowest error in all three squares (e.g. 76.9 against 87.7 and 92.5 on 5161) and trains about 10 times faster. Its lead is significant in 5 of 6 comparisons (Table 8), but it is level with the TCN on 5259. Be open that the order changed from (2,1,1) after the grid search, and that with (2,1,1) the TCN comparison was not significant. The TCN's spread across seeds on 5259 (±9.8). Validation vs test by MAPE: ARIMA 7.53 → 7.33, while both networks get worse (LSTM 8.06 → 8.48, TCN 7.58 → 8.60). Why the structure of the data favours ARIMA. |
| 7:45–9:00 | **Limitation / failure case** | `poorest_period.png`, `error_vs_change_size.png` | Last weekend before Christmas on 5161: Sat 13:20 actual 3,381, ARIMA 4,017. 94–100% of the largest errors are under-reactions to sharp moves. The weekly Fourier shape learned in November underestimates the pre-Christmas weekends. Univariate models, one test week. |
| 9:00–9:45 | Wrap-up | Conclusion section | Answer to the research question. What you would do next (spatial inputs, holiday features, more seeds). |

Tips: record your face or voice throughout, keep the code on screen when you talk about it, and
rehearse once with a timer. The viva can ask about any line above, so make sure you could redraw
the memory table and explain the collinearity bug without notes.
