# First test-week run with the hand-picked ARIMA(2,1,1)

These files are the complete test-week results from before the ARIMA grid search. At that point
`configs/final_models.json` used ARIMA(2,1,1). The grid search, which used only the validation
week, then chose ARIMA(2,1,0), and `scripts/06_evaluate_models.py --models arima` re-ran ARIMA with
that order. The results one folder up are the final ones. This folder is kept so both results can
be checked. The LSTM and TCN rows are identical in both runs.

Diebold-Mariano tests (absolute loss) for this first ARIMA against the seed-42 networks:

| Square | Comparison | DM | p |
|---|---|---:|---:|
| 5161 | ARIMA(2,1,1) vs LSTM | -2.62 | 0.00902 |
| 5161 | ARIMA(2,1,1) vs TCN | -1.19 | 0.236 |
| 5059 | ARIMA(2,1,1) vs LSTM | -0.54 | 0.59 |
| 5059 | ARIMA(2,1,1) vs TCN | -1.91 | 0.057 |
| 5259 | ARIMA(2,1,1) vs LSTM | -2.49 | 0.0131 |
| 5259 | ARIMA(2,1,1) vs TCN | 0.35 | 0.729 |
