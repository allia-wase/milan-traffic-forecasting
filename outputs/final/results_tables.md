### Square 5161

| Model | MAE | MAPE (%) | RMSE | MASE |
|---|---|---|---|---|
| Seasonal naive | 338.59 | 25.94 | 619.04 | 0.975 |
| ARIMA-Fourier | 83.79 | 7.74 | 131.54 | 0.241 |
| LSTM | 92.53 ± 1.94 | 8.48 ± 0.04 | 144.31 ± 3.12 | 0.266 ± 0.006 |
| TCN | 87.74 ± 0.69 | 8.60 ± 0.63 | 132.10 ± 3.52 | 0.253 ± 0.002 |

### Square 5059

| Model | MAE | MAPE (%) | RMSE | MASE |
|---|---|---|---|---|
| Seasonal naive | 171.74 | 18.02 | 245.87 | 0.635 |
| ARIMA-Fourier | 66.56 | 6.43 | 97.25 | 0.246 |
| LSTM | 69.58 ± 2.41 | 6.72 ± 0.24 | 102.17 ± 3.76 | 0.257 ± 0.009 |
| TCN | 73.31 ± 4.75 | 7.50 ± 0.99 | 104.32 ± 4.68 | 0.271 ± 0.018 |

### Square 5259

| Model | MAE | MAPE (%) | RMSE | MASE |
|---|---|---|---|---|
| Seasonal naive | 470.32 | 71.62 | 861.62 | 0.898 |
| ARIMA-Fourier | 62.91 | 6.79 | 92.23 | 0.120 |
| LSTM | 66.63 ± 0.33 | 7.24 ± 0.11 | 96.00 ± 1.44 | 0.127 ± 0.001 |
| TCN | 68.86 ± 9.80 | 7.24 ± 0.39 | 103.54 ± 20.31 | 0.131 ± 0.019 |

Networks: mean ± standard deviation over seeds. ARIMA and the baseline are deterministic.
