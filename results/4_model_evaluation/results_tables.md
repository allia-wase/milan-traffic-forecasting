### Square 5161

| Model | MAE | MAPE (%) | RMSE | MASE |
|---|---|---|---|---|
| Seasonal naive | 338.59 | 25.94 | 619.04 | 0.975 |
| Persistence (lag 1) | 92.80 | 9.19 | 134.88 | 0.267 |
| ARIMA-Fourier | 76.86 | 7.33 | 117.12 | 0.221 |
| LSTM | 92.53 ± 1.94 | 8.48 ± 0.04 | 144.31 ± 3.12 | 0.266 ± 0.006 |
| TCN | 87.74 ± 0.69 | 8.60 ± 0.63 | 132.10 ± 3.52 | 0.253 ± 0.002 |

### Square 5059

| Model | MAE | MAPE (%) | RMSE | MASE |
|---|---|---|---|---|
| Seasonal naive | 171.74 | 18.02 | 245.87 | 0.635 |
| Persistence (lag 1) | 81.52 | 7.96 | 114.38 | 0.302 |
| ARIMA-Fourier | 63.86 | 6.11 | 93.33 | 0.236 |
| LSTM | 69.58 ± 2.41 | 6.72 ± 0.24 | 102.17 ± 3.76 | 0.257 ± 0.009 |
| TCN | 73.31 ± 4.75 | 7.50 ± 0.99 | 104.32 ± 4.68 | 0.271 ± 0.018 |

### Square 5259

| Model | MAE | MAPE (%) | RMSE | MASE |
|---|---|---|---|---|
| Seasonal naive | 470.32 | 71.62 | 861.62 | 0.898 |
| Persistence (lag 1) | 75.97 | 8.11 | 109.58 | 0.145 |
| ARIMA-Fourier | 63.20 | 6.83 | 92.70 | 0.121 |
| LSTM | 66.63 ± 0.33 | 7.24 ± 0.11 | 96.00 ± 1.44 | 0.127 ± 0.001 |
| TCN | 68.86 ± 9.80 | 7.24 ± 0.39 | 103.54 ± 20.31 | 0.131 ± 0.019 |

Networks: mean ± standard deviation over seeds. ARIMA and the baselines are deterministic.
