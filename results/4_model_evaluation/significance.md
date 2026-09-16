Diebold-Mariano tests on the test week (1,008 one-step forecasts per square; networks: seed 42).
Negative statistic: model A has the lower loss. Newey-West variance, HLN-adjusted, t(n-1) p-values.
* p < 0.05, ** p < 0.01, *** p < 0.001.

| Square | A vs B | DM, absolute loss | p | DM, squared loss | p |
|---|---|---:|---:|---:|---:|
| 5161 | ARIMA-Fourier vs LSTM | -4.76*** | 2.2e-06 | -4.23*** | 2.55e-05 |
| 5161 | ARIMA-Fourier vs TCN | -4.35*** | 1.49e-05 | -3.81*** | 0.00015 |
| 5161 | LSTM vs TCN | 1.99* | 0.047 | 2.70** | 0.00714 |
| 5059 | ARIMA-Fourier vs LSTM | -2.12* | 0.0341 | -2.04* | 0.0413 |
| 5059 | ARIMA-Fourier vs TCN | -3.76*** | 0.000177 | -2.96** | 0.00317 |
| 5059 | LSTM vs TCN | -1.43 | 0.152 | -0.63 | 0.529 |
| 5259 | ARIMA-Fourier vs LSTM | -2.28* | 0.0227 | -1.80 | 0.072 |
| 5259 | ARIMA-Fourier vs TCN | 0.53 | 0.598 | 0.78 | 0.433 |
| 5259 | LSTM vs TCN | 3.04** | 0.00243 | 3.11** | 0.00192 |
