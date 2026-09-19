Diebold-Mariano tests on the test week (1,008 one-step forecasts per square; networks: seed 42).
Negative statistic: model A has the lower loss. Newey-West variance, HLN-adjusted, t(n-1) p-values.
The nine pair p-values within each loss function are adjusted with Holm-Bonferroni.
Stars follow the adjusted p-value: * p < 0.05, ** p < 0.01, *** p < 0.001.

| Square | A vs B | DM, absolute loss | p | Holm p | DM, squared loss | p | Holm p |
|---|---|---:|---:|---:|---:|---:|---:|
| 5161 | ARIMA-Fourier vs LSTM | -4.76*** | 2.2e-06 | 1.98e-05 | -4.23*** | 2.55e-05 | 0.000229 |
| 5161 | ARIMA-Fourier vs TCN | -4.35*** | 1.49e-05 | 0.000119 | -3.81** | 0.00015 | 0.0012 |
| 5161 | LSTM vs TCN | 1.99 | 0.047 | 0.141 | 2.70* | 0.00714 | 0.0357 |
| 5059 | ARIMA-Fourier vs LSTM | -2.12 | 0.0341 | 0.136 | -2.04 | 0.0413 | 0.165 |
| 5059 | ARIMA-Fourier vs TCN | -3.76** | 0.000177 | 0.00124 | -2.96* | 0.00317 | 0.019 |
| 5059 | LSTM vs TCN | -1.43 | 0.152 | 0.304 | -0.63 | 0.529 | 0.866 |
| 5259 | ARIMA-Fourier vs LSTM | -2.28 | 0.0227 | 0.113 | -1.80 | 0.072 | 0.216 |
| 5259 | ARIMA-Fourier vs TCN | 0.53 | 0.598 | 0.598 | 0.78 | 0.433 | 0.866 |
| 5259 | LSTM vs TCN | 3.04* | 0.00243 | 0.0146 | 3.11* | 0.00192 | 0.0135 |
