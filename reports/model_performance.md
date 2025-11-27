# Model Evaluation Report
============================================================

## Summary
Number of models evaluated: 3

**Best Model:** linear_regression
- RMSE: 209,922.44
- MAE: 196,510.60
- R² Score: 0.4173

## Detailed Results

### Model Comparison Table

| model             |   rmse |    mae |        r2 |    mape |    rmsle |   median_ae |   max_error |   explained_variance |
|:------------------|-------:|-------:|----------:|--------:|---------:|------------:|------------:|---------------------:|
| linear_regression | 209922 | 196511 |  0.41729  | 20.7011 | 0.233353 |      196511 |      270342 |            0.92792   |
| xgboost           | 297902 | 292168 | -0.173493 | 32.5973 | 0.398973 |      292168 |      350335 |            0.95526   |
| random_forest     | 480176 | 399162 | -2.04884  | 37.9265 | 0.594785 |      399162 |      666067 |            0.0580121 |

### linear_regression

- RMSE: 209,922.44
- MAE: 196,510.60
- R2: 0.4173
- MAPE: 20.70%
- RMSLE: 0.2334
- MEDIAN_AE: 196,510.60
- MAX_ERROR: 270,341.58
- EXPLAINED_VARIANCE: 0.9279

### random_forest

- RMSE: 480,175.53
- MAE: 399,162.50
- R2: -2.0488
- MAPE: 37.93%
- RMSLE: 0.5948
- MEDIAN_AE: 399,162.50
- MAX_ERROR: 666,066.67
- EXPLAINED_VARIANCE: 0.0580

### xgboost

- RMSE: 297,901.64
- MAE: 292,167.69
- R2: -0.1735
- MAPE: 32.60%
- RMSLE: 0.3990
- MEDIAN_AE: 292,167.69
- MAX_ERROR: 350,334.94
- EXPLAINED_VARIANCE: 0.9553

