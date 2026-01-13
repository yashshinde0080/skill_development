# Model Evaluation Report
============================================================

## Summary
Number of models evaluated: 3

**Best Model:** xgboost
- RMSE: 72,243.57
- MAE: 51,467.70
- R² Score: 0.9003

## Detailed Results

### Model Comparison Table

```
            model          rmse          mae       r2      mape    rmsle    median_ae     max_error  explained_variance
          xgboost  72243.566689 51467.703125 0.900301 16.697049 0.218769 37357.500000 429322.000000            0.900651
    random_forest  77933.666317 55619.384624 0.883977 17.431856 0.220844 39329.285754 433155.113606            0.884532
linear_regression 114761.974713 87375.615034 0.748412 31.036713      NaN 71085.492668 543429.346189            0.748993
```

### linear_regression

- RMSE: 114,761.97
- MAE: 87,375.62
- R2: 0.7484
- MAPE: 31.04%
- MEDIAN_AE: 71,085.49
- MAX_ERROR: 543,429.35
- EXPLAINED_VARIANCE: 0.7490

### random_forest

- RMSE: 77,933.67
- MAE: 55,619.38
- R2: 0.8840
- MAPE: 17.43%
- RMSLE: 0.2208
- MEDIAN_AE: 39,329.29
- MAX_ERROR: 433,155.11
- EXPLAINED_VARIANCE: 0.8845

### xgboost

- RMSE: 72,243.57
- MAE: 51,467.70
- R2: 0.9003
- MAPE: 16.70%
- RMSLE: 0.2188
- MEDIAN_AE: 37,357.50
- MAX_ERROR: 429,322.00
- EXPLAINED_VARIANCE: 0.9007

