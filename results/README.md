# Reference results

These values are transcribed from the completed coursework and are included so the GitHub repository can be reviewed without rerunning the full notebook.

| Model | RMSE (BRL) | MAE (BRL) | R² |
|---|---:|---:|---:|
| Baseline multiple linear regression | 11.492 | 5.166 | 0.562 |
| Feature-engineered OLS | 10.601 | 4.755 | 0.627 |
| Tuned Ridge regression | 10.629 | 4.740 | 0.625 |

Ridge is the preferred final model in the coursework because the selection also considers median absolute error, group-aware cross-validation, regularisation and coefficient stability.
