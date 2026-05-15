from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class RegressionMetrics:
    mae: float
    mse: float
    rmse: float
    r2: float
    adjusted_r2: float


def adjusted_r2_score(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    n_samples = len(y_true)
    r2 = r2_score(y_true, y_pred)
    denominator = max(1, n_samples - n_features - 1)
    return 1 - ((1 - r2) * (n_samples - 1) / denominator)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int,
) -> RegressionMetrics:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    adj_r2 = adjusted_r2_score(y_true, y_pred, n_features=n_features)

    return RegressionMetrics(
        mae=float(mae),
        mse=float(mse),
        rmse=rmse,
        r2=float(r2),
        adjusted_r2=float(adj_r2),
    )
