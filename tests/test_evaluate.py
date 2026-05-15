from __future__ import annotations

import numpy as np

from src.evaluate import compute_regression_metrics


def test_metrics_values() -> None:
    y_true = np.array([100, 200, 300, 400])
    y_pred = np.array([110, 190, 310, 390])

    metrics = compute_regression_metrics(y_true, y_pred, n_features=3)

    assert metrics.mae >= 0
    assert metrics.mse >= 0
    assert metrics.rmse >= 0
    assert -1 <= metrics.r2 <= 1
