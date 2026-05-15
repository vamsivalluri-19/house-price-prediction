from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class PriceTrendForecaster:
    """Forecasts future average prices using linear trend approximation."""

    def forecast(self, historical: pd.Series, periods: int = 6) -> pd.Series:
        if historical.empty:
            raise ValueError("Historical series is empty")

        x = np.arange(len(historical)).reshape(-1, 1)
        y = historical.values

        model = LinearRegression()
        model.fit(x, y)

        future_x = np.arange(len(historical), len(historical) + periods).reshape(-1, 1)
        preds = model.predict(future_x)

        start = historical.index[-1] + 1 if isinstance(historical.index[-1], (int, np.integer)) else len(historical)
        idx = range(start, start + periods)
        return pd.Series(preds, index=idx, name="forecast_price")
