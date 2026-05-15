from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

from src.utils import MODELS_DIR, setup_logging

logger = setup_logging("predict")


@dataclass
class PredictionResult:
    predicted_price: float
    confidence_score: float
    similar_properties: list[dict[str, Any]]


class HousePricePredictor:
    """Prediction service for the trained house price model."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path else self._latest_model_path()
        bundle = joblib.load(self.model_path)

        self.best_model_name = bundle["best_model_name"]
        self.model = bundle["best_model"]
        self.all_models = bundle["all_models"]
        self.preprocessor = bundle["preprocessor"]
        self.training_data = bundle["training_data"]

    def _latest_model_path(self) -> Path:
        candidates = sorted(MODELS_DIR.glob("best_model_*.joblib"))
        if not candidates:
            raise FileNotFoundError("No trained model found in models/. Run training first.")
        return candidates[-1]

    def _estimate_confidence(self, features: pd.DataFrame, prediction: float) -> float:
        preds = []
        x = self.preprocessor.transform(features)
        for model in self.all_models.values():
            try:
                preds.append(float(model.predict(x)[0]))
            except Exception:
                continue

        if len(preds) < 2:
            return 0.65

        std = float(np.std(preds))
        scale = max(abs(prediction), 1.0)
        uncertainty_ratio = min(std / scale, 1.0)
        return float(np.clip(1 - uncertainty_ratio, 0.0, 1.0))

    def _find_similar_properties(self, features: pd.DataFrame, k: int = 5) -> list[dict[str, Any]]:
        frame = self.training_data.copy()
        query = features.copy()

        compare_cols = [
            c
            for c in ["area_sqft", "bedrooms", "bathrooms", "parking", "age_of_property", "floors"]
            if c in frame.columns and c in query.columns
        ]

        if not compare_cols:
            return []

        frame_numeric = frame[compare_cols].fillna(frame[compare_cols].median(numeric_only=True))
        query_numeric = query[compare_cols].fillna(frame_numeric.median(numeric_only=True))

        distances = euclidean_distances(frame_numeric.values, query_numeric.values).ravel()
        nearest_idx = np.argsort(distances)[:k]

        display_cols = [
            col
            for col in [
                "price",
                "area_sqft",
                "bedrooms",
                "bathrooms",
                "location",
                "parking",
                "furnishing_status",
                "age_of_property",
            ]
            if col in frame.columns
        ]

        suggestions = frame.iloc[nearest_idx][display_cols].copy()
        suggestions["distance"] = distances[nearest_idx]

        output = suggestions.to_dict(orient="records")
        for item in output:
            if "price" in item:
                item["price"] = float(item["price"])
            item["distance"] = float(item["distance"])
        return output

    def predict(self, payload: dict[str, Any]) -> PredictionResult:
        features = pd.DataFrame([payload])
        transformed = self.preprocessor.transform(features)
        prediction = float(self.model.predict(transformed)[0])

        confidence = self._estimate_confidence(features, prediction)
        similar = self._find_similar_properties(features)

        return PredictionResult(
            predicted_price=prediction,
            confidence_score=confidence,
            similar_properties=similar,
        )
