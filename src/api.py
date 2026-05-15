from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import HousePricePredictor
from src.utils import setup_logging

logger = setup_logging("api")
app = FastAPI(title="House Price Prediction API", version="1.0.0")


class PredictionRequest(BaseModel):
    area_sqft: float = Field(gt=100, description="Property area in sq ft")
    bedrooms: int = Field(ge=0, le=20)
    bathrooms: int = Field(ge=0, le=20)
    location: str
    parking: int = Field(ge=0, le=10)
    furnishing_status: Literal["unfurnished", "semi-furnished", "furnished"]
    age_of_property: int = Field(ge=0, le=200)
    nearby_facilities: Literal["basic", "good", "excellent"] = "good"
    floors: int = Field(ge=1, le=10, default=1)
    property_type: str = "House"


class PredictionResponse(BaseModel):
    predicted_price: float
    confidence_score: float
    similar_properties: list[dict[str, Any]]


_predictor: HousePricePredictor | None = None


def get_predictor() -> HousePricePredictor:
    global _predictor
    if _predictor is None:
        _predictor = HousePricePredictor()
    return _predictor


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_price(request: PredictionRequest) -> PredictionResponse:
    try:
        predictor = get_predictor()
        result = predictor.predict(request.model_dump())
        return PredictionResponse(
            predicted_price=result.predicted_price,
            confidence_score=result.confidence_score,
            similar_properties=result.similar_properties,
        )
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
