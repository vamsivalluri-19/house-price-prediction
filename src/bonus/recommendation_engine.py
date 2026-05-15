from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PriceRecommendation:
    recommended_listing_price: float
    lower_bound: float
    upper_bound: float
    rationale: str


class PriceRecommendationEngine:
    """Suggest listing strategy using model prediction and confidence."""

    def recommend(self, predicted_price: float, confidence: float) -> PriceRecommendation:
        margin = max(0.03, 1 - confidence) * predicted_price
        target = predicted_price * (1 + (confidence - 0.5) * 0.04)

        return PriceRecommendation(
            recommended_listing_price=round(target, 2),
            lower_bound=round(predicted_price - margin, 2),
            upper_bound=round(predicted_price + margin, 2),
            rationale="Band widens when confidence drops to support safer pricing decisions.",
        )
