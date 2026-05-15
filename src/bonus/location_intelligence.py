from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LocationSignal:
    crime_index: float
    school_index: float
    transit_index: float
    livability_score: float


class LocationIntelligenceEngine:
    """Computes location-aware scores for pricing enhancement."""

    def score_location(self, location: str) -> LocationSignal:
        seed = sum(ord(c) for c in location) % 100
        school = min(1.0, 0.4 + seed / 200)
        transit = min(1.0, 0.3 + seed / 250)
        crime = max(0.0, 0.7 - seed / 250)
        livability = (school * 0.45) + (transit * 0.35) + ((1 - crime) * 0.20)

        return LocationSignal(
            crime_index=round(crime, 4),
            school_index=round(school, 4),
            transit_index=round(transit, 4),
            livability_score=round(livability, 4),
        )
