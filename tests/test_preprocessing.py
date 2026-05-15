from __future__ import annotations

import pandas as pd

from src.preprocessing import HousingPreprocessor


def test_preprocessor_fit_transform() -> None:
    df = pd.DataFrame(
        {
            "price": [200000, 250000, 300000, 350000],
            "area_sqft": [1200, 1500, 1800, 2100],
            "bedrooms": [2, 3, 3, 4],
            "bathrooms": [1, 2, 2, 3],
            "location": ["A", "B", "B", "C"],
            "parking": [1, 1, 2, 2],
            "furnishing_status": ["unfurnished", "semi-furnished", "furnished", "furnished"],
            "age_of_property": [20, 15, 10, 8],
            "nearby_facilities": ["basic", "good", "excellent", "excellent"],
            "floors": [1, 1, 2, 2],
            "property_type": ["House", "House", "Condo", "House"],
        }
    )

    preprocessor = HousingPreprocessor()
    x, y, engineered = preprocessor.fit_transform(df)

    assert x.shape[0] == len(df)
    assert len(y) == len(df)
    assert "luxury_score" in engineered.columns
    assert "location_popularity_index" in engineered.columns
