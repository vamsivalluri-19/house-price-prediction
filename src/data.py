from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from src.utils import DATA_DIR, setup_logging

logger = setup_logging("data")


def _derive_lifestyle_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain columns expected by the prediction system if absent."""
    frame = df.copy()

    if "parking" not in frame.columns:
        garage_source = "GarageCars" if "GarageCars" in frame.columns else None
        frame["parking"] = frame[garage_source].fillna(0).astype(int) if garage_source else 1

    if "furnishing_status" not in frame.columns:
        quality_col = "OverallQual" if "OverallQual" in frame.columns else None
        if quality_col:
            q = frame[quality_col].fillna(frame[quality_col].median())
            frame["furnishing_status"] = pd.cut(
                q,
                bins=[-np.inf, 4, 7, np.inf],
                labels=["unfurnished", "semi-furnished", "furnished"],
            ).astype(str)
        else:
            frame["furnishing_status"] = "semi-furnished"

    if "age_of_property" not in frame.columns:
        year_built_col = "YearBuilt" if "YearBuilt" in frame.columns else None
        if year_built_col:
            frame["age_of_property"] = 2026 - frame[year_built_col].fillna(2000).astype(int)
        else:
            frame["age_of_property"] = 10

    if "nearby_facilities" not in frame.columns:
        school_col = "OverallQual" if "OverallQual" in frame.columns else None
        if school_col:
            facility_index = frame[school_col].fillna(frame[school_col].median())
            frame["nearby_facilities"] = pd.cut(
                facility_index,
                bins=[-np.inf, 4, 7, np.inf],
                labels=["basic", "good", "excellent"],
            ).astype(str)
        else:
            frame["nearby_facilities"] = "good"

    if "property_type" not in frame.columns:
        if "BldgType" in frame.columns:
            frame["property_type"] = frame["BldgType"].fillna("House").astype(str)
        else:
            frame["property_type"] = "House"

    if "location" not in frame.columns:
        neighborhood_col = "Neighborhood" if "Neighborhood" in frame.columns else None
        frame["location"] = frame[neighborhood_col].fillna("Unknown") if neighborhood_col else "Unknown"

    return frame


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw dataset column names to project standard naming."""
    rename_map = {
        "SalePrice": "price",
        "GrLivArea": "area_sqft",
        "BedroomAbvGr": "bedrooms",
        "FullBath": "bathrooms",
        "location": "location",
        "Parking": "parking",
        "parking": "parking",
        "furnishingstatus": "furnishing_status",
        "furnishing_status": "furnishing_status",
        "age": "age_of_property",
        "Age": "age_of_property",
        "age_of_property": "age_of_property",
        "nearby_facilities": "nearby_facilities",
        "facilities": "nearby_facilities",
        "Floors": "floors",
        "floors": "floors",
        "stories": "floors",
        "property_type": "property_type",
        "BldgType": "property_type",
    }

    frame = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}).copy()
    frame = _derive_lifestyle_columns(frame)

    if "floors" not in frame.columns:
        if "HouseStyle" in frame.columns:
            frame["floors"] = frame["HouseStyle"].astype(str).str.extract(r"(\d)").fillna(1).astype(int)
        else:
            frame["floors"] = 1

    required_columns = [
        "price",
        "area_sqft",
        "bedrooms",
        "bathrooms",
        "location",
        "parking",
        "furnishing_status",
        "age_of_property",
        "nearby_facilities",
        "floors",
        "property_type",
    ]

    available = [col for col in required_columns if col in frame.columns]
    missing = sorted(set(required_columns) - set(available))
    if missing:
        logger.warning("Some required columns were not found and may be derived later: %s", missing)

    return frame


def load_or_download_dataset(csv_path: Optional[Path] = None, save_download: bool = True) -> pd.DataFrame:
    """Load housing dataset from CSV, or download Ames Housing from OpenML."""
    candidate_path = csv_path or DATA_DIR / "housing.csv"

    if candidate_path.exists():
        logger.info("Loading dataset from %s", candidate_path)
        raw_df = pd.read_csv(candidate_path)
        return standardize_columns(raw_df)

    logger.info("No local dataset found. Downloading Ames Housing from OpenML...")
    dataset = fetch_openml(name="house_prices", version=1, as_frame=True, parser="auto")
    raw_df = dataset.frame

    if save_download:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(DATA_DIR / "housing_openml_raw.csv", index=False)
        logger.info("Saved raw downloaded dataset to data/housing_openml_raw.csv")

    cleaned = standardize_columns(raw_df)
    if save_download:
        cleaned.to_csv(DATA_DIR / "housing.csv", index=False)
        logger.info("Saved standardized dataset to data/housing.csv")

    return cleaned
