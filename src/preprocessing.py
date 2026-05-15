from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


TARGET_COLUMN = "price"


@dataclass
class HousingPreprocessor:
    """Preprocess tabular housing data with feature engineering and encoding."""

    target_column: str = TARGET_COLUMN
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    location_popularity_map: dict[str, float] = field(default_factory=dict)
    label_encoders: dict[str, LabelEncoder] = field(default_factory=dict)
    transformer: Optional[ColumnTransformer] = None

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        frame = frame.drop_duplicates().reset_index(drop=True)
        return frame

    def _fit_location_popularity(self, frame: pd.DataFrame) -> None:
        if "location" not in frame.columns:
            self.location_popularity_map = {}
            return

        grouped = frame.groupby("location", dropna=False)[self.target_column].median().sort_values()
        if grouped.empty:
            self.location_popularity_map = {}
            return

        ranks = grouped.rank(method="dense")
        norm = (ranks - ranks.min()) / max(1, ranks.max() - ranks.min())
        self.location_popularity_map = norm.to_dict()
        # store median price per location for use as a robust location feature
        self.location_median_price_map = grouped.to_dict()

    def _feature_engineering(self, frame: pd.DataFrame, fit: bool) -> pd.DataFrame:
        data = frame.copy()

        if "area_sqft" in data.columns and self.target_column in data.columns:
            data["price_per_sqft"] = data[self.target_column] / data["area_sqft"].replace(0, np.nan)
        else:
            data["price_per_sqft"] = np.nan

        bedrooms = data.get("bedrooms", 0).fillna(0) if "bedrooms" in data.columns else 0
        bathrooms = data.get("bathrooms", 0).fillna(0) if "bathrooms" in data.columns else 0
        data["total_rooms"] = bedrooms + bathrooms

        # additional engineered features
        data["area_per_room"] = (
            data["area_sqft"] / data["total_rooms"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        data["rooms_per_floor"] = (
            data["total_rooms"] / data.get("floors", 1).replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        data["area_sqft_sq"] = data.get("area_sqft", 0).fillna(0) ** 2
        data["area_bedrooms_interaction"] = (
            data.get("area_sqft", 0).fillna(0) * data.get("bedrooms", 0).fillna(0)
        )
        data["area_log"] = np.log1p(data.get("area_sqft", 0).fillna(0))
        data["bedrooms_per_bathroom"] = (
            data.get("bedrooms", 0).fillna(0) / (data.get("bathrooms", 0).fillna(0) + 1)
        )

        furnishing_score_map = {"unfurnished": 0.2, "semi-furnished": 0.6, "furnished": 1.0}
        facility_score_map = {"basic": 0.3, "good": 0.7, "excellent": 1.0}

        furnishing = data.get("furnishing_status", "semi-furnished").astype(str).str.lower()
        facilities = data.get("nearby_facilities", "good").astype(str).str.lower()
        parking = data.get("parking", 0).fillna(0).astype(float) if "parking" in data.columns else 0
        floors = data.get("floors", 1).fillna(1).astype(float) if "floors" in data.columns else 1

        data["luxury_score"] = (
            furnishing.map(furnishing_score_map).fillna(0.5) * 0.35
            + facilities.map(facility_score_map).fillna(0.5) * 0.35
            + np.clip(parking / 3.0, 0, 1) * 0.15
            + np.clip(floors / 3.0, 0, 1) * 0.15
        )

        if fit:
            self._fit_location_popularity(data)

        # location median price mapping (robust location signal)
        if hasattr(self, "location_median_price_map"):
            data["location_median_price"] = (
                data.get("location", "Unknown").astype(str).map(self.location_median_price_map).fillna(np.nan)
            )
        else:
            data["location_median_price"] = np.nan

        data["location_popularity_index"] = (
            data.get("location", "Unknown").astype(str).map(self.location_popularity_map).fillna(0.5)
        )

        # age-based buckets and indicators
        if "age_of_property" in data.columns:
            data["age_bucket"] = pd.cut(
                data["age_of_property"], bins=[-1, 5, 15, 30, np.inf], labels=["new", "recent", "mature", "old"]
            ).astype(str)
            data["is_recent"] = (data["age_of_property"] <= 5).astype(int)
        else:
            data["age_bucket"] = "mature"
            data["is_recent"] = 0

        if "furnishing_status" in data.columns:
            data["furnishing_label"] = self._fit_or_transform_label(data, "furnishing_status", fit)

        if "nearby_facilities" in data.columns:
            data["facilities_label"] = self._fit_or_transform_label(data, "nearby_facilities", fit)

        return data

    def _fit_or_transform_label(self, frame: pd.DataFrame, col: str, fit: bool) -> pd.Series:
        values = frame[col].astype(str).fillna("unknown")
        if fit or col not in self.label_encoders:
            encoder = LabelEncoder()
            frame_values = encoder.fit_transform(values)
            self.label_encoders[col] = encoder
            return pd.Series(frame_values, index=frame.index)

        encoder = self.label_encoders[col]
        known = set(encoder.classes_)
        safe_values = values.where(values.isin(known), other=encoder.classes_[0])
        return pd.Series(encoder.transform(safe_values), index=frame.index)

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, pd.Series, pd.DataFrame]:
        frame = self.clean(df)
        frame = self._feature_engineering(frame, fit=True)

        if self.target_column not in frame.columns:
            raise ValueError(f"Target column '{self.target_column}' is missing from input data.")

        y = frame[self.target_column]
        x_frame = frame.drop(columns=[self.target_column])

        self.numeric_features = x_frame.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = x_frame.select_dtypes(exclude=[np.number]).columns.tolist()

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        self.transformer = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, self.numeric_features),
                ("cat", categorical_pipeline, self.categorical_features),
            ]
        )

        x = self.transformer.fit_transform(x_frame)
        return x, y, frame

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.transformer is None:
            raise RuntimeError("Preprocessor is not fitted. Call fit_transform first.")

        frame = self.clean(df)
        frame = self._feature_engineering(frame, fit=False)

        x_frame = frame.drop(columns=[self.target_column], errors="ignore")
        return self.transformer.transform(x_frame)

    def get_feature_names(self) -> list[str]:
        if self.transformer is None:
            return []

        names: list[str] = []
        names.extend(self.numeric_features)
        onehot = self.transformer.named_transformers_["cat"].named_steps["onehot"]
        names.extend(onehot.get_feature_names_out(self.categorical_features).tolist())
        return names

    def transform_single(self, features: dict[str, Any]) -> np.ndarray:
        """Transform a single property payload for prediction."""
        frame = pd.DataFrame([features])
        return self.transform(frame)
