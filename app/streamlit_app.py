from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.predict import HousePricePredictor


st.set_page_config(page_title="House Price AI", page_icon="🏠", layout="wide")

st.title("House Price Prediction Dashboard")
st.caption("Production-style ML app with confidence score and similar property retrieval")

@st.cache_resource
def load_predictor() -> HousePricePredictor:
    return HousePricePredictor()


predictor = load_predictor()

col1, col2, col3 = st.columns(3)

with col1:
    area_sqft = st.number_input("Area (sq ft)", min_value=200.0, value=1800.0, step=50.0)
    bedrooms = st.number_input("Bedrooms", min_value=0, value=3, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=0, value=2, step=1)

with col2:
    location = st.text_input("Location", value="NAmes")
    parking = st.number_input("Parking", min_value=0, value=1, step=1)
    furnishing_status = st.selectbox("Furnishing", ["unfurnished", "semi-furnished", "furnished"])

with col3:
    age_of_property = st.number_input("Property Age", min_value=0, value=10, step=1)
    nearby_facilities = st.selectbox("Nearby Facilities", ["basic", "good", "excellent"])
    floors = st.number_input("Floors", min_value=1, value=1, step=1)
    property_type = st.text_input("Property Type", value="House")

if st.button("Predict Price", type="primary"):
    payload = {
        "area_sqft": area_sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "location": location,
        "parking": parking,
        "furnishing_status": furnishing_status,
        "age_of_property": age_of_property,
        "nearby_facilities": nearby_facilities,
        "floors": floors,
        "property_type": property_type,
    }

    result = predictor.predict(payload)
    st.subheader("Prediction")
    st.metric("Predicted Price", f"${result.predicted_price:,.0f}")
    st.metric("Confidence", f"{result.confidence_score * 100:.1f}%")

    st.subheader("Similar Properties")
    if result.similar_properties:
        st.dataframe(result.similar_properties, use_container_width=True)
    else:
        st.info("No similar properties found.")

st.markdown("---")
st.write("Model file:", Path(predictor.model_path).name)
