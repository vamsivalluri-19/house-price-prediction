<<<<<<< HEAD
# Production-Ready House Price Prediction System
# Production-Ready House Price Prediction System

This project provides an end-to-end machine learning platform to predict house prices using structured property features and production-grade engineering practices.

## Features

- End-to-end ML workflow: data loading -> preprocessing -> training -> evaluation -> deployment
- Detailed EDA with saved plots:
  - Missing values overview
  - Correlation heatmap
  - Distribution plots
  - Outlier boxplots
  - Pairplot
  - Feature importance chart
  - Prediction vs Actual and Residual plot
  - Model comparison charts
- Feature engineering:
  - `price_per_sqft`
  - `total_rooms`
  - `luxury_score`
  - `location_popularity_index`
- Models implemented and compared:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - XGBoost Regressor (if installed)
  - Stacking Ensemble (advanced)
- Cross-validation and hyperparameter tuning via `GridSearchCV`
- Metrics:
  - MAE
  - MSE
  - RMSE
  - R2
  - Adjusted R2
- Real-time prediction system with:
  - Predicted price
  - Confidence score
  - Similar property suggestions
- Deployment:
  - FastAPI backend
  - Streamlit dashboard
  - Docker and docker-compose support
- Engineering best practices:
  - Modular architecture
  - Logging
  - Exception handling
  - Unit tests
  - Type hints
  - Model persistence via Joblib

## Project Structure

```
house-price-prediction/
├── data/
│   └── sample_housing.csv
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── data.py
│   ├── evaluate.py
│   ├── pipeline.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── train.py
│   └── utils.py
├── tests/
│   ├── test_evaluate.py
│   └── test_preprocessing.py
├── models/
├── app/
│   └── streamlit_app.py
├── logs/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── main.py
```

## Dataset

The project supports:
1. Local CSV dataset at `data/housing.csv` (or custom path)
2. Auto-download of real-world Ames Housing dataset from OpenML if no local dataset is found

A sample dataset is included in `data/sample_housing.csv`.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train the Model

Use sample dataset:

```bash
python main.py train --data-path data/sample_housing.csv
```

Or rely on auto-download:

```bash
python main.py train
```

After training, artifacts are created in `models/`:
- `best_model_*.joblib`
- `metrics_*.json`
- `model_report_*.json`
- `plots_*/`

## Run API (FastAPI)

```bash
python main.py serve --host 0.0.0.0 --port 8000
```

API endpoints:
- `GET /health`
- `POST /predict`

### Example Predict Request

```json
{
  "area_sqft": 1700,
  "bedrooms": 3,
  "bathrooms": 2,
  "location": "NAmes",
  "parking": 2,
  "furnishing_status": "semi-furnished",
  "age_of_property": 12,
  "nearby_facilities": "good",
  "floors": 2,
  "property_type": "House"
}
```

### Example Predict Response

```json
{
  "predicted_price": 312450.34,
  "confidence_score": 0.89,
  "similar_properties": [
    {
      "price": 305000.0,
      "area_sqft": 1600,
      "bedrooms": 3,
      "bathrooms": 2,
      "location": "CollgCr",
      "parking": 2,
      "furnishing_status": "furnished",
      "age_of_property": 10,
      "distance": 115.42
    }
  ]
}
```

## Run Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

## Unit Tests

```bash
pytest -q
```

## Docker Deployment

Build and run API + dashboard:

```bash
docker-compose up --build
```

Services:
- API: `http://localhost:8000`
- Streamlit: `http://localhost:8501`

## Bonus Features Included

Starter modules are included in `src/bonus/` for:
- House image analysis (CNN-ready skeleton)
- Location intelligence scoring
- AI recommendation engine for listing price bands
- Real-estate Q&A chatbot (intent-based)
- Time-series forecasting for future price trends

## Notes on Production Use

For real deployments:
- Connect to production data pipelines and feature store
- Add model monitoring and drift detection
- Version datasets and models (DVC/MLflow)
- Add CI/CD for tests and model release
- Add authentication and rate limiting for API
This project provides an end-to-end machine learning platform to predict house prices using structured property features and production-grade engineering practices.

## Features

- End-to-end ML workflow: data loading -> preprocessing -> training -> evaluation -> deployment
- Detailed EDA with saved plots:
  - Missing values overview
  - Correlation heatmap
  - Distribution plots
  - Outlier boxplots
  - Pairplot
  - Feature importance chart
  - Prediction vs Actual and Residual plot
  - Model comparison charts
- Feature engineering:
  - `price_per_sqft`
  - `total_rooms`
  - `luxury_score`
  - `location_popularity_index`
- Models implemented and compared:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - XGBoost Regressor (if installed)
  - Stacking Ensemble (advanced)
- Cross-validation and hyperparameter tuning via `GridSearchCV`
- Metrics:
  - MAE
  - MSE
  - RMSE
  - R2
  - Adjusted R2
- Real-time prediction system with:
  - Predicted price
  - Confidence score
  - Similar property suggestions
- Deployment:
  - FastAPI backend
  - Streamlit dashboard
  - Docker and docker-compose support
- Engineering best practices:
  - Modular architecture
  - Logging
  - Exception handling
  - Unit tests
  - Type hints
  - Model persistence via Joblib

## Project Structure

```
house-price-prediction/
├── data/
│   └── sample_housing.csv
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── data.py
│   ├── evaluate.py
│   ├── pipeline.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── train.py
│   └── utils.py
├── tests/
│   ├── test_evaluate.py
│   └── test_preprocessing.py
├── models/
├── app/
│   └── streamlit_app.py
├── logs/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── main.py
```

## Dataset

The project supports:
1. Local CSV dataset at `data/housing.csv` (or custom path)
2. Auto-download of real-world Ames Housing dataset from OpenML if no local dataset is found

A sample dataset is included in `data/sample_housing.csv`.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train the Model

Use sample dataset:

```bash
python main.py train --data-path data/sample_housing.csv
```

Or rely on auto-download:

```bash
python main.py train
```

After training, artifacts are created in `models/`:
- `best_model_*.joblib`
- `metrics_*.json`
- `model_report_*.json`
- `plots_*/`

## Run API (FastAPI)

```bash
python main.py serve --host 0.0.0.0 --port 8000
```

API endpoints:
- `GET /health`
- `POST /predict`

### Example Predict Request

```json
{
  "area_sqft": 1700,
  "bedrooms": 3,
  "bathrooms": 2,
  "location": "NAmes",
  "parking": 2,
  "furnishing_status": "semi-furnished",
  "age_of_property": 12,
  "nearby_facilities": "good",
  "floors": 2,
  "property_type": "House"
}
```

### Example Predict Response

```json
{
  "predicted_price": 312450.34,
  "confidence_score": 0.89,
  "similar_properties": [
    {
      "price": 305000.0,
      "area_sqft": 1600,
      "bedrooms": 3,
      "bathrooms": 2,
      "location": "CollgCr",
      "parking": 2,
      "furnishing_status": "furnished",
      "age_of_property": 10,
      "distance": 115.42
    }
  ]
}
```

## Run Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

## Unit Tests

```bash
pytest -q
```

## Docker Deployment

Build and run API + dashboard:

```bash
docker-compose up --build
```

Services:
- API: `http://localhost:8000`
- Streamlit: `http://localhost:8501`

## Bonus Features Included

Starter modules are included in `src/bonus/` for:
- House image analysis (CNN-ready skeleton)
- Location intelligence scoring
- AI recommendation engine for listing price bands
- Real-estate Q&A chatbot (intent-based)
- Time-series forecasting for future price trends

## Notes on Production Use

For real deployments:
- Connect to production data pipelines and feature store
- Add model monitoring and drift detection
- Version datasets and models (DVC/MLflow)
- Add CI/CD for tests and model release
- Add authentication and rate limiting for API
