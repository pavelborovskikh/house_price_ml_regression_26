# 🏠 House Price Prediction API - ML Zoomcamp Capstone Project

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)

A complete end-to-end machine learning system for predicting house prices in King County, WA using XGBoost Regressor. Built as a capstone project for ML Zoomcamp.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Docker Deployment](#docker-deployment)
- [Testing](#testing)
- [Technologies Used](#technologies-used)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

This project implements a production-ready machine learning API that predicts house prices based on various features like square footage, location, number of bedrooms/bathrooms, and property characteristics. The system includes:

- Exploratory data analysis and feature engineering
- Multiple regression models comparison (Linear Regression, Random Forest, XGBoost)
- Hyperparameter tuning using RandomizedSearchCV
- Production-ready FastAPI REST API
- Docker containerization
- Comprehensive testing suite

## ✨ Key Features

- **Multiple Models**: Compares Linear Regression, Random Forest, and XGBoost
- **Preprocessing Pipeline**: StandardScaler for numeric features, OneHotEncoder for categorical
- **REST API**: FastAPI with automatic OpenAPI documentation
- **Batch Predictions**: Support for single and batch predictions
- **Error Handling**: Comprehensive input validation and error messages
- **Docker Ready**: Containerized application with model baked in
- **Production Logging**: Detailed logging for monitoring and debugging
- **Health Checks**: `/health` endpoint for service monitoring

## 📊 Dataset

**Source**: King County House Sales Dataset (https://www.kaggle.com/datasets/harlfoxem/housesalesprediction?resource=download)

The dataset contains house sale prices for King County, WA (includes Seattle) from May 2014 to May 2015.

### Features (19 total)

| Feature | Type | Description |
|---------|------|-------------|
| `bedrooms` | Numeric | Number of bedrooms |
| `bathrooms` | Numeric | Number of bathrooms |
| `sqft_living` | Numeric | Square footage of living space |
| `sqft_lot` | Numeric | Square footage of the lot |
| `floors` | Numeric | Number of floors |
| `waterfront` | Categorical | Waterfront property (0/1) |
| `view` | Categorical | Quality of view (0-4) |
| `condition` | Categorical | Overall condition (1-5) |
| `grade` | Categorical | Overall grade (1-13) |
| `sqft_above` | Numeric | Square footage above ground |
| `sqft_basement` | Numeric | Square footage of basement |
| `yr_built` | Numeric | Year house was built |
| `zipcode` | Categorical | Zipcode |
| `lat` | Numeric | Latitude coordinate |
| `long` | Numeric | Longitude coordinate |
| `sqft_living15` | Numeric | Average sqft of 15 nearest houses |
| `sqft_lot15` | Numeric | Average lot size of 15 nearest houses |
| `house_age` | Numeric | Age of the house in years |
| `renovated` | Numeric | Renovated flag (0 = no, 1 = yes) |

**Target Variable**: `price` (house sale price in USD)

## 🔧 Machine Learning Pipeline

### 1. Data Preprocessing
- Handle missing values (median for numeric, most frequent for categorical)
- Feature engineering: house age, renovation status
- Remove duplicates

### 2. Feature Transformation
- **Numeric Features** (14): StandardScaler normalization
- **Categorical Features** (5): OneHotEncoder for categorical variables

### 3. Model Training
Three regression models trained and compared:
- **Linear Regression**: Baseline model
- **Random Forest Regressor**: Ensemble method with hyperparameter tuning
- **XGBoost Regressor**: Gradient boosting algorithm

### 4. Hyperparameter Tuning
- RandomizedSearchCV on Random Forest:
  - `n_estimators`: [100, 200, 300]
  - `max_depth`: [None, 10, 20, 30]
  - `min_samples_split`: [2, 5, 10]
  - `min_samples_leaf`: [1, 2, 4]
  - `max_features`: ['sqrt', 'log2']

- Ridge (linear) tuning via `GridSearchCV` on `alpha` values: [0.001, 0.01, 0.1, 1, 10, 100, 1000]

- XGBoost tuning via `RandomizedSearchCV` (example search space):
  - `n_estimators`: [100, 200, 400]
  - `max_depth`: [3, 5, 7, 9]
  - `learning_rate`: [0.01, 0.05, 0.1, 0.2]
  - `subsample`: [0.6, 0.8, 1.0]
  - `colsample_bytree`: [0.6, 0.8, 1.0]
  - `gamma`: [0, 1, 5]

### 5. Model Evaluation
Metrics used:
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **MAE** (Mean Absolute Error): Average prediction error
- **R² Score**: Proportion of variance explained

## 📈 Model Performance

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| Linear Regression (Tuned) | $157,312.83 | $90,950.29 | 0.8363 |
| Random Forest (Tuned) | $152,176.22 | $80,284.27 | 0.8468 |
| XGBoost (Tuned) | $137,520.86 | $69,157.68 | 0.8749 |

**Selected Model**: XGBoost (Tuned) - Best balance of performance and interpretability

## 📁 Project Structure

```
├── kc_house_data.csv           # Dataset (1000 houses)
├── notebook.ipynb              # Exploratory data analysis
├── train.py                    # Model training script
├── predict.py                  # Prediction service
├── app.py                      # FastAPI application
├── test.py                     # Integration tests
├── test_api.py                 # Unit tests (pytest)
├── Dockerfile                  # Docker configuration
├── pyproject.toml              # Dependencies (uv/pip)
├── model.pkl                   # Trained model (generated)
├── preprocessor.pkl            # Fitted preprocessor (generated)
├── README.md                   # This file
├── quickstart.md               # Quick setup guide
└── submission_checklist.md     # ML Zoomcamp requirements
```

## 🚀 Installation

### Option 1: Using `uv` (Recommended)

```bash
# Install uv
pip install uv

# Install dependencies
uv pip install -r pyproject.toml --system
```

### Option 2: Using `pip`

```bash
pip install pandas numpy scikit-learn xgboost fastapi uvicorn pydantic joblib matplotlib seaborn pytest requests
```

## 💻 Usage

### 1. Train the Model

```bash
python train.py
```

This will:
- Load and preprocess data
- Train multiple models
- Perform hyperparameter tuning
- Save best model to `model.pkl`
- Save preprocessor to `preprocessor.pkl`

### 2. Start the API Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Or:

```bash
python app.py
```

The API will be available at `http://localhost:8000`

### 3. Test the API

```bash
# Run integration tests
python test.py

# Run unit tests
python test_api.py

# Or use pytest
pytest test_api.py -v
```

## 📚 API Documentation

Once the server is running, visit:
- **Interactive API docs**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc

### Endpoints

#### `GET /`
Welcome message and API information

#### `GET /health`
Health check endpoint
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "RandomForestRegressor with preprocessing pipeline"
}
```

#### `GET /features`
List of expected features
```json
{
  "numeric_features": ["bedrooms", "bathrooms", ...],
  "categorical_features": ["waterfront", "view", ...],
  "all_features": [...]
}
```

#### `POST /predict`
Single house price prediction

**Request:**
```json
{
  "bedrooms": 3,
  "bathrooms": 2.0,
  "sqft_living": 2000,
  "sqft_lot": 5000,
  "floors": 1.0,
  "waterfront": 0,
  "view": 0,
  "condition": 3,
  "grade": 7,
  "sqft_above": 1500,
  "sqft_basement": 500,
  "yr_built": 1990,
  "zipcode": 98001,
  "lat": 47.3073,
  "long": -122.2108,
  "sqft_living15": 1900,
  "sqft_lot15": 4800,
  "house_age": 30,
  "renovated": 0
}
```

**Response:**
```json
{
  "predicted_price": 882924.29,
  "price_lower": 623344.55,
  "price_upper": 1142504.03,
  "formatted_price": "$882,924.29"
}
```

#### `POST /predict/batch`
Batch predictions (multiple houses)

**Request:** Array of house objects

**Response:** Array of prediction objects

#### `GET /info`
Model metadata and endpoint information

### Example cURL Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "bedrooms": 4,
    "bathrooms": 2.5,
    "sqft_living": 2500,
    "sqft_lot": 6000,
    "floors": 2.0,
    "waterfront": 0,
    "view": 0,
    "condition": 4,
    "grade": 8,
    "sqft_above": 2000,
    "sqft_basement": 500,
    "yr_built": 2000,
    "zipcode": 98004,
    "lat": 47.6205,
    "long": -122.2047,
    "sqft_living15": 2400,
    "sqft_lot15": 5800,
    "house_age": 20,
    "renovated": 0
  }'
```

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t house-price-api .
```

### Run Container

```bash
docker run -p 8000:8000 house-price-api
```

The API will be available at `http://localhost:8000`

### What Happens During Build

The provided `Dockerfile` is production-oriented and **does not** run model training during image build by default. Recommended approach:

1. Train the model locally to produce `model.pkl` and `preprocessor.pkl` (recommended)
2. Copy application code and the prebuilt artifacts into the image
3. Expose port 8000
4. Start uvicorn server

If you want to bake the model into the image, you can add `RUN python train.py` to the Dockerfile — note this increases build time and memory requirements.

### Fly.io Deployment

Web link: https://house-price-ml.fly.dev

This repository includes a `fly.toml` to deploy on Fly.io. Typical steps:

```bash
flyctl auth login
flyctl apps create <app-name>
flyctl deploy --app <app-name>
flyctl logs --app <app-name>
```

Make sure `model.pkl` and `preprocessor.pkl` are present in the project directory before deploying (recommended) or adapt the image to download artifacts at startup.

## 🧪 Testing

### Integration Tests

```bash
python test.py
```

Tests the API with three sample houses (high, mid, low price).

### Unit Tests

```bash
python test_api.py
```

Or with pytest:

```bash
pytest test_api.py -v
```

Tests include:
- Model initialization
- Single predictions
- Batch predictions
- Feature name validation
- Error handling for missing features

## 🛠️ Technologies Used

- **Python 3.11+**: Programming language
- **scikit-learn**: Machine learning models and preprocessing
- **XGBoost**: Gradient boosting algorithm
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **FastAPI**: Web framework for API
- **uvicorn**: ASGI server
- **Pydantic**: Data validation
- **Docker**: Containerization
- **pytest**: Testing framework
- **matplotlib/seaborn**: Data visualization

## 👏 Acknowledgments

- **ML Zoomcamp**: DataTalks.Club for the excellent ML engineering course
- **King County**: For the house sales dataset
- **FastAPI**: For the modern, fast web framework
- **scikit-learn**: For comprehensive ML tools

**Built with ❤️ for ML Zoomcamp Capstone Project**
