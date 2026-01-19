"""
FastAPI application for House Price Prediction ML model.
Handles predictions via REST API endpoints.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from predict import PredictionService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global prediction service instance
service: PredictionService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle - startup and shutdown.
    """
    global service
    # Startup
    logger.info("FastAPI application starting up...")
    service = PredictionService()
    logger.info(f"Numeric features: {service.get_feature_names()['numeric_features']}")
    logger.info(f"Categorical features: {service.get_feature_names()['categorical_features']}")
    
    yield
    
    # Shutdown
    logger.info("FastAPI application shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="House Price Prediction API",
    description="ML API for predicting house prices in King County using Random Forest",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================================
# Pydantic Models
# ============================================================================

class HousePredictionRequest(BaseModel):
    """
    Request schema for house price prediction.
    All fields are required.
    """
    model_config = ConfigDict(protected_namespaces=())
    
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    zipcode: int
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int
    house_age: int
    renovated: int


class PredictionResponse(BaseModel):
    """Response schema for prediction results."""
    model_config = ConfigDict(protected_namespaces=())
    
    predicted_price: float
    price_lower: float
    price_upper: float
    formatted_price: str


class HealthResponse(BaseModel):
    """Response schema for health check."""
    model_config = ConfigDict(protected_namespaces=())
    
    status: str
    model_loaded: bool
    model_type: str


class FeaturesResponse(BaseModel):
    """Response schema for feature information."""
    model_config = ConfigDict(protected_namespaces=())
    
    numeric_features: List[str]
    categorical_features: List[str]
    all_features: List[str]


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - welcome message."""
    return {
        "message": "House Price Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Verifies that the model is loaded and ready.
    """
    return {
        "status": "healthy" if service.model is not None else "unhealthy",
        "model_loaded": service.model is not None,
        "model_type": "RandomForestRegressor with preprocessing pipeline"
    }


@app.get("/features", response_model=FeaturesResponse, tags=["Info"])
async def get_features():
    """
    Get expected feature names for predictions.
    """
    features = service.get_feature_names()
    return {
        "numeric_features": features["numeric_features"],
        "categorical_features": features["categorical_features"],
        "all_features": features["all_features"]
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: HousePredictionRequest):
    """
    Make a single prediction for house price.
    
    **Parameters:**
    - **bedrooms**: Number of bedrooms (int)
    - **bathrooms**: Number of bathrooms (float, can have .5 for half bath)
    - **sqft_living**: Square footage of living space (int)
    - **sqft_lot**: Square footage of lot (int)
    - **floors**: Number of floors (float, can have .5)
    - **waterfront**: Waterfront property (0/1)
    - **view**: Quality of view (0-4)
    - **condition**: Overall condition (1-5)
    - **grade**: Overall grade given to the housing unit (1-13)
    - **sqft_above**: Square footage above ground (int)
    - **sqft_basement**: Square footage of basement (int)
    - **yr_built**: Year built (int)
    - **zipcode**: Zipcode (int)
    - **lat**: Latitude (float)
    - **long**: Longitude (float)
    - **sqft_living15**: Average square footage of 15 nearest houses (int)
    - **sqft_lot15**: Average lot size of 15 nearest houses (int)
    - **house_age**: Age of the house in years (int)
    - **renovated**: Whether the house was renovated (0/1)
    
    **Returns:**
    - **predicted_price**: Predicted house price ($)
    - **price_lower**: Lower bound of 95% prediction interval
    - **price_upper**: Upper bound of 95% prediction interval
    - **formatted_price**: Formatted price string
    """
    try:
        result = service.predict(request.model_dump())
        return result
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=List[PredictionResponse], tags=["Predictions"])
async def predict_batch(requests: List[HousePredictionRequest]):
    """
    Make predictions for multiple houses at once.
    
    **Parameters:**
    - List of HousePredictionRequest objects
    
    **Returns:**
    - List of PredictionResponse objects
    """
    try:
        data_list = [req.model_dump() for req in requests]
        results = service.predict_batch(data_list)
        return results
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/info", tags=["Info"])
async def model_info():
    """
    Get detailed information about the model and API.
    """
    return {
        "model_type": "Random Forest Regressor",
        "framework": "scikit-learn",
        "preprocessing": "StandardScaler + OneHotEncoder via ColumnTransformer",
        "features_total": 19,
        "numeric_features": 14,
        "categorical_features": 5,
        "target": "House price in King County, WA (regression)",
        "endpoints": {
            "health": "GET /health",
            "features": "GET /features",
            "predict": "POST /predict",
            "batch_predict": "POST /predict/batch",
            "model_info": "GET /info",
            "docs": "GET /docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
