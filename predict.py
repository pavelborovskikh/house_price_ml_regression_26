"""
Prediction service for House Price Prediction.
Handles single and batch predictions with proper preprocessing.
"""

import logging
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Configure logging
logger = logging.getLogger(__name__)

# Feature definitions
NUMERIC_FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "sqft_above", "sqft_basement", "yr_built", "lat", "long",
    "sqft_living15", "sqft_lot15", "house_age", "renovated"
]
CATEGORICAL_FEATURES = [
    "waterfront", "view", "condition", "grade", "zipcode"
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MODEL_PATH = "model.pkl"
PREPROCESSOR_PATH = "preprocessor.pkl"

# Prediction uncertainty factor (15% of predicted value for confidence interval)
PREDICTION_UNCERTAINTY_FACTOR = 0.15


class PredictionService:
    """
    Service for loading trained model and making predictions.
    Handles both single and batch predictions.
    """

    def __init__(self):
        """Initialize the prediction service by loading model and preprocessor."""
        self.model = None
        self.preprocessor = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the trained model and preprocessor from pickle files."""
        try:
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded from {MODEL_PATH}")
            
            # Try to load preprocessor
            try:
                with open(PREPROCESSOR_PATH, "rb") as f:
                    self.preprocessor = pickle.load(f)
                logger.info(f"Preprocessor loaded from {PREPROCESSOR_PATH}")
            except FileNotFoundError:
                logger.warning(f"Preprocessor file not found: {PREPROCESSOR_PATH}")
                self.preprocessor = None
                
        except FileNotFoundError:
            logger.error(f"Model file not found: {MODEL_PATH}")
            raise

    def _validate_input(self, data: Dict[str, Any]) -> None:
        """Validate that input contains all required features."""
        missing_features = set(ALL_FEATURES) - set(data.keys())
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

    def _prepare_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare a pandas DataFrame from input data.
        
        Args:
            data: Dictionary containing feature values
            
        Returns:
            DataFrame with features in correct order
        """
        # Validate input
        self._validate_input(data)
        
        # Create DataFrame with features in correct order
        df = pd.DataFrame([data])
        
        # Select only the required features in correct order
        df = df[ALL_FEATURES]
        
        return df

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a single prediction.
        
        Args:
            data: Dictionary with features for one house
                 Expected keys: bedrooms, bathrooms, sqft_living, sqft_lot, floors,
                              waterfront, view, condition, grade, sqft_above,
                              sqft_basement, yr_built, zipcode, lat, long,
                              sqft_living15, sqft_lot15, house_age, renovated
        
        Returns:
            Dictionary with prediction and metadata
        """
        try:
            # Prepare data
            df = self._prepare_dataframe(data)
            
            # Use model's built-in preprocessing pipeline
            prediction = self.model.predict(df)[0]
            
            # For regression, we don't have probability but can provide prediction interval
            # Using a simple approximation based on model performance
            prediction_std = prediction * PREDICTION_UNCERTAINTY_FACTOR
            
            return {
                "predicted_price": float(prediction),
                "price_lower": float(prediction - 1.96 * prediction_std),
                "price_upper": float(prediction + 1.96 * prediction_std),
                "formatted_price": f"${prediction:,.2f}"
            }
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def predict_batch(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple samples.
        
        Args:
            data_list: List of dictionaries, each containing features for one house
        
        Returns:
            List of prediction dictionaries
        """
        try:
            # Prepare data
            dfs = [self._prepare_dataframe(data) for data in data_list]
            df_batch = pd.concat(dfs, ignore_index=True)
            
            # Make predictions
            predictions = self.model.predict(df_batch)
            
            results = []
            for i, pred in enumerate(predictions):
                prediction_std = pred * PREDICTION_UNCERTAINTY_FACTOR
                results.append({
                    "sample": i,
                    "predicted_price": float(pred),
                    "price_lower": float(pred - 1.96 * prediction_std),
                    "price_upper": float(pred + 1.96 * prediction_std),
                    "formatted_price": f"${pred:,.2f}"
                })
            
            return results
        except Exception as e:
            logger.error(f"Error during batch prediction: {str(e)}")
            raise

    def get_feature_names(self) -> Dict[str, List[str]]:
        """
        Get the names of features the model expects.
        
        Returns:
            Dictionary with numeric and categorical feature names
        """
        return {
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "all_features": ALL_FEATURES
        }
