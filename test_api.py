"""
Test suite for House Price Prediction API.
Tests model loading, predictions, and API endpoints.
"""

import pytest
import os
from predict import PredictionService, NUMERIC_FEATURES, CATEGORICAL_FEATURES


class TestPredictionService:
    """Test cases for PredictionService."""

    @pytest.fixture(scope="session", autouse=True)
    def setup(self):
        """Check if model files exist before running tests."""
        if not os.path.exists("model.pkl"):
            raise FileNotFoundError(
                "model.pkl not found. Run 'python train.py' first to train and save the model."
            )
        if not os.path.exists("preprocessor.pkl"):
            raise FileNotFoundError(
                "preprocessor.pkl not found. Run 'python train.py' first."
            )

    def test_prediction_service_init(self):
        """Test PredictionService initialization."""
        service = PredictionService()
        assert service.model is not None
        assert service.preprocessor is not None
        print("✓ PredictionService initialized successfully")

    def test_single_prediction(self):
        """Test single prediction."""
        service = PredictionService()
        
        # Test data - mid-price house
        test_data = {
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
            "house_age": 25,
            "renovated": 0
        }
        
        result = service.predict(test_data)
        
        assert "predicted_price" in result
        assert "price_lower" in result
        assert "price_upper" in result
        assert "formatted_price" in result
        assert result["predicted_price"] > 0
        assert result["price_lower"] < result["predicted_price"] < result["price_upper"]
        
        print(f"✓ Single prediction successful: {result}")

    def test_batch_prediction(self):
        """Test batch predictions."""
        service = PredictionService()
        
        # Test data - multiple houses
        test_data_list = [
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
                "house_age": 25,
                "renovated": 0
            },
            {
                "bedrooms": 5,
                "bathrooms": 3.5,
                "sqft_living": 4000,
                "sqft_lot": 10000,
                "floors": 2.0,
                "waterfront": 1,
                "view": 4,
                "condition": 5,
                "grade": 11,
                "sqft_above": 3000,
                "sqft_basement": 1000,
                "yr_built": 2010,
                "zipcode": 98004,
                "lat": 47.6205,
                "long": -122.2047,
                "sqft_living15": 3800,
                "sqft_lot15": 9500,
                "house_age": 5,
                "renovated": 1
            }
        ]
        
        results = service.predict_batch(test_data_list)
        
        assert len(results) == 2
        for result in results:
            assert "predicted_price" in result
            assert "price_lower" in result
            assert "price_upper" in result
            assert result["predicted_price"] > 0
        
        print(f"✓ Batch prediction successful: {len(results)} predictions made")

    def test_feature_names(self):
        """Test that feature names are correct."""
        service = PredictionService()
        features = service.get_feature_names()
        
        assert "numeric_features" in features
        assert "categorical_features" in features
        
        # Verify correct feature counts (including engineered features)
        assert len(features["numeric_features"]) == 14
        assert len(features["categorical_features"]) == 5
        assert len(features["all_features"]) == 19
        
        print(f"✓ Feature names correct:")
        print(f"  Numeric: {features['numeric_features']}")
        print(f"  Categorical: {features['categorical_features']}")

    def test_missing_features_error(self):
        """Test handling of missing features."""
        service = PredictionService()
        
        # Test data with missing feature
        incomplete_data = {
            "bedrooms": 3,
            "bathrooms": 2.0,
            "sqft_living": 2000,
            # Missing other required features
        }
        
        # This should raise an error
        try:
            result = service.predict(incomplete_data)
            print("✗ Should have raised an error for missing features")
            assert False, "Expected error for missing features"
        except Exception as e:
            print(f"✓ Correctly raised error for missing features: {type(e).__name__}")


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    # Run without pytest (simple script mode)
    print("=" * 70)
    print("Running House Price Prediction Tests")
    print("=" * 70)
    
    try:
        # Check if model files exist
        if not os.path.exists("model.pkl"):
            raise FileNotFoundError(
                "model.pkl not found. Run 'python train.py' first to train and save the model."
            )
        if not os.path.exists("preprocessor.pkl"):
            raise FileNotFoundError(
                "preprocessor.pkl not found. Run 'python train.py' first."
            )
        
        test = TestPredictionService()
        
        print("\n[1/5] Testing PredictionService Initialization...")
        test.test_prediction_service_init()
        
        print("\n[2/5] Testing Single Prediction...")
        test.test_single_prediction()
        
        print("\n[3/5] Testing Batch Prediction...")
        test.test_batch_prediction()
        
        print("\n[4/5] Testing Feature Names...")
        test.test_feature_names()
        
        print("\n[5/5] Testing Missing Features Error Handling...")
        test.test_missing_features_error()
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        raise
