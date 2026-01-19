#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for House Price Prediction API
ML Zoomcamp Capstone Project
"""

import requests

# Configuration
# Change this URL to your deployed Fly.io URL after deployment
url = "http://localhost:8000/predict"
# For Fly.io deployment, use:
# url = "https://house-price-ml.fly.dev/predict"

# Sample house data - High-price house
house_high_price = {
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
    "sqft_lot15": 9500
    ,"house_age": 5,
    "renovated": 1
}

# Sample house data - Mid-price house
house_mid_price = {
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
    "sqft_lot15": 4800
    ,"house_age": 25,
    "renovated": 0
}

# Sample house data - Low-price house
house_low_price = {
    "bedrooms": 2,
    "bathrooms": 1.0,
    "sqft_living": 1200,
    "sqft_lot": 3000,
    "floors": 1.0,
    "waterfront": 0,
    "view": 0,
    "condition": 3,
    "grade": 6,
    "sqft_above": 1000,
    "sqft_basement": 200,
    "yr_built": 1970,
    "zipcode": 98002,
    "lat": 47.3089,
    "long": -122.2348,
    "sqft_living15": 1100,
    "sqft_lot15": 2900
    ,"house_age": 45,
    "renovated": 0
}

def test_prediction(house_data, house_name):
    """Test the prediction endpoint with house data."""
    print(f"\n{'='*60}")
    print(f"Testing: {house_name}")
    print(f"{'='*60}")
    
    try:
        response = requests.post(url, json=house_data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful!")
            print(f"\nResults:")
            print(f"  - Predicted Price: {result['formatted_price']}")
            print(f"  - Price Range: ${result['price_lower']:,.2f} - ${result['price_upper']:,.2f}")
            print(f"  - Exact Value: ${result['predicted_price']:,.2f}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
    
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection Error: Could not connect to {url}")
        print("   Make sure the API is running!")
    except requests.exceptions.Timeout:
        print(f"❌ Timeout: Request took too long")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("House Price Prediction API - Test Script")
    print("ML Zoomcamp Capstone Project")
    print("="*60)
    print(f"\nTesting endpoint: {url}")
    
    # Test with high-price house
    test_prediction(house_high_price, "High-Price House (5BR, Waterfront, Luxury)")
    
    # Test with mid-price house
    test_prediction(house_mid_price, "Mid-Price House (3BR, Standard)")
    
    # Test with low-price house
    test_prediction(house_low_price, "Low-Price House (2BR, Older)")
    
    print(f"\n{'='*60}")
    print("Testing complete!")
    print("="*60 + "\n")
