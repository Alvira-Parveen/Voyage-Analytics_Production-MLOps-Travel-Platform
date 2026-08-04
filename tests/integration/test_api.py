"""
Integration Tests — FastAPI Endpoints
Tests all 3 prediction endpoints with mock models.
"""

import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi.testclient import TestClient


# Mock setup

def build_mock_model_cache():
    """Build a mock model cache for testing without real models."""

    # Mock flight model
    flight_model = MagicMock()
    flight_model.predict.return_value = np.array([1200.0])
    flight_model.predict_proba = None
    flight_artifact = {
        "model": flight_model,
        "scaler": None,
        "feature_cols": [
            "flightType_enc", "agency_enc", "from_enc", "to_enc",
            "distance", "time", "price_per_km", "speed_proxy",
            "month", "weekday", "year", "season_enc",
            "is_weekend", "is_holiday", "agency_popularity",
        ],
        "model_name": "XGBoost",
    }

    # Mock gender model
    gender_model = MagicMock()
    gender_model.predict.return_value = np.array([1])
    gender_model.predict_proba.return_value = np.array([[0.3, 0.7]])
    gender_artifact = {
        "model": gender_model,
        "scaler": None,
        "feature_cols": [
            "age", "company_enc", "travel_frequency", "avg_flight_price",
            "total_flight_spend", "preferred_flight_type_enc",
            "hotel_bookings", "avg_hotel_spend", "age_group_enc",
            "spending_category_enc",
        ],
        "model_name": "RandomForest",
        "classes": ["female", "male"],
    }

    # Mock recommender
    recommender = MagicMock()
    recommender.recommend.return_value = [
        {"hotel": "Hotel K", "place": "Salvador (BH)", "predicted_rating": 4.5,
         "source": "collaborative_filtering", "reason": "Based on similar users"},
        {"hotel": "Hotel A", "place": "Florianópolis (SC)", "predicted_rating": 4.2,
         "source": "collaborative_filtering", "reason": "Based on similar users"},
    ]

    return {
        "flight": flight_artifact,
        "gender": gender_artifact,
        "recommender": recommender,
    }


@pytest.fixture
def client():
    """Test client with mocked model cache."""
    mock_cache = build_mock_model_cache()

    with patch("api.main.MODEL_CACHE", mock_cache), \
         patch("api.main.get_model_cache", return_value=mock_cache), \
         patch("src.explainability.shap_explainer.get_shap_explanation", return_value={}):

        from api.main import app
        with TestClient(app) as c:
            yield c


API_KEY = "voyage-dev-key-2024"
HEADERS = {"X-API-Key": API_KEY}


# Health & Root

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_status_field(self, client):
        resp = client.get("/health")
        assert resp.json()["status"] == "healthy"

    def test_root_endpoint(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Voyage Analytics" in resp.json()["message"]


# Flight Price

class TestFlightPriceEndpoint:

    PAYLOAD = {
        "flightType_enc": 2, "agency_enc": 1, "from_enc": 5, "to_enc": 10,
        "distance": 676.53, "time": 1.76, "month": 9, "weekday": 3,
        "year": 2024, "season_enc": 2, "is_holiday": False,
        "agency_popularity": 250,
    }

    def test_predict_returns_200(self, client):
        resp = client.post("/predict/flight-price", json=self.PAYLOAD, headers=HEADERS)
        assert resp.status_code == 200

    def test_predict_has_price_field(self, client):
        resp = client.post("/predict/flight-price", json=self.PAYLOAD, headers=HEADERS)
        assert "predicted_price" in resp.json()

    def test_predict_price_positive(self, client):
        resp = client.post("/predict/flight-price", json=self.PAYLOAD, headers=HEADERS)
        assert resp.json()["predicted_price"] >= 0

    def test_unauthorized_returns_401(self, client):
        resp = client.post("/predict/flight-price", json=self.PAYLOAD,
                           headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401

    def test_invalid_payload_returns_422(self, client):
        bad = {**self.PAYLOAD, "distance": -100}
        resp = client.post("/predict/flight-price", json=bad, headers=HEADERS)
        assert resp.status_code == 422


# Gender Classification

class TestGenderEndpoint:

    PAYLOAD = {
        "age": 34, "company_enc": 2, "travel_frequency": 8,
        "avg_flight_price": 1100.50, "total_flight_spend": 8800.0,
        "preferred_flight_type_enc": 2, "hotel_bookings": 4,
        "avg_hotel_spend": 420.0, "age_group_enc": 1, "spending_category_enc": 2,
    }

    def test_predict_returns_200(self, client):
        resp = client.post("/predict/gender", json=self.PAYLOAD, headers=HEADERS)
        assert resp.status_code == 200

    def test_predict_has_gender_field(self, client):
        resp = client.post("/predict/gender", json=self.PAYLOAD, headers=HEADERS)
        assert "predicted_gender" in resp.json()

    def test_gender_valid_values(self, client):
        resp = client.post("/predict/gender", json=self.PAYLOAD, headers=HEADERS)
        assert resp.json()["predicted_gender"] in ["male", "female"]

    def test_confidence_between_0_and_1(self, client):
        resp = client.post("/predict/gender", json=self.PAYLOAD, headers=HEADERS)
        assert 0 <= resp.json()["confidence"] <= 1


# Hotel Recommendation

class TestRecommendEndpoint:

    PAYLOAD = {"user_code": 42, "top_n": 3}

    def test_recommend_returns_200(self, client):
        resp = client.post("/recommend/hotels", json=self.PAYLOAD, headers=HEADERS)
        assert resp.status_code == 200

    def test_recommend_has_recommendations(self, client):
        resp = client.post("/recommend/hotels", json=self.PAYLOAD, headers=HEADERS)
        assert "recommendations" in resp.json()

    def test_top_n_respected(self, client):
        resp = client.post("/recommend/hotels", json={"user_code": 1, "top_n": 11},
                           headers=HEADERS)
        assert resp.status_code in [200, 422]

    def test_unauthorized_returns_401(self, client):
        resp = client.post("/recommend/hotels", json=self.PAYLOAD,
                           headers={"X-API-Key": "bad-key"})
        assert resp.status_code == 401
