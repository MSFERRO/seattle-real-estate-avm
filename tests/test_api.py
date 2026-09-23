"""Integration tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from src.api.app import app

SAMPLE_HOUSE = {
    "bedrooms": 3,
    "bathrooms": 2.5,
    "sqft_living": 2200,
    "sqft_lot": 6000,
    "floors": 1.5,
    "waterfront": 0,
    "view": 0,
    "condition": 4,
    "grade": 8,
    "sqft_above": 1700,
    "sqft_basement": 500,
    "yr_built": 1955,
    "yr_renovated": 0,
    "zipcode": 98115,
    "lat": 47.6974,
    "long": -122.313,
    "sqft_living15": 1600,
    "sqft_lot15": 6000
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "app_requests_total" in response.text


def test_predict_validation_error(client):
    invalid_payload = {
        "bedrooms": 3,
        "bathrooms": 2.0,
        "zipcode": 98178
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_single_success(client):
    response = client.post("/predict", json=SAMPLE_HOUSE)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert data["predicted_price"] > 100000
    assert "confidence_interval_90_usd" in data
    assert "predicted_price_formatted" in data


def test_predict_explain_success(client):
    response = client.post("/predict/explain", json=SAMPLE_HOUSE)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert "explanations" in data
    assert len(data["explanations"]) > 0
    first_exp = data["explanations"][0]
    assert "feature" in first_exp
    assert "impact_percentage" in first_exp
    assert "direction" in first_exp


def test_predict_batch_success(client):
    response = client.post("/predict/batch", json={"houses": [SAMPLE_HOUSE, SAMPLE_HOUSE]})
    assert response.status_code == 200
    data = response.json()
    assert data["total_properties"] == 2
    assert len(data["predictions"]) == 2
