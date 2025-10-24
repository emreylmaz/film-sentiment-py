import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# It's important to import the app object from the module
from api.main import app, API_VERSION, MODEL_NAME

# Create a TestClient instance
client = TestClient(app)

# --- Mocking the Model --- 
# We don't want to load the actual model during tests.
# We will "patch" the model object in the API module.

class MockModel:
    """A mock model that simulates the behavior of the real model."""
    def predict(self, texts):
        # Simple logic: if 'good' is in the text, predict positive (1), else negative (0).
        return [1 if 'good' in text.lower() else 0 for text in texts]

# Use pytest's autouse fixture to patch the model for all tests in this file
@pytest.fixture(autouse=True)
def override_model_dependency():
    """Patch the model object before tests run, and clean up after."""
    with patch('api.main.model', new_callable=MockModel) as mock_model:
        yield mock_model

# --- Test Cases ---

def test_health_check():
    """Test the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}

def test_version_check():
    """Test the /version endpoint."""
    response = client.get("/version")
    assert response.status_code == 200
    assert response.json() == {"version": API_VERSION}

def test_sentiment_prediction_single():
    """Test the /v1/sentiment endpoint with a single text."""
    response = client.post("/v1/sentiment", json={"texts": ["This is a good movie"]})
    assert response.status_code == 200
    data = response.json()
    assert data["model"]["name"] == MODEL_NAME
    assert data["model"]["version"] == API_VERSION
    assert len(data["results"]) == 1
    assert data["results"][0]["label"] == "positive"

def test_sentiment_prediction_multiple():
    """Test the /v1/sentiment endpoint with multiple texts."""
    texts = ["What a bad film", "Such a good story"]
    response = client.post("/v1/sentiment", json={"texts": texts})
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 2
    assert data["results"][0]["label"] == "negative"
    assert data["results"][1]["label"] == "positive"

def test_sentiment_prediction_empty_list():
    """Test the /v1/sentiment endpoint with an empty list of texts."""
    response = client.post("/v1/sentiment", json={"texts": []})
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 0

def test_sentiment_prediction_invalid_input():
    """Test the /v1/sentiment endpoint with an invalid request body."""
    # The model expects a 'texts' field with a list of strings.
    # Sending a different structure should result in a 422 Unprocessable Entity error.
    response = client.post("/v1/sentiment", json={"text": "a single bad text"})
    assert response.status_code == 422 # Unprocessable Entity
