from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
from typing import List

# --- Configuration ---
API_VERSION = "1.0.0"
MODEL_NAME = "sentiment-logistic-regression"
MODEL_PATH = os.getenv("MODEL_PATH", "models/imdb_lr.joblib")

# --- Application State ---
model = None
app = FastAPI(
    title="Film Sentiment API",
    version=API_VERSION,
    description="API for predicting sentiment of film reviews, based on the SRS document."
)

# --- Pydantic Models (Data Contracts) ---
class SentimentRequest(BaseModel):
    texts: List[str]

class SentimentResult(BaseModel):
    label: str
    score: float | None = None

class ModelInfo(BaseModel):
    name: str
    version: str

class SentimentResponse(BaseModel):
    results: List[SentimentResult]
    model: ModelInfo

class HealthResponse(BaseModel):
    status: str

class VersionResponse(BaseModel):
    version: str

# --- Model Loading ---
def load_model():
    """Loads the sentiment analysis model from the specified path."""
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

@app.on_event("startup")
def startup_event():
    """Load the model during application startup."""
    load_model()

# --- API Endpoints ---
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def get_health():
    """Check the health status of the API."""
    return HealthResponse(status="OK")

@app.get("/version", response_model=VersionResponse, tags=["Monitoring"])
def get_version():
    """Get the version of the API."""
    return VersionResponse(version=API_VERSION)

@app.post("/v1/sentiment", response_model=SentimentResponse, tags=["Sentiment Analysis"])
def predict_sentiment(req: SentimentRequest):
    """
    Analyzes a list of texts and returns their sentiment.
    This endpoint aligns with the v1 API contract defined in the SRS.
    """
    if not req.texts:
        return SentimentResponse(results=[], model=ModelInfo(name=MODEL_NAME, version=API_VERSION))

    predictions = model.predict(req.texts)
    
    results = []
    for pred in predictions:
        label = "positive" if pred == 1 else "negative"
        results.append(SentimentResult(label=label))
        
    return SentimentResponse(
        results=results,
        model=ModelInfo(name=MODEL_NAME, version=API_VERSION)
    )
