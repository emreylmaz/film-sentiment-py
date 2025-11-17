"""
IMDB Sentiment Analizi FastAPI Servisi

Bu servis, eğitilmiş sentiment analizi modelini REST API olarak sunar.
"""

import os
import pickle
import json
import time
from typing import Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

# Logger
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Modelleri
# ============================================================================

class PredictionRequest(BaseModel):
    """
    Tahmin isteği modeli.
    
    Attributes:
        text: Analiz edilecek film yorumu
    """
    text: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Film yorumu metni (10-5000 karakter arası)"
    )
    
    @validator('text')
    def text_must_not_be_empty(cls, v):
        """Metnin boş olmamasını kontrol eder."""
        if not v or not v.strip():
            raise ValueError('Text alanı boş olamaz')
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "text": "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout."
            }
        }


class PredictionResponse(BaseModel):
    """
    Tahmin yanıtı modeli.
    
    Attributes:
        sentiment: Tahmin edilen sentiment (positive/negative)
        confidence: Tahmin güven skoru (0-1 arası)
        prediction_time_ms: Tahmin süresi (milisaniye)
    """
    sentiment: str = Field(..., description="Tahmin edilen sentiment")
    confidence: float = Field(..., ge=0, le=1, description="Güven skoru")
    prediction_time_ms: int = Field(..., description="Tahmin süresi (ms)")
    
    class Config:
        schema_extra = {
            "example": {
                "sentiment": "positive",
                "confidence": 0.92,
                "prediction_time_ms": 23
            }
        }


class HealthResponse(BaseModel):
    """
    Sağlık kontrolü yanıtı.
    
    Attributes:
        status: Servis durumu
        model_loaded: Model yüklenmiş mi
        model_version: Model versiyonu
    """
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    model_type: Optional[str] = None


# ============================================================================
# Model Yöneticisi (Singleton Pattern)
# ============================================================================

class ModelManager:
    """
    Model ve preprocessor yöneticisi.
    
    Singleton pattern kullanarak model ve preprocessor'ı bir kez yükler.
    """
    
    _instance = None
    _model = None
    _preprocessor = None
    _metadata = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def load_model(self, model_path: str = "models/model.pkl"):
        """Model dosyasını yükler."""
        if self._model is None:
            logger.info(f"Model yükleniyor: {model_path}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
            
            with open(model_path, 'rb') as f:
                self._model = pickle.load(f)
            
            logger.info("✓ Model başarıyla yüklendi")
        
        return self._model
    
    def load_preprocessor(self, preprocessor_path: str = "models/vectorizer.pkl"):
        """Preprocessor dosyasını yükler."""
        if self._preprocessor is None:
            logger.info(f"Preprocessor yükleniyor: {preprocessor_path}")
            
            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(f"Preprocessor dosyası bulunamadı: {preprocessor_path}")
            
            with open(preprocessor_path, 'rb') as f:
                self._preprocessor = pickle.load(f)
            
            logger.info("✓ Preprocessor başarıyla yüklendi")
        
        return self._preprocessor
    
    def load_metadata(self, metadata_path: str = "models/metadata.json"):
        """Metadata dosyasını yükler."""
        if self._metadata is None:
            logger.info(f"Metadata yükleniyor: {metadata_path}")
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self._metadata = json.load(f)
                logger.info("✓ Metadata başarıyla yüklendi")
            else:
                logger.warning(f"Metadata dosyası bulunamadı: {metadata_path}")
                self._metadata = {}
        
        return self._metadata
    
    def predict(self, text: str) -> dict:
        """
        Metin için sentiment tahmini yapar.
        
        Args:
            text: Analiz edilecek metin
            
        Returns:
            Tahmin sonuçları dictionary'si
        """
        start_time = time.time()
        
        # Preprocessor ile vektörize et
        vector = self._preprocessor.transform([text])
        
        # Tahmin yap
        prediction = self._model.predict(vector)[0]
        
        # Confidence score hesapla
        if hasattr(self._model, 'predict_proba'):
            proba = self._model.predict_proba(vector)[0]
            # Pozitif sınıfın olasılığını al
            confidence = float(proba[1] if prediction == 'positive' else proba[0])
        elif hasattr(self._model, 'decision_function'):
            decision = self._model.decision_function(vector)[0]
            # Sigmoid ile olasılığa çevir
            confidence = float(1 / (1 + np.exp(-abs(decision))))
        else:
            confidence = 1.0  # Default
        
        prediction_time = int((time.time() - start_time) * 1000)  # ms
        
        return {
            'sentiment': prediction,
            'confidence': confidence,
            'prediction_time_ms': prediction_time
        }
    
    @property
    def is_loaded(self) -> bool:
        """Model ve preprocessor yüklenmiş mi kontrol eder."""
        return self._model is not None and self._preprocessor is not None
    
    @property
    def metadata(self) -> dict:
        """Metadata'yı döndürür."""
        return self._metadata or {}


# ============================================================================
# FastAPI Uygulaması
# ============================================================================

# App oluştur
app = FastAPI(
    title="IMDB Sentiment Analizi API",
    description="Film yorumları için sentiment analizi servisi. "
                "Pozitif veya negatif sentiment tahmini yapar.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware ekle
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production'da spesifik origin'ler belirtilmeli
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model manager
model_manager = ModelManager()


# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Uygulama başlangıcında çalışır.
    Model ve preprocessor'ı yükler.
    """
    logger.info("=" * 60)
    logger.info("FastAPI Servisi Başlatılıyor...")
    logger.info("=" * 60)
    
    try:
        # Model ve preprocessor yükle
        model_manager.load_model()
        model_manager.load_preprocessor()
        model_manager.load_metadata()
        
        logger.info("✓ Tüm bileşenler başarıyla yüklendi")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"✗ Başlatma hatası: {str(e)}")
        raise


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Ana"])
async def root():
    """
    Ana endpoint - API bilgilerini döndürür.
    """
    return {
        "message": "IMDB Sentiment Analizi API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "prediction": "/predict",
            "health": "/health",
            "documentation": "/docs"
        }
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Tahmin"])
async def predict_sentiment(request: PredictionRequest):
    """
    Film yorumu için sentiment tahmini yapar.
    
    - **text**: Analiz edilecek film yorumu (10-5000 karakter)
    
    Returns:
        - **sentiment**: Tahmin edilen sentiment (positive/negative)
        - **confidence**: Güven skoru (0-1 arası)
        - **prediction_time_ms**: Tahmin süresi (milisaniye)
    
    Örnek:
        ```json
        {
            "text": "This movie was absolutely fantastic!"
        }
        ```
    """
    try:
        # Model yüklü mü kontrol et
        if not model_manager.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model henüz yüklenmedi"
            )
        
        # Tahmin yap
        result = model_manager.predict(request.text)
        
        logger.info(f"Tahmin: {result['sentiment']} (güven: {result['confidence']:.2f})")
        
        return result
        
    except Exception as e:
        logger.error(f"Tahmin hatası: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tahmin sırasında hata oluştu: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse, tags=["Sistem"])
async def health_check():
    """
    Servis sağlık kontrolü.
    
    Model yüklenmiş mi ve servis çalışıyor mu kontrol eder.
    """
    metadata = model_manager.metadata
    
    return {
        "status": "healthy" if model_manager.is_loaded else "unhealthy",
        "model_loaded": model_manager.is_loaded,
        "model_version": metadata.get("version"),
        "model_type": metadata.get("model_type")
    }


@app.get("/model/info", tags=["Model"])
async def model_info():
    """
    Model hakkında detaylı bilgi döndürür.
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model henüz yüklenmedi"
        )
    
    metadata = model_manager.metadata
    
    return {
        "model_name": metadata.get("model_name"),
        "model_type": metadata.get("model_type"),
        "version": metadata.get("version"),
        "training_date": metadata.get("training_date"),
        "metrics": metadata.get("metrics", {}),
        "vocabulary_size": metadata.get("vocabulary_size")
    }


# ============================================================================
# Ana Çalıştırma
# ============================================================================

if __name__ == "__main__":
    # Servisi başlat
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


