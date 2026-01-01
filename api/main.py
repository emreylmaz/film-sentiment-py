"""
IMDB Sentiment Analizi FastAPI Servisi

Bu servis, eğitilmiş sentiment analizi modelini REST API olarak sunar.
JWT authentication ve MongoDB ile prompt logging destekler.
"""

# .env dosyasını en başta yükle (diğer import'lardan önce!)
from dotenv import load_dotenv
load_dotenv()  # .env dosyasını oku

import os

# Merkezi config (yaml + env)
from api.config import settings
import pickle
import json
import time
from typing import Optional
from fastapi import FastAPI, HTTPException, status, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic import field_validator  # Yeni Pydantic API
import uvicorn

# Proje modülleri
from api.database import connect_to_mongo, close_mongo_connection, get_database, create_indexes
from api.redis_client import connect_to_redis, close_redis_connection, get_redis_info  # Redis bağlantısı
from api.blacklist import get_blacklist_stats  # Blacklist istatistikleri
from api.models import UserInDB, PromptLogCreate
from api.dependencies import get_current_user
from api.crud import create_prompt_log
from api.auth import router as auth_router
from motor.motor_asyncio import AsyncIOMotorDatabase

# Logger
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Modelleri (Sentiment Prediction için)
# ============================================================================

class TextInput(BaseModel):
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
    
    @field_validator('text')
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        """Metnin boş olmamasını kontrol eder."""
        if not v or not v.strip():
            raise ValueError('Text alanı boş olamaz')
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout."
            }
        }


class PredictionOutput(BaseModel):
    """
    Tahmin yanıtı modeli.
    
    Attributes:
        sentiment: Tahmin edilen sentiment (positive/negative)
        confidence: Tahmin güven skoru (0-1 arası)
        prediction_time_ms: Tahmin süresi (milisaniye)
    """
    sentiment: str = Field(..., description="Tahmin edilen sentiment")
    confidence: float = Field(..., ge=0, le=1, description="Güven skoru")
    prediction_time_ms: float = Field(..., description="Tahmin süresi (ms)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "sentiment": "positive",
                "confidence": 0.92,
                "prediction_time_ms": 23.5
            }
        }


class ModelInfo(BaseModel):
    """Model bilgileri response modeli."""
    model_type: str
    version: str
    training_date: str
    metrics: dict
    max_features: int
    ngram_range: list


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
        else:
            confidence = 1.0  # Default
        
        prediction_time = (time.time() - start_time) * 1000  # ms
        
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
    description="""
Film yorumları için sentiment analizi servisi.

**Features:**
* JWT Authentication
* Sentiment Prediction (Positive/Negative)
* Prompt Logging
* User Management

**Authentication:**
* Register: POST /auth/register
* Login: POST /auth/login
* Protected endpoints require Bearer token
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware ekle
# Config'den CORS ayarlarını al (env > yaml > default)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.cors.allow_credentials,
    allow_methods=settings.cors.allow_methods,
    allow_headers=settings.cors.allow_headers,
)

# Model manager
model_manager = ModelManager()


# ============================================================================
# Startup & Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Uygulama başlangıcında çalışır.
    Model, preprocessor, MongoDB ve Redis bağlantılarını yükler.
    """
    logger.info("=" * 60)
    logger.info("FastAPI Servisi Başlatılıyor...")
    logger.info("=" * 60)
    
    try:
        # Model ve preprocessor yükle
        model_manager.load_model()
        model_manager.load_preprocessor()
        model_manager.load_metadata()
        
        # MongoDB'ye bağlan
        await connect_to_mongo()
        
        # Index'leri oluştur
        await create_indexes()
        
        # Redis'e bağlan (JWT blacklist için)
        await connect_to_redis()
        
        logger.info("✓ Tüm bileşenler başarıyla yüklendi")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"✗ Başlatma hatası: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """
    Uygulama kapanırken çalışır.
    MongoDB ve Redis bağlantılarını kapatır.
    """
    logger.info("Uygulama kapatılıyor...")
    await close_mongo_connection()
    logger.info("✓ MongoDB bağlantısı kapatıldı")
    await close_redis_connection()
    logger.info("✓ Redis bağlantısı kapatıldı")


# ============================================================================
# Auth Router'ı Dahil Et
# ============================================================================

app.include_router(auth_router)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Ana"])
async def root():
    """
    Ana endpoint - API bilgilerini döndürür.
    """
    return {
        "message": "IMDB Sentiment Analizi API v2.0",
        "status": "running",
        "features": [
            "JWT Authentication",
            "Sentiment Analysis",
            "Prompt Logging",
            "User Management"
        ],
        "endpoints": {
            "auth": {
                "register": "POST /auth/register",
                "login": "POST /auth/login",
                "me": "GET /auth/me (protected)"
            },
            "prediction": "POST /predict (protected)",
            "health": "GET /health",
            "model_info": "GET /model/info",
            "documentation": "/docs"
        }
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Tahmin"])
async def predict_sentiment(
    input_data: TextInput,
    request: Request,
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Film yorumu için sentiment tahmini yapar.
    
    **Protected Endpoint:** JWT token gereklidir.
    
    **Authorization Header:**
    ```
    Authorization: Bearer <access_token>
    ```
    
    **Request Body:**
    ```json
    {
        "text": "This movie was absolutely fantastic!"
    }
    ```
    
    **Response:**
    ```json
    {
        "sentiment": "positive",
        "confidence": 0.92,
        "prediction_time_ms": 23.5
    }
    ```
    
    **Process:**
    1. Authenticate user via JWT token
    2. Predict sentiment using ML model
    3. Log prediction to database
    4. Return prediction result
    """
    try:
        # Model yüklü mü kontrol et
        if not model_manager.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model henüz yüklenmedi"
            )
        
        # Tahmin yap
        result = model_manager.predict(input_data.text)
        
        logger.info(f"Tahmin: {result['sentiment']} (güven: {result['confidence']:.2f}) - User: {current_user.username}")
        
        # Prompt log oluştur
        log_data = PromptLogCreate(
            text=input_data.text,
            sentiment=result['sentiment'],
            confidence=result['confidence'],
            prediction_time_ms=result['prediction_time_ms']
        )
        
        # Log'u veritabanına kaydet
        try:
            ip_address = request.client.host if request.client else None
            await create_prompt_log(db, log_data, current_user.id, current_user.username, ip_address)
        except Exception as log_error:
            # Logging hatası prediction'ı etkilemez
            logger.error(f"Prompt log kaydetme hatası: {log_error}")
        
        return PredictionOutput(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Tahmin hatası: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tahmin sırasında hata oluştu: {str(e)}"
        )


@app.get("/health", tags=["Sistem"])
async def health_check():
    """
    Servis sağlık kontrolü.
    
    Model, MongoDB, Redis durumlarını kontrol eder.
    
    **Returns:**
    - status: Genel sağlık durumu
    - model_loaded: Model yüklenmiş mi
    - database_connected: MongoDB bağlantısı
    - redis_connected: Redis bağlantısı (JWT blacklist için)
    - redis_info: Redis detay bilgileri
    - blacklist_stats: Aktif blacklist sayısı
    """
    metadata = model_manager.metadata
    
    # MongoDB durumu
    db_connected = True
    try:
        db = get_database()
        await db.command("ping")
    except:
        db_connected = False
    
    # Redis durumu ve blacklist stats
    redis_info = await get_redis_info()
    blacklist_stats = await get_blacklist_stats()
    
    # Genel status
    is_healthy = (
        model_manager.is_loaded and 
        db_connected
        # Redis opsiyonel, health'i etkilemez
    )
    
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "model_loaded": model_manager.is_loaded,
        "model_version": metadata.get("version"),
        "model_type": metadata.get("model_type"),
        "database_connected": db_connected,
        "redis_connected": redis_info.get("available", False),
        "redis_info": redis_info,
        "blacklist_stats": blacklist_stats
    }


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
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
    
    return ModelInfo(
        model_type=metadata.get("model_type", "unknown"),
        version=metadata.get("version", "unknown"),
        training_date=metadata.get("training_date", "unknown"),
        metrics=metadata.get("metrics", {}),
        max_features=metadata.get("max_features", 0),
        ngram_range=metadata.get("ngram_range", [])
    )


# ============================================================================
# User Stats Endpoint (Opsiyonel - Analytics için)
# ============================================================================

@app.get("/stats/me", tags=["İstatistikler"])
async def get_my_stats(
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Mevcut kullanıcının sentiment analizi istatistiklerini döndürür.
    
    **Protected Endpoint:** JWT token gereklidir.
    
    **Returns:**
    - Toplam tahmin sayısı
    - Pozitif/negatif dağılımı
    - Ortalama güven skoru
    - Ortalama tahmin süresi
    """
    from api.crud import get_user_statistics
    
    try:
        stats = await get_user_statistics(db, current_user.id)
        logger.info(f"İstatistikler getir ildi: {current_user.username}")
        return stats
    except Exception as e:
        logger.error(f"İstatistik getirme hatası: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="İstatistikler getirilemedi"
        )


# ============================================================================
# Ana Çalıştırma
# ============================================================================

if __name__ == "__main__":
    # Servisi başlat (config'den ayarları al)
    uvicorn.run(
        "main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.debug,
        log_level=settings.api.log_level.lower()
    )
