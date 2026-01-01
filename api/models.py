"""
API için Pydantic model tanımlamaları.

Bu modül, authentication, kullanıcı yönetimi ve prompt logging 
için gerekli veri modellerini içerir.
"""

from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional
from datetime import datetime


# ==================== User Models ====================

class UserBase(BaseModel):
    """
    Kullanıcı için temel alanlar.
    Create ve Response modelleri için base class.
    """
    username: str = Field(..., min_length=3, max_length=50, 
                          description="Kullanıcı adı (3-50 karakter)")
    email: EmailStr = Field(..., description="Geçerli email adresi")
    full_name: str = Field(..., min_length=2, max_length=100,
                           description="Kullanıcının tam adı")
    organization: Optional[str] = Field(None, max_length=100,
                                       description="Organizasyon/Şirket adı (opsiyonel)")
    role: str = Field(default="user", description="Kullanıcı rolü: user, admin, analyst")
    
    @field_validator('role')
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Role alanının geçerli değerlerden biri olmasını sağlar."""
        allowed_roles = ['user', 'admin', 'analyst']
        if v not in allowed_roles:
            raise ValueError(f"Role must be one of {allowed_roles}")
        return v
    
    @field_validator('username')
    @classmethod
    def validate_username(cls, v: str) -> str:
        """Username'in alphanumeric ve underscore içermesini sağlar."""
        if not v.replace('_', '').isalnum():
            raise ValueError("Username can only contain letters, numbers, and underscores")
        return v.lower()


class UserCreate(UserBase):
    """
    Yeni kullanıcı oluşturma için model.
    Password alanını ekler.
    """
    password: str = Field(..., min_length=8, max_length=100,
                         description="Şifre (minimum 8 karakter)")
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Şifre güvenliği kontrolü."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        if not any(c.isalpha() for c in v):
            raise ValueError("Password must contain at least one letter")
        return v


class UserInDB(UserBase):
    """
    Veritabanında saklanan kullanıcı modeli.
    Hashed password ve metadata içerir.
    """
    id: str = Field(..., description="MongoDB ObjectId (string format)")
    hashed_password: str = Field(..., description="Bcrypt ile hashlenmiş şifre")
    created_at: datetime = Field(..., description="Hesap oluşturma tarihi")
    is_active: bool = Field(default=True, description="Hesap aktif mi?")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "username": "emre_yilmaz",
                "email": "emre@example.com",
                "full_name": "Emre Yılmaz",
                "organization": "AI Research Lab",
                "role": "analyst",
                "hashed_password": "$2b$12$...",
                "created_at": "2025-11-23T10:30:00",
                "is_active": True
            }
        }


class UserResponse(UserBase):
    """
    API response'unda dönen kullanıcı bilgileri.
    Hassas bilgiler (password) içermez.
    """
    id: str = Field(..., description="Kullanıcı ID")
    created_at: datetime = Field(..., description="Hesap oluşturma tarihi")
    is_active: bool = Field(..., description="Hesap aktif mi?")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "username": "emre_yilmaz",
                "email": "emre@example.com",
                "full_name": "Emre Yılmaz",
                "organization": "AI Research Lab",
                "role": "analyst",
                "created_at": "2025-11-23T10:30:00",
                "is_active": True
            }
        }


# ==================== Authentication Models ====================

class Token(BaseModel):
    """
    JWT token response modeli.
    Login sonrası dönen token bilgisi.
    """
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token tipi (her zaman 'bearer')")
    
    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer"
            }
        }


class TokenData(BaseModel):
    """
    JWT token içindeki payload bilgisi.
    Token decode edildiğinde kullanılır.
    """
    username: Optional[str] = Field(None, description="Token'a ait kullanıcı adı")
    
    class Config:
        json_schema_extra = {
            "example": {
                "username": "emre_yilmaz"
            }
        }


class LoginRequest(BaseModel):
    """
    Login isteği için alternatif model.
    OAuth2PasswordRequestForm yerine JSON body kullanımı için.
    """
    username: str = Field(..., description="Kullanıcı adı")
    password: str = Field(..., description="Şifre")
    
    class Config:
        json_schema_extra = {
            "example": {
                "username": "emre_yilmaz",
                "password": "securepass123"
            }
        }


# ==================== Prompt Log Models ====================

class PromptLogCreate(BaseModel):
    """
    Yeni prompt log kaydı oluşturma modeli.
    Sentiment prediction sonucu kaydedilir.
    """
    text: str = Field(..., description="Kullanıcının gönderdiği film yorumu metni")
    sentiment: str = Field(..., description="Model tarafından tahmin edilen sentiment (positive/negative)")
    confidence: float = Field(..., ge=0.0, le=1.0, 
                             description="Tahmin güven skoru (0-1 arası)")
    prediction_time_ms: float = Field(..., ge=0, 
                                     description="Tahmin süresi (milisaniye)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "This movie was absolutely fantastic! Great acting and plot.",
                "sentiment": "positive",
                "confidence": 0.95,
                "prediction_time_ms": 23.5
            }
        }


class PromptLogInDB(PromptLogCreate):
    """
    Veritabanında saklanan prompt log modeli.
    Kullanıcı bilgileri ve metadata eklenir.
    """
    id: str = Field(..., description="MongoDB ObjectId (string format)")
    user_id: str = Field(..., description="Kullanıcı ID (MongoDB ObjectId)")
    username: str = Field(..., description="Kullanıcı adı")
    timestamp: datetime = Field(..., description="Log oluşturma zamanı (UTC)")
    ip_address: Optional[str] = Field(None, description="İstek yapan IP adresi")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "507f1f77bcf86cd799439012",
                "user_id": "507f1f77bcf86cd799439011",
                "username": "emre_yilmaz",
                "text": "This movie was absolutely fantastic!",
                "sentiment": "positive",
                "confidence": 0.95,
                "prediction_time_ms": 23.5,
                "timestamp": "2025-11-23T10:35:00",
                "ip_address": "192.168.1.100"
            }
        }


class PromptLogResponse(BaseModel):
    """
    Prompt log sorgusu response modeli.
    Kullanıcıya dönen log bilgisi.
    """
    id: str
    text: str
    sentiment: str
    confidence: float
    prediction_time_ms: float
    timestamp: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "507f1f77bcf86cd799439012",
                "text": "This movie was absolutely fantastic!",
                "sentiment": "positive",
                "confidence": 0.95,
                "prediction_time_ms": 23.5,
                "timestamp": "2025-11-23T10:35:00"
            }
        }


# ==================== Statistics Models ====================

class UserStatsResponse(BaseModel):
    """
    Kullanıcı istatistikleri response modeli.
    Kullanıcının toplam tahmin sayısı ve dağılımı.
    """
    total_predictions: int = Field(..., description="Toplam tahmin sayısı")
    positive_count: int = Field(..., description="Pozitif sentiment sayısı")
    negative_count: int = Field(..., description="Negatif sentiment sayısı")
    average_confidence: float = Field(..., description="Ortalama güven skoru")
    average_prediction_time_ms: float = Field(..., description="Ortalama tahmin süresi")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_predictions": 150,
                "positive_count": 95,
                "negative_count": 55,
                "average_confidence": 0.89,
                "average_prediction_time_ms": 28.3
            }
        }


# Örnek kullanım ve validasyon test
if __name__ == "__main__":
    # User oluşturma test
    try:
        user = UserCreate(
            username="test_user",
            email="test@example.com",
            password="SecurePass123",
            full_name="Test User",
            organization="Test Org",
            role="user"
        )
        print("✓ UserCreate validation başarılı")
        print(f"  Username: {user.username}")
        print(f"  Email: {user.email}")
    except Exception as e:
        print(f"✗ UserCreate validation hatası: {e}")
    
    # Geçersiz password test
    try:
        user_invalid = UserCreate(
            username="test",
            email="test@example.com",
            password="weak",  # Çok kısa
            full_name="Test",
            role="user"
        )
    except Exception as e:
        print(f"✓ Geçersiz password yakalandı: {e}")
    
    # Token test
    token = Token(access_token="test_token_123")
    print(f"✓ Token oluşturuldu: {token.token_type}")
    
    # Prompt log test
    log = PromptLogCreate(
        text="Great movie!",
        sentiment="positive",
        confidence=0.95,
        prediction_time_ms=25.0
    )
    print(f"✓ PromptLogCreate validation başarılı")
    print(f"  Sentiment: {log.sentiment}")
    print(f"  Confidence: {log.confidence}")

