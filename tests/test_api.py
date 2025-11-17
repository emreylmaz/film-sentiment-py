"""
FastAPI Endpoint Testleri

Bu modül, API endpoint'lerinin doğru çalışıp çalışmadığını test eder.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Proje kök dizinini path'e ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

# Test client oluştur
client = TestClient(app)


class TestRootEndpoint:
    """Ana endpoint testleri."""
    
    def test_root_endpoint(self):
        """Ana endpoint'in doğru bilgi döndürdüğünü test eder."""
        response = client.get("/")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"


class TestHealthEndpoint:
    """Health check endpoint testleri."""
    
    def test_health_check(self):
        """Health endpoint'in çalıştığını test eder."""
        response = client.get("/health")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data


class TestPredictionEndpoint:
    """Prediction endpoint testleri."""
    
    def test_predict_positive_sentiment(self):
        """Pozitif sentiment tahmini testi."""
        # Model yüklü olmayabilir, bu durumda 503 dönebilir
        response = client.post(
            "/predict",
            json={"text": "This movie was absolutely fantastic! Great acting and wonderful plot."}
        )
        
        # Model yüklü değilse 503, yüklüyse 200 olmalı
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "sentiment" in data
            assert "confidence" in data
            assert "prediction_time_ms" in data
            assert data["sentiment"] in ["positive", "negative"]
            assert 0 <= data["confidence"] <= 1
    
    def test_predict_negative_sentiment(self):
        """Negatif sentiment tahmini testi."""
        response = client.post(
            "/predict",
            json={"text": "Terrible movie! Waste of time and money. Very disappointing."}
        )
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "sentiment" in data
            assert data["sentiment"] in ["positive", "negative"]
    
    def test_predict_with_empty_text(self):
        """Boş text ile istek testi - hata dönmeli."""
        response = client.post(
            "/predict",
            json={"text": ""}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_with_short_text(self):
        """Çok kısa text ile istek testi - hata dönmeli."""
        response = client.post(
            "/predict",
            json={"text": "Bad"}
        )
        
        assert response.status_code == 422  # Validation error (min 10 karakter)
    
    def test_predict_with_missing_text_field(self):
        """Text field olmadan istek testi - hata dönmeli."""
        response = client.post(
            "/predict",
            json={}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_response_structure(self):
        """Response yapısının doğru olduğunu test eder."""
        response = client.post(
            "/predict",
            json={"text": "This is a moderately good film with decent acting."}
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Required fields
            assert "sentiment" in data
            assert "confidence" in data
            assert "prediction_time_ms" in data
            
            # Type checks
            assert isinstance(data["sentiment"], str)
            assert isinstance(data["confidence"], float)
            assert isinstance(data["prediction_time_ms"], int)


class TestModelInfoEndpoint:
    """Model info endpoint testleri."""
    
    def test_model_info(self):
        """Model info endpoint'in çalıştığını test eder."""
        response = client.get("/model/info")
        
        # Model yüklü değilse 503, yüklüyse 200 olmalı
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            # Metadata varsa bu alanlar olmalı
            assert "model_name" in data or "model_type" in data


class TestEdgeCases:
    """Edge case testleri."""
    
    def test_predict_with_html_tags(self):
        """HTML tagleri içeren metin testi."""
        response = client.post(
            "/predict",
            json={"text": "<br />This movie was <b>great</b>! Highly recommended.<br/>"}
        )
        
        # HTML taglerle bile çalışmalı
        assert response.status_code in [200, 503]
    
    def test_predict_with_special_characters(self):
        """Özel karakterler içeren metin testi."""
        response = client.post(
            "/predict",
            json={"text": "Amazing movie!!! $$$ Worth every penny!!! ⭐⭐⭐⭐⭐"}
        )
        
        assert response.status_code in [200, 503]
    
    def test_predict_with_long_text(self):
        """Uzun metin testi."""
        long_text = "Great movie! " * 200  # Uzun bir metin
        
        response = client.post(
            "/predict",
            json={"text": long_text}
        )
        
        # 5000 karakterden uzunsa 422, değilse 200 veya 503
        if len(long_text) > 5000:
            assert response.status_code == 422
        else:
            assert response.status_code in [200, 503]


# pytest ile çalıştırmak için
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


