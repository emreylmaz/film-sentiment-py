# Software Requirements Specification (SRS)
# IMDB Film Sentiment Analizi Projesi

**Versiyon:** 1.0  
**Tarih:** 5 Kasım 2025  
**Proje:** IMDB Sentiment Analizi API  
**Durum:** Onaylandı

---

## 1. Giriş

### 1.1 Amaç

Bu doküman, IMDB Film Sentiment Analizi projesinin yazılım gereksinimlerini detaylı olarak tanımlar. Geliştirici, test ve deployment ekipleri için teknik referans dokümanıdır.

### 1.2 Kapsam

**Proje Adı:** IMDB Sentiment Analizi API

**Ürün Özellikleri:**
- Film yorumları için binary sentiment sınıflandırma
- REST API servisi
- Docker containerization
- Automated testing ve CI/CD hazırlığı

**Hedef Kullanıcılar:**
- Film prodüksiyon şirketleri
- Dijital platformlar
- Veri analistleri
- Pazarlama ekipleri

### 1.3 Tanımlar ve Kısaltmalar

| Terim | Açıklama |
|-------|----------|
| API | Application Programming Interface |
| ML | Machine Learning |
| TF-IDF | Term Frequency-Inverse Document Frequency |
| REST | Representational State Transfer |
| JSON | JavaScript Object Notation |
| CORS | Cross-Origin Resource Sharing |
| SRS | Software Requirements Specification |

### 1.4 Referanslar

- [BRD.md](BRD.md) - Business Requirements Document
- [ARCHITECTURE.md](../docs/ARCHITECTURE.md) - Sistem Mimarisi
- [API_DOCUMENTATION.md](../docs/API_DOCUMENTATION.md) - API Dokümantasyonu

---

## 2. Genel Tanımlama

### 2.1 Ürün Perspektifi

IMDB Sentiment Analizi API, bağımsız bir web servisidir. Film platformları ve analiz araçlarına RESTful API üzerinden entegre edilebilir.

**Sistem Bağlamı:**

```
┌─────────────┐       HTTP/JSON        ┌──────────────────┐
│   Client    │ ───────────────────────> │  FastAPI Server  │
│ (Web/Mobile)│ <─────────────────────  │                  │
└─────────────┘                         │  - Model Manager │
                                        │  - Preprocessor  │
                                        │  - ML Models     │
┌─────────────┐                         │                  │
│   Analyst   │ ───────────────────────> │                  │
│   Tools     │                         └──────────────────┘
└─────────────┘                                  │
                                                 │
                                                 ▼
                                        ┌──────────────────┐
                                        │  Models/         │
                                        │  - model.pkl     │
                                        │  - vectorizer.pkl│
                                        └──────────────────┘
```

### 2.2 Ürün Fonksiyonları

1. **Sentiment Tahmini:** Metin girdisi için pozitif/negatif sentiment analizi
2. **API Servisi:** HTTP üzerinden erişilebilir REST endpoints
3. **Model Yönetimi:** Model yükleme, metadata yönetimi
4. **Sağlık İzleme:** Servis durumu ve model kontrolü

### 2.3 Kullanıcı Karakteristikleri

| Kullanıcı Tipi | Teknik Seviye | Beklenti |
|----------------|---------------|----------|
| Veri Analisti | Orta | Python/API bilgisi, tahmin sonuçları |
| Backend Developer | Yüksek | API entegrasyonu, performans |
| Pazarlama Ekibi | Düşük | Web arayüzü (future), rapor |

### 2.4 Kısıtlar

**Teknik Kısıtlar:**
- Python 3.10+ zorunlu
- 2GB+ RAM gereksinimi
- İnternet bağlantısı (cloud deployment için)

**Regülatif Kısıtlar:**
- GDPR uyumlu veri işleme
- Kullanıcı verilerinin saklanmaması

**İş Kısıtları:**
- Sadece İngilizce metin desteği
- Binary classification (pozitif/negatif)
- Single-node deployment (ilk versiyon)

### 2.5 Varsayımlar ve Bağımlılıklar

**Varsayımlar:**
- Model eğitimi tamamlanmış
- IMDB dataset erişilebilir
- Docker runtime mevcut (containerization için)

**Bağımlılıklar:**
- scikit-learn 1.3.0
- FastAPI 0.103.1
- Python 3.10+
- NLTK stopwords data

---

## 3. Fonksiyonel Gereksinimler

### 3.1 Sentiment Tahmini

**FR-3.1.1: Metin İşleme**

**Tanım:** Sistem, kullanıcı tarafından gönderilen metni işleyebilmeli.

**Girdi:**
- `text`: String (10-5000 karakter arası)

**İşlem:**
1. HTML taglerini temizle
2. Küçük harfe çevir
3. Özel karakterleri temizle
4. TF-IDF vektörize et

**Çıktı:**
- Vektörize edilmiş metin

**Öncelik:** Yüksek  
**Durum:** Zorunlu

---

**FR-3.1.2: Model Tahmini**

**Tanım:** Vektörize edilmiş metin için sentiment tahmini yap.

**Girdi:**
- TF-IDF vektör

**İşlem:**
1. Model predict() çağrısı
2. Confidence score hesaplama
3. Response oluşturma

**Çıktı:**
```json
{
  "sentiment": "positive" | "negative",
  "confidence": 0.0-1.0,
  "prediction_time_ms": integer
}
```

**Performans:** <100ms  
**Öncelik:** Yüksek  
**Durum:** Zorunlu

---

### 3.2 API Endpoints

**FR-3.2.1: POST /predict**

**Tanım:** Sentiment tahmini endpoint'i

**HTTP Method:** POST  
**Content-Type:** application/json

**Request Body:**
```json
{
  "text": "string (required, 10-5000 chars)"
}
```

**Response (200 OK):**
```json
{
  "sentiment": "positive",
  "confidence": 0.92,
  "prediction_time_ms": 23
}
```

**Hata Durumları:**
- 400: Geçersiz input
- 422: Validation error
- 500: Internal server error
- 503: Service unavailable

**Rate Limit:** 100 req/min (future)  
**Authentication:** None (v1.0), API key (future)  
**Öncelik:** Kritik

---

**FR-3.2.2: GET /health**

**Tanım:** Servis sağlık kontrolü

**HTTP Method:** GET

**Response (200 OK):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0",
  "model_type": "LogisticRegression"
}
```

**Kullanım:** Load balancer health checks, monitoring  
**Öncelik:** Yüksek

---

**FR-3.2.3: GET /model/info**

**Tanım:** Model detay bilgileri

**HTTP Method:** GET

**Response (200 OK):**
```json
{
  "model_name": "logistic_regression",
  "model_type": "LogisticRegression",
  "version": "1.0.0",
  "training_date": "2025-11-05 14:30:00",
  "metrics": {
    "accuracy": 0.88,
    "f1_score": 0.88,
    "roc_auc": 0.95
  },
  "vocabulary_size": 5000
}
```

**Öncelik:** Orta

---

### 3.3 Model Yönetimi

**FR-3.3.1: Model Yükleme**

**Tanım:** Uygulama başlangıcında model ve preprocessor yüklenmeli.

**Süreç:**
1. `models/model.pkl` yükle
2. `models/vectorizer.pkl` yükle
3. `models/metadata.json` yükle (opsiyonel)
4. Singleton pattern ile cache'le

**Hata Yönetimi:**
- Model dosyası yoksa → HTTP 503
- Yükleme hatası → Log + Exception

**Öncelik:** Kritik

---

**FR-3.3.2: Metadata Yönetimi**

**Tanım:** Model metadata'sı JSON formatında saklanmalı.

**Metadata İçeriği:**
```json
{
  "model_name": "string",
  "model_type": "string",
  "version": "string",
  "training_date": "datetime",
  "metrics": {
    "accuracy": float,
    "precision": float,
    "recall": float,
    "f1_score": float,
    "roc_auc": float
  },
  "config": {},
  "vocabulary_size": integer
}
```

**Öncelik:** Orta

---

## 4. Teknik Olmayan Gereksinimler

### 4.1 Performans Gereksinimleri

**NFR-4.1.1: Response Time**
- Ortalama: <50ms
- 95th percentile: <100ms
- 99th percentile: <200ms

**NFR-4.1.2: Throughput**
- Minimum: 100 req/saniye
- Hedef: 500 req/saniye

**NFR-4.1.3: Kaynak Kullanımı**
- RAM: <2GB
- CPU: <50% (idle), <80% (peak)
- Disk: <500MB (model dahil)

---

### 4.2 Güvenlik Gereksinimleri

**NFR-4.2.1: Input Validation**
- XSS saldırılarına karşı koruma
- SQL injection (veritabanı kullanılmıyor ama best practice)
- Max input length: 5000 karakter

**NFR-4.2.2: HTTPS**
- Production'da HTTPS zorunlu
- TLS 1.2+

**NFR-4.2.3: Rate Limiting**
- IP bazlı: 100 req/min
- Future: API key bazlı tier system

**NFR-4.2.4: Logging**
- Sensitive data loglanmamalı
- Request/response body loglanmamalı
- Sadece metadata (timestamp, endpoint, duration)

---

### 4.3 Güvenilirlik

**NFR-4.3.1: Uptime**
- Hedef: %99.5
- Planlı bakım: Ayda 2 saat

**NFR-4.3.2: Error Handling**
- Graceful degradation
- Anlamlı hata mesajları
- HTTP status kodları standardına uygun

**NFR-4.3.3: Recovery**
- Automatic restart on crash
- Health check ile liveness probe

---

### 4.4 Bakım ve Destek

**NFR-4.4.1: Logging**
- Structured logging (JSON format)
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Log rotation (1GB max, 7 gün)

**NFR-4.4.2: Monitoring**
- Future: Prometheus metrics
- Current: Application logs

**NFR-4.4.3: Deployment**
- Docker containerization
- Environment-based configuration
- Zero-downtime deployment capability (future)

---

### 4.5 Ölçeklenebilirlik

**NFR-4.5.1: Horizontal Scaling**
- Stateless design
- Load balancer ready
- Shared model storage (future: S3, GCS)

**NFR-4.5.2: Vertical Scaling**
- Efficient memory usage
- CPU optimization (vectorization)

---

### 4.6 Kullanılabilirlik

**NFR-4.6.1: API Dokümantasyonu**
- Swagger UI otomatik
- ReDoc otomatik
- Örnek request/response
- Error code açıklamaları

**NFR-4.6.2: Developer Experience**
- Clear error messages
- Consistent response format
- CORS desteği

---

### 4.7 Taşınabilirlik

**NFR-4.7.1: Platform Independence**
- Docker containerization
- Linux, Windows, Mac desteği
- Cloud-agnostic (Render, Heroku, AWS, GCP)

**NFR-4.7.2: Python Version**
- Python 3.10+
- Backward compatibility (3.9 ile test edilmeli)

---

## 5. Veri Gereksinimleri

### 5.1 Veri Modelleri

**Input Model:**
```python
class PredictionRequest(BaseModel):
    text: str = Field(min_length=10, max_length=5000)
```

**Output Model:**
```python
class PredictionResponse(BaseModel):
    sentiment: str  # "positive" or "negative"
    confidence: float  # 0.0-1.0
    prediction_time_ms: int
```

### 5.2 Veri Validasyonu

- Text boş olamaz
- Min 10 karakter
- Max 5000 karakter
- UTF-8 encoding

### 5.3 Veri Gizliliği

- Kullanıcı verileri saklanmaz
- Request/response loglanmaz
- GDPR compliant

---

## 6. Ara Yüz Gereksinimleri

### 6.1 Kullanıcı Ara Yüzü

**v1.0:** Yok (Sadece API)  
**Future:** Web dashboard

### 6.2 Yazılım Ara Yüzleri

**SI-6.2.1: REST API**
- Protocol: HTTP/HTTPS
- Format: JSON
- Authentication: None (v1.0)

**SI-6.2.2: Model Interface**
- scikit-learn compatible
- pickle serialization

### 6.3 Donanım Ara Yüzleri

Yok (Cloud-based deployment)

---

## 7. Test Gereksinimleri

### 7.1 Unit Tests

- Data loader tests
- Preprocessor tests
- Model evaluation tests
- API endpoint tests

**Coverage:** Minimum %80

### 7.2 Integration Tests

- End-to-end API tests
- Model loading tests
- Error handling tests

### 7.3 Performance Tests

- Load testing (100+ concurrent users)
- Stress testing
- Latency benchmarks

### 7.4 Security Tests

- Input validation tests
- XSS attack simulation
- Rate limiting tests

---

## 8. Deployment Gereksinimleri

### 8.1 Container Requirements

**Dockerfile Specifications:**
- Base: python:3.10-slim
- Port: 8000
- Health check: /health endpoint
- Startup command: uvicorn

### 8.2 Environment Variables

```bash
# Opsiyonel environment variables
PORT=8000
HOST=0.0.0.0
LOG_LEVEL=INFO
MODEL_PATH=models/model.pkl
```

### 8.3 Resource Requirements

**Minimum:**
- CPU: 1 core
- RAM: 1GB
- Disk: 500MB

**Önerilen:**
- CPU: 2 cores
- RAM: 2GB
- Disk: 1GB

---

## 9. Dokümantasyon Gereksinimleri

### 9.1 Teknik Dokümantasyon

- ✅ README.md
- ✅ ARCHITECTURE.md
- ✅ API_DOCUMENTATION.md
- ✅ DEVELOPMENT_GUIDE.md

### 9.2 API Dokümantasyonu

- ✅ Swagger UI (/docs)
- ✅ ReDoc (/redoc)
- ✅ Örnek kullanım kodları

### 9.3 Kod Dokümantasyonu

- Türkçe docstrings
- Type hints
- PEP8 uyumlu

---

## 10. Versiyon Yönetimi

### 10.1 Versiyon Numaralandırma

Semantic Versioning (SemVer): MAJOR.MINOR.PATCH

**v1.0.0:**
- MAJOR: Backward incompatible changes
- MINOR: Yeni özellikler (backward compatible)
- PATCH: Bug fixes

### 10.2 Changelog

- ✅ CHANGELOG.md
- Her versiyon için değişiklikler dokümante edilmeli

---

## 11. Bakım ve Destek

### 11.1 Bakım Planı

- Aylık dependency güncellemeleri
- Quarterly model retraining
- Bug fixes: <24 saat
- Feature requests: Sprint basis

### 11.2 Destek

- GitHub Issues
- Dokümantasyon referansı
- Email support (proje sahibi)

---

## 12. Onaylar

| Rol | İsim | İmza | Tarih |
|-----|------|------|-------|
| Teknik Lead | AI Yazılım Mühendisi: Emre Yılmaz | ✓ | 5 Kasım 2025 |
| QA Lead | AI Yazılım Mühendisi: Emre Yılmaz | ✓ | 5 Kasım 2025 |
| Proje Sahibi | AI Yazılım Mühendisi: Emre Yılmaz | ✓ | 5 Kasım 2025 |

---

## 13. Revizyon Geçmişi

| Versiyon | Tarih | Değişiklikler | Yazar |
|----------|-------|---------------|-------|
| 1.0 | 5 Kasım 2025 | İlk versiyon - Complete SRS | AI Yazılım Mühendisi: Emre Yılmaz |

---

**Doküman Sonu**


