# API Dokümantasyonu
# IMDB Sentiment Analizi API

**Versiyon:** 1.0.0  
**Base URL:** `http://localhost:8000`  
**Format:** JSON

---

## Hızlı Başlangıç

### API Başlatma

```bash
# Virtual environment aktive et
source venv/bin/activate  # Windows: venv\Scripts\activate

# API'yi başlat
uvicorn api.main:app --reload

# Artık API şu adreste: http://localhost:8000
```

### Swagger UI

Otomatik interaktif dokümantasyon: **http://localhost:8000/docs**

---

## Endpoints

### 1. Ana Endpoint

#### `GET /`

**Açıklama:** API bilgilerini döndürür.

**Request:** Yok

**Response (200 OK):**
```json
{
  "message": "IMDB Sentiment Analizi API",
  "version": "1.0.0",
  "status": "running",
  "endpoints": {
    "prediction": "/predict",
    "health": "/health",
    "documentation": "/docs"
  }
}
```

**cURL:**
```bash
curl http://localhost:8000/
```

---

### 2. Sentiment Tahmini

#### `POST /predict`

**Açıklama:** Film yorumu için sentiment tahmini yapar.

**Request Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "text": "string (zorunlu, 10-5000 karakter arası)"
}
```

**Response (200 OK):**
```json
{
  "sentiment": "positive" | "negative",
  "confidence": 0.92,
  "prediction_time_ms": 23
}
```

**Response Fields:**
- `sentiment` (string): Tahmin edilen sentiment ("positive" veya "negative")
- `confidence` (float): Güven skoru (0.0-1.0 arası)
- `prediction_time_ms` (integer): Tahmin süresi (milisaniye)

---

#### Hata Durumları

**400 Bad Request:**
```json
{
  "detail": "Text alanı boş olamaz"
}
```

**422 Validation Error:**
```json
{
  "detail": [
    {
      "loc": ["body", "text"],
      "msg": "ensure this value has at least 10 characters",
      "type": "value_error.any_str.min_length"
    }
  ]
}
```

**500 Internal Server Error:**
```json
{
  "detail": "Tahmin sırasında hata oluştu: [error message]"
}
```

**503 Service Unavailable:**
```json
{
  "detail": "Model henüz yüklenmedi"
}
```

---

#### Örnek İstekler

**cURL:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely fantastic! Great acting and wonderful plot."}'
```

**Python (requests):**
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Amazing film with great acting"}
)

result = response.json()
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2f}")
```

**JavaScript (fetch):**
```javascript
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({text: 'Great movie!'})
})
.then(res => res.json())
.then(data => console.log(data));
```

**Python (httpx - async):**
```python
import httpx
import asyncio

async def predict(text):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/predict",
            json={"text": text}
        )
        return response.json()

result = asyncio.run(predict("Wonderful movie!"))
```

---

### 3. Sağlık Kontrolü

#### `GET /health`

**Açıklama:** Servis durumunu ve model yüklenme kontrolünü yapar.

**Request:** Yok

**Response (200 OK):**
```json
{
  "status": "healthy" | "unhealthy",
  "model_loaded": true,
  "model_version": "1.0.0",
  "model_type": "LogisticRegression"
}
```

**Kullanım:**
- Load balancer health checks
- Monitoring sistemleri
- Deployment smoke tests

**cURL:**
```bash
curl http://localhost:8000/health
```

**Python:**
```python
import requests

health = requests.get("http://localhost:8000/health").json()
if health["status"] == "healthy":
    print("✓ Service is healthy")
else:
    print("✗ Service is unhealthy")
```

---

### 4. Model Bilgisi

#### `GET /model/info`

**Açıklama:** Yüklü model hakkında detaylı bilgi döndürür.

**Request:** Yok

**Response (200 OK):**
```json
{
  "model_name": "logistic_regression",
  "model_type": "LogisticRegression",
  "version": "1.0.0",
  "training_date": "2025-11-05 14:30:00",
  "metrics": {
    "accuracy": 0.88,
    "precision": 0.88,
    "recall": 0.88,
    "f1_score": 0.88,
    "roc_auc": 0.95,
    "confusion_matrix": [[4500, 500], [400, 4600]]
  },
  "vocabulary_size": 5000
}
```

**503 Service Unavailable:**
```json
{
  "detail": "Model henüz yüklenmedi"
}
```

**cURL:**
```bash
curl http://localhost:8000/model/info
```

---

### 5. Swagger Dokümantasyon

#### `GET /docs`

**Açıklama:** Swagger UI interaktif dokümantasyon.

Browser'da aç: **http://localhost:8000/docs**

**Özellikler:**
- Tüm endpoint'leri interaktif test et
- Request/Response şemalarını gör
- "Try it out" ile canlı istek gönder

---

### 6. ReDoc Dokümantasyon

#### `GET /redoc`

**Açıklama:** ReDoc alternatif dokümantasyon.

Browser'da aç: **http://localhost:8000/redoc**

**Özellikler:**
- Daha temiz, okunabilir format
- Print-friendly
- Şema görselleştirme

---

## Rate Limiting

**v1.0.0:** Rate limiting yok

**v1.1.0 (Planlanıyor):**
- IP bazlı: 100 request/dakika
- API key bazlı: Tier system

**Limit Aşımı Response (429):**
```json
{
  "detail": "Rate limit exceeded. Try again in 60 seconds."
}
```

---

## Authentication

**v1.0.0:** Authentication yok (public API)

**v1.1.0 (Planlanıyor):**

**API Key Authentication:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "X-API-Key: your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{"text": "Great movie!"}'
```

**401 Unauthorized:**
```json
{
  "detail": "Invalid API key"
}
```

---

## CORS

**CORS ayarları:**
- **Allow Origins:** `*` (development), specific domains (production)
- **Allow Methods:** GET, POST, OPTIONS
- **Allow Headers:** Content-Type, X-API-Key

**JavaScript Cross-Origin İstek:**
```javascript
// CORS otomatik olarak handle edilir
fetch('http://api.example.com/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({text: 'Great movie!'})
})
```

---

## Error Codes

| HTTP Code | Açıklama | Örnek |
|-----------|----------|-------|
| **200** | Success | Tahmin başarılı |
| **400** | Bad Request | Geçersiz input |
| **422** | Validation Error | Text çok kısa |
| **429** | Too Many Requests | Rate limit aşımı (future) |
| **500** | Internal Server Error | Model hatası |
| **503** | Service Unavailable | Model yüklenmedi |

---

## Performance

### Response Times

| Endpoint | Ortalama | 95th Percentile |
|----------|----------|----------------|
| `/predict` | ~30ms | <100ms |
| `/health` | <5ms | <10ms |
| `/model/info` | <5ms | <10ms |

### Throughput

- **Concurrent Requests:** 100+
- **Throughput:** ~500 req/saniye (single instance)
- **Max Payload:** 5000 karakter

---

## Best Practices

### 1. Error Handling

**Her zaman status code kontrol edin:**
```python
response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
elif response.status_code == 422:
    print("Validation error:", response.json()['detail'])
elif response.status_code == 503:
    print("Service unavailable, try again later")
else:
    print(f"Error {response.status_code}: {response.text}")
```

### 2. Retry Logic

**Exponential backoff ile retry:**
```python
import time

def predict_with_retry(text, max_retries=3):
    for i in range(max_retries):
        try:
            response = requests.post(url, json={"text": text}, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if i == max_retries - 1:
                raise
            time.sleep(2 ** i)  # 1, 2, 4 saniye
```

### 3. Batch Processing

**Birden fazla yorum için serial processing:**
```python
texts = ["Review 1", "Review 2", "Review 3"]
results = []

for text in texts:
    response = requests.post(url, json={"text": text})
    if response.status_code == 200:
        results.append(response.json())
```

**v1.1.0'da batch endpoint eklenecek:**
```python
# Future
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={"texts": texts}
)
```

### 4. Connection Pooling

**Session kullanarak connection pooling:**
```python
session = requests.Session()

for text in texts:
    response = session.post(url, json={"text": text})
    result = response.json()
```

---

## Deployment URLs

### Local Development
```
http://localhost:8000
```

### Docker
```
http://localhost:8000
```

### Cloud (Örnek)
```
https://imdb-sentiment-api.onrender.com
https://imdb-sentiment-api.herokuapp.com
```

---

## Postman Collection

**Import edilebilir Postman collection:**

```json
{
  "info": {
    "name": "IMDB Sentiment API",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Predict Sentiment",
      "request": {
        "method": "POST",
        "header": [{"key": "Content-Type", "value": "application/json"}],
        "body": {
          "mode": "raw",
          "raw": "{\"text\": \"This movie was great!\"}"
        },
        "url": {
          "raw": "{{base_url}}/predict",
          "host": ["{{base_url}}"],
          "path": ["predict"]
        }
      }
    },
    {
      "name": "Health Check",
      "request": {
        "method": "GET",
        "url": {
          "raw": "{{base_url}}/health",
          "host": ["{{base_url}}"],
          "path": ["health"]
        }
      }
    }
  ],
  "variable": [
    {
      "key": "base_url",
      "value": "http://localhost:8000"
    }
  ]
}
```

---

## SDK / Client Libraries (Future)

### Python Client (Planlanıyor)

```python
from imdb_sentiment import SentimentClient

client = SentimentClient(api_key="your-key")
result = client.predict("Great movie!")
print(result.sentiment, result.confidence)
```

### JavaScript Client (Planlanıyor)

```javascript
import { SentimentClient } from 'imdb-sentiment-js';

const client = new SentimentClient({apiKey: 'your-key'});
const result = await client.predict('Great movie!');
```

---

## Changelog

**API Versiyonları:**
- **v1.0.0 (Current):** Binary sentiment, single prediction
- **v1.1.0 (Planned):** Batch prediction, authentication
- **v2.0.0 (Future):** Multi-class sentiment, multi-language

---

**Hazırlayan:** AI Yazılım Mühendisi  
**Son Güncelleme:** 5 Kasım 2025


