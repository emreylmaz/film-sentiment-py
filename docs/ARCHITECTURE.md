# Sistem Mimarisi
# IMDB Sentiment Analizi Projesi

**Versiyon:** 1.0.0  
**Tarih:** 5 Kasım 2025

---

## 1. Genel Bakış

IMDB Sentiment Analizi, modüler bir makine öğrenmesi projesidir. Katmanlı mimari kullanarak veri işleme, model eğitimi ve API servisi sunumu gerçekleştirir.

###1.1 Mimari Prensipler

- **Modülerlik:** Her katman bağımsız çalışabilir
- **Separation of Concerns:** Veri, model, API ayrı
- **Reusability:** Fonksiyonlar tekrar kullanılabilir
- **Testability:** Her modül unit test edilebilir
- **Scalability:** Horizontal ve vertical scaling hazır

---

## 2. Sistem Katmanları

```
┌─────────────────────────────────────────────────────────┐
│                    CLIENT LAYER                          │
│  (Web Apps, Mobile Apps, Data Analysis Tools)           │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP/JSON
                     ▼
┌─────────────────────────────────────────────────────────┐
│                    API LAYER (FastAPI)                   │
│  ┌────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │ /predict   │  │ /health     │  │ /model/info     │ │
│  └────────────┘  └─────────────┘  └─────────────────┘ │
│                                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │      Model Manager (Singleton)                   │   │
│  └─────────────────────────────────────────────────┘   │
└────────────────────┬───────────────┬────────────────────┘
                     │               │
                     ▼               ▼
┌──────────────────────────┐  ┌──────────────────────┐
│   MODEL LAYER            │  │  PREPROCESSOR LAYER  │
│  - ML Models             │  │  - Text Cleaning     │
│  - Logistic Regression   │  │  - TF-IDF            │
│  - Random Forest         │  │  - Vectorization     │
└──────────────────────────┘  └──────────────────────┘
           │                            │
           ▼                            ▼
┌─────────────────────────────────────────────────────────┐
│                   DATA LAYER                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ model.pkl    │  │vectorizer.pkl│  │metadata.json │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Modül Bağımlılık Grafiği

```
config.yaml
    │
    ├─> data_loader.py
    │       │
    │       ├─> preprocessor.py
    │       │       │
    │       │       └─> train_model.py
    │       │               │
    │       │               ├─> evaluate_model.py
    │       │               │       │
    │       │               │       └─> models/
    │       │               │           ├─ model.pkl
    │       │               │           ├─ vectorizer.pkl
    │       │               │           └─ metadata.json
    │       │               │
    │       │               └─> logger.py
    │       │
    │       └─> api/main.py
    │               │
    │               ├─> ModelManager (loads models)
    │               │
    │               └─> FastAPI App
    │
    └─> Docker
            └─> uvicorn
```

---

## 4. Veri Akış Senaryoları

### 4.1 Training Flow

```
[1] CSV Dosyası
    │
    ├─> load_data()
    │       │
    │       └─> validate_data()
    │
[2] DataFrame (50K rows)
    │
    ├─> split_data()
    │
[3] Train (40K) + Test (10K)
    │
    ├─> TextPreprocessor.fit_transform(train)
    │       │
    │       ├─> clean_html()
    │       ├─> lowercase()
    │       ├─> remove_special_chars()
    │       └─> TfidfVectorizer.fit_transform()
    │
[4] X_train (40K x 5000) + y_train
    │
    ├─> train_logistic_regression()
    │   └─> model.fit(X_train, y_train)
    │
    ├─> train_random_forest()
    │   └─> model.fit(X_train, y_train)
    │
[5] Trained Models
    │
    ├─> evaluate_model()
    │   ├─> model.predict(X_test)
    │   ├─> calculate_metrics()
    │   └─> confusion_matrix
    │
[6] Metrics + Best Model Selection
    │
    └─> save_model()
        ├─> pickle.dump(model, "model.pkl")
        ├─> preprocessor.save("vectorizer.pkl")
        └─> json.dump(metadata, "metadata.json")
```

### 4.2 Inference Flow

```
[1] HTTP Request
    │
    POST /predict
    {
      "text": "This movie was great!"
    }
    │
[2] FastAPI Endpoint
    │
    ├─> Pydantic validation
    │   └─> 10-5000 chars check
    │
[3] ModelManager.predict(text)
    │
    ├─> preprocessor.transform([text])
    │   ├─> clean_html()
    │   ├─> lowercase()
    │   ├─> remove_special_chars()
    │   └─> TfidfVectorizer.transform()
    │
[4] X_vector (1 x 5000)
    │
    ├─> model.predict(X_vector)
    │   └─> "positive"
    │
    ├─> model.predict_proba(X_vector)
    │   └─> [0.08, 0.92]
    │
[5] Result
    │
    └─> HTTP Response
        {
          "sentiment": "positive",
          "confidence": 0.92,
          "prediction_time_ms": 23
        }
```

---

## 5. Design Patterns

### 5.1 Singleton Pattern (ModelManager)

**Amaç:** Model ve preprocessor'ı bir kez yükle, tüm isteklerde paylaş.

```python
class ModelManager:
    _instance = None
    _model = None
    _preprocessor = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_model(self):
        if self._model is None:
            self._model = pickle.load(...)
```

**Faydalar:**
- Memory efficiency
- Tek kez yükleme maliyeti
- Thread-safe (GIL sayesinde)

### 5.2 Strategy Pattern (Model Selection)

**Amaç:** Farklı model algoritmalarını değiştirilebilir şekilde kullan.

```python
def train_logistic_regression(X, y):
    return LogisticRegression().fit(X, y)

def train_random_forest(X, y):
    return RandomForestClassifier().fit(X, y)

# Strategy selection
models = {
    'lr': train_logistic_regression,
    'rf': train_random_forest
}

model = models[config['model_type']](X, y)
```

### 5.3 Pipeline Pattern (Data Processing)

**Amaç:** Veri işleme adımlarını sıralı şekilde uygula.

```python
text → clean_html → lowercase → remove_chars → vectorize → model
```

---

## 6. Configuration Management

### 6.1 Config Dosyası (config.yaml)

**Merkezi konfigürasyon:**

```yaml
data:
  raw_path: "data/IMDB Dataset.csv"
  test_size: 0.2

preprocessing:
  max_features: 5000
  ngram_range: [1, 2]

models:
  logistic_regression:
    C: 1.0
    max_iter: 1000
```

### 6.2 Environment Variables

**Docker deployment için:**

```bash
PORT=8000
LOG_LEVEL=INFO
MODEL_PATH=models/model.pkl
```

---

## 7. Error Handling Stratejisi

### 7.1 Hata Tipleri

| Hata Tipi | HTTP Code | Handling |
|-----------|-----------|----------|
| **Validation Error** | 422 | Pydantic otomatik |
| **Model Not Found** | 503 | Startup check |
| **Prediction Error** | 500 | Try-catch + log |
| **Invalid Input** | 400 | Custom validation |

### 7.2 Error Flow

```
Request
    │
    ├─> Validation Error? → 422 + detail
    │
    ├─> Model Not Loaded? → 503 + "Model not loaded"
    │
    ├─> Prediction Error? → 500 + logged error
    │
    └─> Success → 200 + prediction
```

---

## 8. Logging Stratejisi

### 8.1 Log Levels

- **DEBUG:** Detaylı debugging bilgisi
- **INFO:** Genel bilgi (request, model load)
- **WARNING:** Uyarılar (missing metadata)
- **ERROR:** Hatalar (prediction fail)
- **CRITICAL:** Kritik hatalar (model load fail)

### 8.2 Log Format

```
2025-11-05 14:30:15 - api.main - INFO - Model yüklendi: models/model.pkl
2025-11-05 14:30:20 - api.main - INFO - Tahmin: positive (güven: 0.92)
2025-11-05 14:30:25 - api.main - ERROR - Prediction error: ...
```

### 8.3 Log Rotation

```python
# logger.py'de
RotatingFileHandler(
    filename="logs/app.log",
    maxBytes=1GB,
    backupCount=7  # 7 gün
)
```

---

## 9. Security Considerations

### 9.1 Input Validation

- **Length check:** 10-5000 karakter
- **XSS protection:** HTML tags cleaned
- **Injection protection:** No SQL (dosya sistemi kullanımı)

### 9.2 Rate Limiting (Future)

```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)

@app.post("/predict")
@limiter.limit("100/minute")
async def predict(...):
    ...
```

### 9.3 HTTPS

Production'da HTTPS zorunlu (reverse proxy ile).

---

## 10. Scalability Strategy

### 10.1 Horizontal Scaling

```
         Load Balancer
              │
    ┌─────────┼─────────┐
    │         │         │
  API-1     API-2     API-3
    │         │         │
    └────Shared Model Storage (S3/GCS)
```

### 10.2 Vertical Scaling

- CPU: Vectorization optimizasyonu
- RAM: Model quantization
- Disk: Model compression

### 10.3 Caching (Future)

```python
# Redis caching
@cache(ttl=3600)
def predict(text):
    return model.predict(text)
```

---

## 11. Deployment Mimarisi

### 11.1 Local Development

```
Developer Machine
    │
    ├─> venv (Python 3.10)
    │   └─> pip install -r requirements.txt
    │
    └─> uvicorn --reload
```

### 11.2 Docker Deployment

```
Docker Host
    │
    ├─> docker build
    │   └─> Python 3.10 slim
    │       ├─> requirements.txt
    │       ├─> NLTK data
    │       └─> app code
    │
    └─> docker run
        └─> uvicorn (port 8000)
```

### 11.3 Cloud Deployment

```
GitHub Repo
    │
    ├─> git push
    │
    └─> Render/Heroku
        ├─> Auto build
        ├─> Auto deploy
        └─> Public URL
```

---

## 12. Monitoring ve Observability

### 12.1 Metrics (Future)

**Prometheus Metrics:**
- `request_count`
- `request_duration_seconds`
- `prediction_confidence_distribution`
- `error_count`

### 12.2 Health Checks

```python
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model_manager.is_loaded,
        "uptime": get_uptime()
    }
```

### 12.3 Alerting (Future)

- Model accuracy drop
- High error rate
- High latency

---

## 13. CI/CD Pipeline (Future)

```
GitHub Push
    │
    ├─> GitHub Actions
    │   ├─> Lint (flake8, black)
    │   ├─> Test (pytest)
    │   ├─> Build Docker
    │   └─> Push to Registry
    │
    └─> Auto Deploy
        └─> Render/Heroku
```

---

## 14. Database Schema (Not Applicable)

Bu projede veritabanı kullanılmıyor. Tüm veriler dosya sisteminde:

```
models/
├── model.pkl          # Binary (pickle)
├── vectorizer.pkl     # Binary (pickle)
└── metadata.json      # JSON
```

**Future:** Model versiyonlama için PostgreSQL/MongoDB kullanılabilir.

---

## 15. API Versiyonlama

### v1.0.0 (Current)

- Binary sentiment (positive/negative)
- Single prediction
- No authentication

### v1.1.0 (Planned)

- Batch prediction
- API key authentication
- Rate limiting

### v2.0.0 (Future)

- Multi-class sentiment (1-5 stars)
- Multi-language support
- WebSocket support

---

## 16. Testing Stratejisi

### 16.1 Test Piramidi

```
        /\
       /  \  E2E Tests (5%)
      /____\
     /      \  Integration Tests (15%)
    /________\
   /          \  Unit Tests (80%)
  /______________\
```

### 16.2 Test Coverage

- **Unit Tests:** >80%
- **Integration Tests:** API endpoints
- **E2E Tests:** Docker container

---

## 17. Performance Optimization

### 17.1 Model Optimization

- **Vocabulary size:** 5000 (vs 10000) → -50% memory
- **Sparse matrices:** scipy.sparse → memory efficient
- **Model choice:** LogisticRegression → fast inference

### 17.2 API Optimization

- **Singleton pattern:** Model bir kez yüklenir
- **Async FastAPI:** Concurrent requests
- **Uvicorn workers:** Multi-process

### 17.3 Benchmarks

| Metrik | Target | Actual |
|--------|--------|--------|
| **Response Time** | <100ms | ~30ms |
| **Throughput** | >100 req/s | ~500 req/s |
| **Memory** | <2GB | ~1.5GB |

---

**Doküman Hazırlayan:** AI Yazılım Mühendisi  
**Son Güncelleme:** 5 Kasım 2025


