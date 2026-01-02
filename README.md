# 🎬 IMDB Film Sentiment Analizi Projesi

Film yorumları üzerinde yapay zeka destekli sentiment analizi yapan, **FastAPI** ile servis edilen ve **Docker** ile dağıtılabilen kapsamlı bir makine öğrenmesi projesi.

## 📋 Proje Özeti

Bu proje, 50,000 IMDB film yorumu üzerinde sentiment analizi (pozitif/negatif sınıflandırma) gerçekleştirir. TF-IDF vektörizasyonu ve makine öğrenmesi modelleri (Logistic Regression, Random Forest) kullanılarak %85+ doğruluk oranı hedeflenmiştir.

### 🎯 Özellikler

- ✅ 50,000 IMDB film yorumu sentiment analizi
- ✅ TF-IDF ile metin vektörizasyonu
- ✅ Multiple model karşılaştırma (Logistic Regression, Random Forest)
- ✅ **JWT Authentication & Authorization**
- ✅ **Redis JWT Blacklist** (Gerçek logout sistemi - Best Practice)
- ✅ **MongoDB ile Prompt Logging**
- ✅ **Kullanıcı Yönetimi (Register/Login)**
- ✅ FastAPI ile REST API servisi
- ✅ Docker containerization
- ✅ Kapsamlı test coverage
- ✅ Türkçe dokümantasyon
- ✅ Agent-friendly proje yapısı
- ✅ **Next.js entegrasyona hazır**

## 🏗️ Proje Yapısı

```
film-sentiment-py/
├── data/                      # Veri dosyaları
│   └── IMDB Dataset.csv
├── src/                       # Kaynak kod
│   ├── data_loader.py         # Veri yükleme
│   ├── preprocessor.py        # Metin ön işleme
│   ├── train_model.py         # Model eğitimi
│   ├── evaluate_model.py      # Model değerlendirme
│   └── utils/
│       └── logger.py          # Loglama sistemi
├── api/                       # FastAPI servisi
│   ├── main.py                # Ana FastAPI app
│   ├── auth.py                # Auth endpoints
│   ├── models.py              # Pydantic models
│   ├── database.py            # MongoDB connection
│   ├── redis_client.py        # Redis connection (JWT blacklist)
│   ├── blacklist.py           # JWT token blacklist service
│   ├── crud.py                # Database operations
│   ├── dependencies.py        # Auth dependencies
│   └── auth_utils.py          # JWT utils
├── models/                    # Eğitilmiş modeller
│   ├── model.pkl
│   ├── vectorizer.pkl
│   └── metadata.json
├── tests/                     # Testler
│   └── test_api.py
├── notebooks/                 # Jupyter notebooks
│   ├── 01_veri_analizi.ipynb
│   └── 02_model_karsilastirma.ipynb
├── reports/                   # Raporlar
│   ├── BRD.md
│   ├── SRS.md
│   └── model_rapor.md
├── docs/                      # Dokümantasyon
│   ├── PROJECT_PLAN.md        # Master proje planı
│   ├── ARCHITECTURE.md        # Sistem mimarisi
│   ├── FEATURES.md            # Feature açıklamaları
│   ├── API_DOCUMENTATION.md   # API kılavuzu
│   ├── AUTHENTICATION_GUIDE.md # Auth rehberi
│   ├── REDIS_BLACKLIST.md     # JWT Blacklist sistemi
│   ├── REDIS_KURULUM.md       # Redis kurulum
│   ├── DOCKER_KULLANIM.md     # Docker kılavuzu
│   ├── ENV_SETUP.md           # Environment variables
│   ├── BASLANGIC_KILAVUZU.md  # Hızlı başlangıç
│   ├── TODO_TRACKING.md       # İlerleme takibi
│   └── CHANGELOG.md           # Versiyon geçmişi
├── config.yaml                # Konfigürasyon
├── requirements.txt           # Python bağımlılıklar
├── Dockerfile                 # Docker image tanımı
└── README.md                  # Bu dosya
```

## 🚀 Hızlı Başlangıç

### Gereksinimler

- Python 3.10+
- pip
- **MongoDB** (local veya MongoDB Atlas)
- **Redis** (JWT blacklist için - opsiyonel ama önerilen)
- (Opsiyonel) Docker

### 1. Kurulum

```bash
# Projeyi klonla
git clone <repo-url>
cd film-sentiment-py

# Virtual environment oluştur
python -m venv venv

# Aktive et (Windows)
venv\Scripts\activate

# Aktive et (Linux/Mac)
source venv/bin/activate

# Bağımlılıkları yükle
pip install -r requirements.txt

# NLTK stopwords indir
python -c "import nltk; nltk.download('stopwords')"

# MongoDB kurulumu (Seçenek 1: Local)
# MongoDB'yi indirin: https://www.mongodb.com/try/download/community

# veya (Seçenek 2: Docker ile)
docker run -d -p 27017:27017 --name mongodb mongo:latest

# veya (Seçenek 3: MongoDB Atlas - Cloud)
# https://www.mongodb.com/cloud/atlas/register adresinden free cluster oluşturun

# Redis kurulumu (JWT Blacklist için - önerilir)
# Docker ile (en kolay)
docker run -d -p 6379:6379 --name redis-blacklist redis:7-alpine

# veya Mac: brew install redis && brew services start redis
# veya Linux: sudo apt-get install redis-server

# Redis kurulumu için detaylı bilgi: docs/REDIS_KURULUM.md

# Environment variables ayarla
# docs/ENV_SETUP.md dosyasına bakın ve .env oluşturun
```

### 2. Model Eğitimi

```bash
# Modeli eğit
python src/train_model.py
```

Bu komut:
- IMDB dataset'ini yükler
- Veriyi train/test olarak ayırır (%80/%20)
- TF-IDF vektörizasyonu yapar
- Logistic Regression ve Random Forest modellerini eğitir
- En iyi modeli `models/` klasörüne kaydeder

**Çıktı:**
```
models/
├── model.pkl           # Eğitilmiş model
├── vectorizer.pkl      # TF-IDF vectorizer
└── metadata.json       # Model metrikleri
```

### 3. API Servisi Başlatma

```bash
# FastAPI servisini başlat
uvicorn api.main:app --reload
```

API şu adreste çalışacak: `http://localhost:8000`

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

⚠️ **Önemli:** API artık authentication gerektiriyor! Önce kullanıcı oluşturun ve giriş yapın.

### 4. Authentication ve API Kullanımı

#### a) Kullanıcı Kaydı

```bash
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "emre_yilmaz",
    "email": "emre@example.com",
    "password": "SecurePass123",
    "full_name": "Emre Yılmaz",
    "organization": "AI Research Lab",
    "role": "user"
  }'
```

#### b) Giriş Yapma (Token Alma)

```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=emre_yilmaz&password=SecurePass123"
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

#### c) Sentiment Prediction (Token ile)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <ACCESS_TOKEN>" \
  -d '{"text": "This movie was absolutely fantastic!"}'
```

#### Python ile Authentication

```python
import requests

# 1. Login
login_response = requests.post(
    "http://localhost:8000/auth/login",
    data={"username": "emre_yilmaz", "password": "SecurePass123"}
)
token = login_response.json()["access_token"]

# 2. Prediction
response = requests.post(
    "http://localhost:8000/predict",
    headers={"Authorization": f"Bearer {token}"},
    json={"text": "Great movie!"}
)

print(response.json())
# {"sentiment": "positive", "confidence": 0.92, "prediction_time_ms": 23.5}
```

#### Sağlık Kontrolü (Auth gerektirmez)

```bash
curl http://localhost:8000/health
```

## 🐳 Docker ile Çalıştırma

### Hızlı Başlangıç (Docker Compose - Önerilen)

Tüm sistemi (MongoDB + Redis + API) tek komutla başlatın:

```bash
# Tüm servisleri başlat
docker-compose up -d

# Logları izle
docker-compose logs -f

# Durumu kontrol et
docker-compose ps
```

**Başlayan Servisler:**
| Servis | Port | URL |
|--------|------|-----|
| API | 8000 | http://localhost:8000 |
| MongoDB | 27017 | mongodb://localhost:27017 |
| Redis | 6379 | redis://localhost:6379 |

### Development Mode (Sadece DB'ler)

```bash
# Sadece MongoDB + Redis + UI araçları
docker-compose -f docker-compose.dev.yml up -d

# API'yi local'de çalıştır
uvicorn api.main:app --reload
```

**Development UI Araçları:**
- Redis Commander: http://localhost:8081
- Mongo Express: http://localhost:8082 (admin/admin123)

### Durdurma

```bash
# Servisleri durdur (data korunur)
docker-compose down

# Servisleri durdur + data'yı sil
docker-compose down -v
```

### Tek Container (Legacy)

```bash
# Image oluştur
docker build -t imdb-sentiment-api .

# Container başlat (MongoDB ve Redis ayrı çalışıyor olmalı!)
docker run -d -p 8000:8000 \
  -e MONGO_URL=mongodb://host.docker.internal:27017 \
  -e REDIS_URL=redis://host.docker.internal:6379 \
  --name sentiment-api imdb-sentiment-api
```

📖 **Detaylı Docker Kılavuzu:** [`docs/DOCKER_KULLANIM.md`](docs/DOCKER_KULLANIM.md)

## 📊 Model Performansı

| Model                | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---------------------|----------|-----------|---------|----------|---------|
| Logistic Regression | ~0.88    | ~0.88     | ~0.88   | ~0.88    | ~0.95   |
| Random Forest       | ~0.85    | ~0.85     | ~0.85   | ~0.85    | ~0.92   |

*Not: Gerçek metrikler model eğitimi sonrası `models/metadata.json` dosyasında bulunur.*

## 🧪 Testler

```bash
# Tüm testleri çalıştır
pytest tests/ -v

# Sadece API testleri
pytest tests/test_api.py -v

# Coverage ile
pytest tests/ --cov=src --cov=api
```

## 📚 Dokümantasyon

> **⚠️ Dokümantasyon Güncellemeleri Hakkında**
> 
> Proje değişikliklerinde ilgili dokümanları **MUTLAKA** güncelleyin!
> - Yeni feature → `docs/FEATURES.md` + `docs/PROJECT_PLAN.md`
> - API değişikliği → `docs/API_DOCUMENTATION.md`
> - Mimari değişiklik → `docs/ARCHITECTURE.md` + `docs/PROJECT_PLAN.md`
> - Versiyon → `docs/CHANGELOG.md` + `docs/PROJECT_PLAN.md`
> - Task tamamlama → `docs/TODO_TRACKING.md`
> 
> Detaylı güncelleme kuralları: `docs/PROJECT_PLAN.md` Bölüm 17

Detaylı dokümantasyon için `docs/` klasörüne bakın:

| Dosya | Açıklama |
|-------|----------|
| [PROJECT_PLAN.md](docs/PROJECT_PLAN.md) | 📋 Master proje planı |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | 🏗️ Sistem mimarisi |
| [FEATURES.md](docs/FEATURES.md) | ✨ Feature açıklamaları |
| [API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md) | 🌐 API kullanım kılavuzu |
| [AUTHENTICATION_GUIDE.md](docs/AUTHENTICATION_GUIDE.md) | 🔐 Auth sistemi rehberi |
| [REDIS_BLACKLIST.md](docs/REDIS_BLACKLIST.md) | 🔴 JWT Blacklist sistemi |
| [REDIS_KURULUM.md](docs/REDIS_KURULUM.md) | ⚙️ Redis kurulum kılavuzu |
| [DOCKER_KULLANIM.md](docs/DOCKER_KULLANIM.md) | 🐳 Docker kullanım kılavuzu |
| [ENV_SETUP.md](docs/ENV_SETUP.md) | 🔧 Environment variables |
| [BASLANGIC_KILAVUZU.md](docs/BASLANGIC_KILAVUZU.md) | 🚀 Hızlı başlangıç |
| [TODO_TRACKING.md](docs/TODO_TRACKING.md) | ✅ İlerleme takibi |
| [CHANGELOG.md](docs/CHANGELOG.md) | 📝 Versiyon geçmişi |

### Raporlar

- **[BRD.md](reports/BRD.md)** - Business Requirements Document
- **[SRS.md](reports/SRS.md)** - Software Requirements Specification
- **[model_rapor.md](reports/model_rapor.md)** - Model performans raporu

## 🔧 Konfigürasyon

`config.yaml` dosyasında proje ayarlarını değiştirebilirsiniz:

```yaml
data:
  raw_path: "data/IMDB Dataset.csv"
  test_size: 0.2
  random_state: 42

preprocessing:
  max_features: 5000
  ngram_range: [1, 2]
  min_df: 5
  max_df: 0.8

models:
  logistic_regression:
    C: 1.0
    max_iter: 1000
  
  random_forest:
    n_estimators: 100
    max_depth: 50
```

## 📖 API Endpoints

### POST /predict
Film yorumu için sentiment tahmini yapar.

**Request:**
```json
{
  "text": "This movie was great!"
}
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.89,
  "prediction_time_ms": 15
}
```

### GET /health
Servis sağlık kontrolü.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0",
  "model_type": "LogisticRegression"
}
```

### GET /model/info
Model detayları.

**Response:**
```json
{
  "model_name": "logistic_regression",
  "model_type": "LogisticRegression",
  "version": "1.0.0",
  "training_date": "2025-11-05 14:30:00",
  "metrics": {
    "accuracy": 0.88,
    "f1_score": 0.88
  },
  "vocabulary_size": 5000
}
```

## 🌐 Deployment

### Render

1. GitHub'a push yapın
2. Render dashboard'da "New Web Service" seçin
3. Repository'yi bağlayın
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

### Heroku

```bash
# Procfile oluştur
echo "web: uvicorn api.main:app --host 0.0.0.0 --port \$PORT" > Procfile

# Deploy
heroku create imdb-sentiment-api
git push heroku main
```

## 🔮 Gelecek Geliştirmeler

- [ ] BERT/RoBERTa transformer modelleri
- [ ] Çok sınıflı sentiment (1-5 yıldız)
- [ ] Türkçe film yorumu desteği
- [ ] Batch prediction endpoint
- [ ] Redis caching
- [ ] Prometheus monitoring
- [ ] A/B testing altyapısı

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Commit yapın (`git commit -am 'feat: yeni özellik eklendi'`)
4. Branch'i push edin (`git push origin feature/yeni-ozellik`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje akademik amaçlı geliştirilmiştir.

## 👥 İletişim

**Proje Sahibi:** AI Yazılım Mühendisi: Emre Yılmaz

---

**⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın!**


```bash
.venv\Scripts\activate; python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```