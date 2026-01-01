# Değişiklik Geçmişi
# IMDB Sentiment Analizi Projesi

Tüm önemli değişiklikler bu dosyada dokümante edilir.

Format: [Semantic Versioning](https://semver.org/lang/tr/)

---

> **⚠️ GÜNCELLEME HATIRLATMASI**
> 
> Bu dosya her versiyon değişikliğinde MUTLAKA güncellenmelidir!
> 
> **Güncelleme Durumları:**
> - ✅ Yeni feature eklendi → [Unreleased] / Eklenenler
> - ✅ Bug düzeltildi → [Unreleased] / Düzeltilenler
> - ✅ Breaking change → [Unreleased] / Değiştirilenler + Not
> - ✅ Release yapıldı → Yeni versiyon bölümü oluştur
> 
> **Diğer Güncellenecek Dosyalar:**
> - docs/PROJECT_PLAN.md (Bölüm 16: Versiyon Bilgileri)
> - src/__init__.py (__version__ değişkeni)
> - README.md (version badge)

---

## [Unreleased]

### Planlanıyor
- BERT model entegrasyonu
- Batch prediction endpoint (`POST /predict/batch`)
- Prometheus metrics
- Web dashboard
- Rate limiting

---

## [2.0.0] - 2026-01-01

### 🚀 Major Release: Authentication & Database Integration

Bu versiyon, projeyi production-ready hale getiren büyük bir güncellemedir.

### ✨ Eklenenler

**Authentication Sistemi:**
- ✅ JWT (JSON Web Token) authentication
- ✅ User registration (`POST /auth/register`)
- ✅ User login (`POST /auth/login`)
- ✅ User logout (`POST /auth/logout`)
- ✅ Protected endpoints (Bearer token)
- ✅ Password hashing (bcrypt)
- ✅ Token expiration (24 saat default)

**Redis JWT Blacklist (Best Practice):**
- ✅ Gerçek logout sistemi (token iptal)
- ✅ Redis ile token blacklist
- ✅ TTL ile otomatik temizleme
- ✅ Unique token ID (jti claim)
- ✅ Graceful degradation (Redis yoksa çalışır)
- ✅ Blacklist istatistikleri (`/health` endpoint)

**MongoDB Entegrasyonu:**
- ✅ Async MongoDB driver (motor)
- ✅ User collection (username, email, password, role, organization)
- ✅ Prompt logging collection (user_id, text, sentiment, confidence, timestamp)
- ✅ Database indexes
- ✅ CRUD operations

**Merkezi Config Sistemi:**
- ✅ `api/config.py` - Settings manager
- ✅ `config.yaml` + `.env` desteği
- ✅ Environment variable override
- ✅ Typed configuration (dataclasses)

**Docker Compose:**
- ✅ `docker-compose.yml` - Production (MongoDB + Redis + API)
- ✅ `docker-compose.dev.yml` - Development (DB'ler + UI araçları)
- ✅ Redis Commander UI (port 8081)
- ✅ Mongo Express UI (port 8082)
- ✅ Health checks
- ✅ Volume persistence

**Yeni API Modülleri:**
- ✅ `api/auth.py` - Authentication endpoints
- ✅ `api/models.py` - Pydantic models (User, Token, PromptLog)
- ✅ `api/database.py` - MongoDB connection
- ✅ `api/redis_client.py` - Redis connection
- ✅ `api/blacklist.py` - JWT blacklist service
- ✅ `api/crud.py` - Database operations
- ✅ `api/dependencies.py` - Auth dependencies
- ✅ `api/auth_utils.py` - JWT utilities
- ✅ `api/config.py` - Merkezi config manager

**Yeni Dokümantasyon:**
- ✅ `docs/REDIS_BLACKLIST.md` - JWT Blacklist sistemi
- ✅ `docs/REDIS_KURULUM.md` - Redis kurulum kılavuzu
- ✅ `docs/DOCKER_KULLANIM.md` - Docker kullanım kılavuzu
- ✅ `docs/ENV_SETUP.md` - Environment variables
- ✅ `docs/AUTHENTICATION_GUIDE.md` - Auth sistemi rehberi
- ✅ `docs/QUICKSTART_AUTH.md` - Hızlı başlangıç

### 🔄 Değiştirilenler
- `api/main.py` - Authentication, MongoDB, Redis entegrasyonu
- `requirements.txt` - Yeni dependencies (motor, redis, passlib, python-jose)
- `Dockerfile` - Redis, MongoDB env vars eklendi
- `README.md` - Docker Compose, Auth dokümantasyonu
- Dokümanlar `docs/` klasörüne taşındı

### 🐛 Düzeltilenler
- bcrypt 5.0.0 → 4.3.0 (passlib uyumluluk)
- Logger path sorunu düzeltildi (çift logs/ path)
- Config circular import sorunu çözüldü

### 📦 Yeni Dependencies
```
motor>=3.3.0,<4.0.0          # Async MongoDB
pymongo>=4.6.0,<5.0.0        # MongoDB driver
python-jose[cryptography]    # JWT
passlib>=1.7.4               # Password hashing
bcrypt>=4.0.0,<5.0.0         # bcrypt backend
redis>=5.0.0,<6.0.0          # Redis client
python-dotenv>=1.0.0         # .env dosyası
```

### ⚠️ Breaking Changes
- API artık authentication gerektiriyor
- `/predict` endpoint'i protected (Bearer token gerekli)
- MongoDB bağlantısı gerekli
- Yeni environment variables (MONGO_URL, REDIS_URL, SECRET_KEY)

### 🔒 Güvenlik
- JWT token authentication
- Password hashing (bcrypt)
- Token blacklist (logout)
- CORS configuration
- Input validation (Pydantic)

---

## [1.0.1] - 2025-11-18

### ✅ Tamamlanan
- **Model Eğitimi:** Model başarıyla eğitildi
  - Logistic Regression: Accuracy %89.05, F1 Score %89.15
  - Random Forest: Accuracy %86.98, F1 Score %87.11
  - En iyi model: Logistic Regression (F1: 0.8915)
  - ROC-AUC: %95.83 (Mükemmel performans)
- **Model Dosyaları:** model.pkl, vectorizer.pkl, metadata.json oluşturuldu

### 🐛 Düzeltilenler
- **Path Sorunları:** Config ve data dosyalarının path hataları düzeltildi
  - Script artık herhangi bir dizinden çalıştırılabilir
  - Config.yaml, data/, models/, logs/ path'leri absolute path'e dönüştürülüyor
- **Model Kaydetme:** Model karşılaştırma ve kaydetme hataları düzeltildi
  - evaluate_model fonksiyonu model_name'i metrics'e ekliyor
  - Model isim mapping tablosu eklendi
  - KeyError: 'unknown' hatası çözüldü

### 📊 Raporlar
- Model performans raporu güncellendi (reports/model_rapor.md)
- Gerçek metrikler eklendi
- TODO tracking güncellendi

### 🎯 Performans
- Tüm hedefler aşıldı:
  - Accuracy: %89.05 > %85 hedef ✅
  - F1 Score: %89.15 > %85 hedef ✅
  - ROC-AUC: %95.83 > %90 hedef ✅
  - Training Time: 29.63 saniye ✅

---

## [1.0.0] - 2025-11-05

### ✨ Eklenenler

**Core Features:**
- IMDB dataset sentiment analizi (50,000 film yorumu)
- TF-IDF metin vektörizasyonu (max_features: 5000, ngram_range: 1-2)
- Logistic Regression modeli
- Random Forest modeli
- Model karşılaştırma ve otomatik en iyi model seçimi
- Model değerlendirme sistemi (Accuracy, Precision, Recall, F1, ROC-AUC)
- Confusion matrix ve classification report

**API:**
- FastAPI REST servisi
- `POST /predict` - Sentiment tahmini endpoint'i
- `GET /health` - Sağlık kontrolü endpoint'i
- `GET /model/info` - Model bilgisi endpoint'i
- `GET /docs` - Swagger UI otomatik dokümantasyon
- `GET /redoc` - ReDoc alternatif dokümantasyon
- Pydantic ile input/output validasyonu
- CORS desteği
- Error handling ve anlamlı hata mesajları

**Data Processing:**
- HTML tag temizleme
- Küçük harfe çevirme
- Özel karakter temizleme
- Stop words kaldırma (NLTK English)
- Stratified train/test split (%80/%20)

**Infrastructure:**
- Docker containerization
- Docker health check
- Structured logging sistemi
- Configuration management (config.yaml)
- Model ve preprocessor kaydetme/yükleme (pickle)
- Metadata yönetimi (JSON)

**Testing:**
- API endpoint testleri (pytest)
- Unit test altyapısı
- Test coverage desteği

**Documentation:**
- Kapsamlı README.md
- Business Requirements Document (BRD.md)
- Software Requirements Specification (SRS.md)
- Model performans raporu template (model_rapor.md)
- Agent dokümantasyonu sistemi:
  - ARCHITECTURE.md - Sistem mimarisi
  - FEATURES.md - Feature açıklamaları (F001-F006)
  - TODO_TRACKING.md - İlerleme takibi
  - API_DOCUMENTATION.md - API kullanım kılavuzu
  - DEVELOPMENT_GUIDE.md - Geliştirici rehberi
  - CHANGELOG.md - Bu dosya
- Jupyter notebook template'leri
- Türkçe docstring'ler ve yorumlar

**Project Structure:**
- Modüler proje yapısı (src/, api/, models/, tests/, docs/, reports/)
- Clean code prensiplerine uygun
- PEP8 standardı
- Type hints kullanımı

### 🔄 Değiştirilenler
- N/A (İlk versiyon)

### 🐛 Düzeltilenler
- N/A (İlk versiyon)

### 🔒 Güvenlik
- Input validation (10-5000 karakter kontrolü)
- XSS koruması (HTML tag temizleme)
- Pydantic ile type-safe validasyon
- HTTPS ready (production için)
- Güvenli model yükleme mekanizması

### 📊 Performans
- <100ms tahmin süresi hedefi
- Singleton pattern ile model caching
- Sparse matrix kullanımı (memory efficiency)
- Optimized TF-IDF parametreleri

---

## [0.1.0] - 2025-11-04

### ✨ Eklenenler
- İlk proje kurulumu
- Proje klasör yapısı oluşturma
- Temel README

---

## Versiyon Stratejisi

**Semantic Versioning (SemVer):**

`MAJOR.MINOR.PATCH`

- **MAJOR:** Backward incompatible API değişiklikleri
- **MINOR:** Yeni özellikler (backward compatible)
- **PATCH:** Bug fixes (backward compatible)

**Örnekler:**
- `1.0.0` → `1.0.1`: Bug fix
- `1.0.0` → `1.1.0`: Yeni feature (batch prediction)
- `1.0.0` → `2.0.0`: Breaking change (API değişikliği)

---

## Gelecek Versiyonlar

### v1.1.0 (Planlanıyor - Q1 2026)

**Hedef:** API iyileştirmeleri

**Features:**
- Batch prediction endpoint
- API key authentication
- Rate limiting (100 req/min)
- Response caching (Redis)
- Async batch processing

**Performance:**
- 50% daha hızlı inference
- Horizontal scaling desteği

### v1.2.0 (Planlanıyor - Q2 2026)

**Hedef:** Monitoring ve observability

**Features:**
- Prometheus metrics
- Grafana dashboard
- Error tracking (Sentry)
- Log aggregation
- A/B testing framework

### v2.0.0 (Planlanıyor - Q3 2026)

**Hedef:** Advanced ML ve multi-language

**Breaking Changes:**
- BERT model (API response formatı değişebilir)
- Multi-class sentiment (1-5 yıldız)

**Features:**
- BERT/RoBERTa transformer modeller
- Türkçe sentiment analizi
- Aspect-based sentiment
- Model explainability (SHAP/LIME)

---

## Katkıda Bulunanlar

- **AI Yazılım Mühendisi** - Initial development, architecture, documentation

---

## Lisans

Bu proje akademik amaçlı geliştirilmiştir.

---

**Not:** Bu dosya [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) formatını takip eder.


