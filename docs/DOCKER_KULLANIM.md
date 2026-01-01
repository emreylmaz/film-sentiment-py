# 🐳 Docker ile Çalıştırma Kılavuzu

## 📋 İçindekiler
1. [Gereksinimler](#gereksinimler)
2. [Hızlı Başlangıç](#hızlı-başlangıç)
3. [Development Mode](#development-mode)
4. [Production Mode](#production-mode)
5. [Komutlar Özeti](#komutlar-özeti)
6. [Sorun Giderme](#sorun-giderme)

---

## Gereksinimler

- **Docker Desktop** (Windows/Mac) veya **Docker Engine** (Linux)
- **Docker Compose** (Docker Desktop ile birlikte gelir)

### Docker Kurulumu

**Windows/Mac:**
1. https://www.docker.com/products/docker-desktop/ adresinden Docker Desktop indirin
2. Kurun ve başlatın
3. Sistem tepsisinde Docker simgesi yeşil olmalı

**Linux (Ubuntu):**
```bash
# Docker yükle
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Docker Compose yükle
sudo apt-get install docker-compose-plugin

# Kullanıcıyı docker grubuna ekle (sudo gerekmez)
sudo usermod -aG docker $USER
# Oturumu kapatıp açın
```

### Docker Çalışıyor mu?

```bash
docker --version
# Docker version 24.0.x, build ...

docker-compose --version
# Docker Compose version v2.x.x
```

---

## Hızlı Başlangıç

### Tüm Sistemi Başlat (MongoDB + Redis + API)

```powershell
# 1. Proje dizinine git
cd D:\Development\MasterSWE\film-sentiment-py

# 2. Tüm servisleri başlat
docker-compose up -d

# 3. Logları izle
docker-compose logs -f

# 4. Servislerin durumunu kontrol et
docker-compose ps
```

**Çıktı:**
```
NAME                 STATUS              PORTS
sentiment-api        running (healthy)   0.0.0.0:8000->8000/tcp
sentiment-mongodb    running (healthy)   0.0.0.0:27017->27017/tcp
sentiment-redis      running (healthy)   0.0.0.0:6379->6379/tcp
```

### Servislere Erişim

| Servis | URL | Açıklama |
|--------|-----|----------|
| **API** | http://localhost:8000 | FastAPI Sentiment API |
| **Swagger UI** | http://localhost:8000/docs | API Dokümantasyonu |
| **ReDoc** | http://localhost:8000/redoc | Alternatif API Docs |
| **MongoDB** | mongodb://localhost:27017 | Database |
| **Redis** | redis://localhost:6379 | JWT Blacklist |

### Test Et

```bash
# Health check
curl http://localhost:8000/health

# Response:
{
  "status": "healthy",
  "model_loaded": true,
  "database_connected": true,
  "redis_connected": true
}
```

### Durdur

```bash
# Servisleri durdur (data korunur)
docker-compose down

# Servisleri durdur + tüm data'yı sil
docker-compose down -v
```

---

## Development Mode

Development'ta genellikle **sadece MongoDB ve Redis** Docker'da çalışır, API local'de çalışır (hot reload için).

### 1. Database'leri Başlat

```powershell
# Sadece MongoDB + Redis + UI araçları
docker-compose -f docker-compose.dev.yml up -d
```

**Başlayan servisler:**
- MongoDB: `localhost:27017`
- Redis: `localhost:6379`
- Redis Commander (UI): `http://localhost:8081`
- Mongo Express (UI): `http://localhost:8082` (admin/admin123)

### 2. API'yi Local'de Başlat

```powershell
# Virtual environment aktif et
.venv\Scripts\activate

# API'yi başlat (hot reload ile)
uvicorn api.main:app --reload
```

### 3. UI Araçları ile Database'leri Görüntüle

**Redis Commander:** http://localhost:8081
- Blacklist key'leri görüntüle
- TTL'leri izle
- Key ekle/sil

**Mongo Express:** http://localhost:8082
- Username: `admin`
- Password: `admin123`
- Users ve prompt_logs collection'larını görüntüle

### 4. Durdur

```bash
docker-compose -f docker-compose.dev.yml down
```

---

## Production Mode

### 1. Environment Variables Ayarla

`.env` dosyası oluştur:

```bash
# .env
MONGO_URL=mongodb://mongodb:27017
REDIS_URL=redis://redis:6379
SECRET_KEY=cok-guclu-gizli-anahtar-en-az-32-karakter-olmali
CORS_ORIGINS=https://your-frontend-domain.com
```

### 2. Image Oluştur ve Başlat

```bash
# Image'ı yeniden build et
docker-compose build

# Başlat
docker-compose up -d
```

### 3. Production Önerileri

#### MongoDB Güvenliği
```yaml
# docker-compose.yml içinde
mongodb:
  environment:
    MONGO_INITDB_ROOT_USERNAME: admin
    MONGO_INITDB_ROOT_PASSWORD: VerySecurePassword123!
```

API'de connection string:
```bash
MONGO_URL=mongodb://admin:VerySecurePassword123!@mongodb:27017
```

#### Redis Güvenliği
```yaml
# docker-compose.yml içinde
redis:
  command: redis-server --appendonly yes --requirepass VerySecureRedisPassword
```

API'de connection string:
```bash
REDIS_URL=redis://:VerySecureRedisPassword@redis:6379
```

#### SSL/TLS
Production'da NGINX veya Traefik reverse proxy kullanın.

---

## Komutlar Özeti

### Docker Compose Komutları

| Komut | Açıklama |
|-------|----------|
| `docker-compose up -d` | Tüm servisleri başlat (background) |
| `docker-compose up -d --build` | Image'ları yeniden build et ve başlat |
| `docker-compose down` | Servisleri durdur |
| `docker-compose down -v` | Durdur + volumes sil |
| `docker-compose ps` | Servis durumlarını göster |
| `docker-compose logs -f` | Tüm logları izle |
| `docker-compose logs -f api` | Sadece API loglarını izle |
| `docker-compose restart api` | API'yi yeniden başlat |
| `docker-compose exec api bash` | API container'a bağlan |
| `docker-compose exec mongodb mongosh` | MongoDB shell'e bağlan |
| `docker-compose exec redis redis-cli` | Redis CLI'a bağlan |

### Development Komutları

```powershell
# Development DB'leri başlat
docker-compose -f docker-compose.dev.yml up -d

# UI araçları ile birlikte
docker-compose -f docker-compose.dev.yml up -d

# Durdur
docker-compose -f docker-compose.dev.yml down
```

### Docker Komutları

```bash
# Container'ları listele
docker ps

# Tüm container'ları listele (durmuş olanlar dahil)
docker ps -a

# Image'ları listele
docker images

# Logs
docker logs sentiment-api
docker logs sentiment-mongodb
docker logs sentiment-redis

# Container'a bağlan
docker exec -it sentiment-api bash
docker exec -it sentiment-mongodb mongosh
docker exec -it sentiment-redis redis-cli

# Container istatistikleri
docker stats
```

### Temizlik Komutları

```bash
# Kullanılmayan container'ları sil
docker container prune

# Kullanılmayan image'ları sil
docker image prune

# Kullanılmayan volume'ları sil
docker volume prune

# Tüm kullanılmayanları sil
docker system prune -a
```

---

## Sorun Giderme

### Problem 1: Port Kullanımda

**Hata:**
```
Error: Bind for 0.0.0.0:8000 failed: port is already allocated
```

**Çözüm:**
```bash
# Port'u kim kullanıyor?
# Windows
netstat -ano | findstr :8000

# Port'u değiştir (docker-compose.yml)
ports:
  - "8001:8000"  # Host:Container
```

### Problem 2: MongoDB Bağlanamıyor

**Hata:**
```
pymongo.errors.ServerSelectionTimeoutError: mongodb:27017
```

**Çözüm:**
```bash
# MongoDB çalışıyor mu?
docker-compose ps mongodb

# Logları kontrol et
docker-compose logs mongodb

# Yeniden başlat
docker-compose restart mongodb

# Network kontrolü
docker network ls
docker network inspect film-sentiment-py_sentiment-network
```

### Problem 3: Redis Bağlanamıyor

**Hata:**
```
redis.exceptions.ConnectionError: Error connecting to redis://redis:6379
```

**Çözüm:**
```bash
# Redis çalışıyor mu?
docker-compose ps redis

# Ping test
docker exec sentiment-redis redis-cli ping
# PONG dönmeli

# Yeniden başlat
docker-compose restart redis
```

### Problem 4: API Başlamıyor

**Hata:**
```
sentiment-api exited with code 1
```

**Çözüm:**
```bash
# Logları kontrol et
docker-compose logs api

# Image'ı yeniden build et
docker-compose build api

# Tekrar başlat
docker-compose up -d api
```

### Problem 5: Model Bulunamadı

**Hata:**
```
FileNotFoundError: models/model.pkl
```

**Çözüm:**
```bash
# Model dosyası var mı?
ls models/
# model.pkl, vectorizer.pkl, metadata.json olmalı

# Yoksa eğit
python src/train_model.py

# Tekrar başlat
docker-compose restart api
```

### Problem 6: Volume İzinleri (Linux)

**Hata:**
```
PermissionError: [Errno 13] Permission denied
```

**Çözüm:**
```bash
# Volume dizinlerinin izinlerini düzelt
sudo chown -R $USER:$USER ./logs ./models

# Veya Docker'ın user'ını kullan
sudo chown -R 1000:1000 ./logs ./models
```

---

## Örnek Akış

### İlk Kurulum

```powershell
# 1. Projeye git
cd D:\Development\MasterSWE\film-sentiment-py

# 2. Virtual environment ve bağımlılıklar (local development için)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 3. Model eğit (bir kere yapılır)
python src/train_model.py

# 4. Docker servisleri başlat
docker-compose up -d

# 5. Health check
curl http://localhost:8000/health

# 6. Swagger'ı aç
start http://localhost:8000/docs
```

### Günlük Development

```powershell
# 1. DB'leri başlat (zaten çalışıyorsa atla)
docker-compose -f docker-compose.dev.yml up -d

# 2. API'yi local'de başlat
.venv\Scripts\activate
uvicorn api.main:app --reload

# 3. Kod değişikliği yap, otomatik reload olur

# 4. Bitince DB'leri durdur (opsiyonel)
docker-compose -f docker-compose.dev.yml down
```

### Production Deployment

```bash
# 1. .env ayarla
cp .env.example .env
# .env'i düzenle

# 2. Build ve başlat
docker-compose build
docker-compose up -d

# 3. Kontrol et
docker-compose ps
docker-compose logs -f

# 4. Test
curl https://your-domain.com/health
```

---

## Dosya Yapısı

```
film-sentiment-py/
├── Dockerfile              # API image tanımı
├── docker-compose.yml      # Production: MongoDB + Redis + API
├── docker-compose.dev.yml  # Development: MongoDB + Redis + UI tools
├── .dockerignore           # Docker build'e dahil edilmeyenler
├── .env                    # Environment variables (gitignore'da)
├── models/                 # Eğitilmiş model (volume olarak mount)
│   ├── model.pkl
│   ├── vectorizer.pkl
│   └── metadata.json
└── logs/                   # Log dosyaları (volume olarak mount)
```

---

## Portlar Özeti

| Port | Servis | Açıklama |
|------|--------|----------|
| 8000 | API | FastAPI Sentiment API |
| 27017 | MongoDB | Database |
| 6379 | Redis | JWT Blacklist |
| 8081 | Redis Commander | Redis UI (dev only) |
| 8082 | Mongo Express | MongoDB UI (dev only) |

---

**AI Yazılım Mühendisi: Emre Yılmaz**  
**Tarih:** 2025-11-23  
**Version:** 2.0.0

