# Environment Variables Setup

Bu dosya, projenin çalışması için gerekli environment variable'ları açıklar.

> **✅ Otomatik Yükleme:** API başlarken `.env` dosyasını **otomatik olarak okur** (`python-dotenv` ile).  
> Proje kök dizininde `.env` dosyası oluşturmanız yeterli!

## Kurulum

### 1. .env Dosyası Oluşturma

Proje kök dizininde `.env` dosyası oluşturun:

```bash
# Windows
copy NUL .env

# Linux/Mac
touch .env
```

### 2. Environment Variables

`.env` dosyanıza aşağıdaki değişkenleri ekleyin:

```bash
# ============================================================================
# MongoDB Ayarları
# ============================================================================

# Local MongoDB
MONGO_URL=mongodb://localhost:27017

# MongoDB Atlas (Cloud) - Önerilen production için
# MONGO_URL=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority

DATABASE_NAME=imdb_sentiment_db

# ============================================================================
# JWT Authentication
# ============================================================================

# SECRET KEY - ÖNEMLİ: Production'da mutlaka değiştirin!
# Güçlü bir random string kullanın (32+ karakter)
JWT_SECRET_KEY=your-very-secure-secret-key-change-this-in-production-use-32plus-chars

# JWT Algorithm
JWT_ALGORITHM=HS256

# Token geçerlilik süresi (dakika - 1440 = 24 saat)
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# ============================================================================
# API Ayarları
# ============================================================================

API_HOST=0.0.0.0
API_PORT=8000

# ============================================================================
# CORS Ayarları
# ============================================================================

# Virgülle ayrılmış origin listesi
# Development
CORS_ORIGINS=http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000

# Production - Kendi domain'inizi ekleyin
# CORS_ORIGINS=https://your-frontend-app.com,https://www.your-frontend-app.com

# ============================================================================
# Redis Ayarları (JWT Blacklist & Caching)
# ============================================================================

# Local Redis
REDIS_URL=redis://localhost:6379

# Redis Cloud/Upstash (Cloud Redis)
# REDIS_URL=redis://default:password@redis-xxxxx.cloud.redislabs.com:port

# Redis Password (opsiyonel)
REDIS_PASSWORD=

# Redis Database Number
REDIS_DB=0
```

## Secret Key Oluşturma

Güvenli bir JWT secret key oluşturmak için:

```python
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Çıktıyı `JWT_SECRET_KEY` değişkenine yapıştırın.

## MongoDB Kurulum Seçenekleri

### Seçenek 1: Local MongoDB

1. MongoDB'yi indirin ve kurun: https://www.mongodb.com/try/download/community
2. MongoDB servisini başlatın
3. `.env` dosyasında `MONGO_URL=mongodb://localhost:27017` kullanın

### Seçenek 2: MongoDB Atlas (Cloud - Önerilen)

1. MongoDB Atlas hesabı oluşturun: https://www.mongodb.com/cloud/atlas/register
2. Free cluster oluşturun
3. Database User oluşturun (username ve password)
4. Network Access'e kendi IP'nizi ekleyin (veya 0.0.0.0/0 - tüm IP'ler)
5. "Connect" butonuna tıklayıp connection string alın
6. `.env` dosyasında connection string'i kullanın:
   ```
   MONGO_URL=mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
   ```

### Seçenek 3: Docker ile MongoDB

```bash
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

## CORS Ayarları

**Development:**
```
CORS_ORIGINS=http://localhost:3000,http://localhost:3001
```

**Production:**
```
CORS_ORIGINS=https://your-frontend-domain.com,https://www.your-frontend-domain.com
```

## Güvenlik Notları

⚠️ **ÖNEMLİ:**
- `.env` dosyası asla git'e commit edilmemelidir!
- Production'da güçlü secret key kullanın
- MongoDB connection string'de şifre varsa güvenli tutun
- CORS'u production'da sadece kendi domain'lerinize açın

## Doğrulama

Environment variable'ların doğru yüklendiğini kontrol etmek için:

```python
python -c "import os; print('MONGO_URL:', os.getenv('MONGO_URL', 'NOT SET'))"
```

## Sorun Giderme

### MongoDB bağlantı hatası
- MongoDB servisinin çalıştığından emin olun
- Connection string'in doğru olduğunu kontrol edin
- Firewall ayarlarını kontrol edin

### JWT token hatası
- `JWT_SECRET_KEY` environment variable'ının set edildiğinden emin olun
- Secret key'in en az 32 karakter olduğundan emin olun

### CORS hatası
- Frontend URL'inin `CORS_ORIGINS` listesinde olduğundan emin olun
- Virgül ile ayrılmış format kullanın (boşluk yok)

