# 🚀 Authentication Quick Start

Bu kılavuz, API'yi hızlıca kullanmaya başlamanız için adım adım talimatlar sunar.

## 1️⃣ MongoDB Kurulumu

### Seçenek A: MongoDB Atlas (Cloud - Önerilen)

1. https://www.mongodb.com/cloud/atlas/register adresine gidin
2. Free Cluster oluşturun
3. Database User oluşturun (username ve password)
4. Network Access'e `0.0.0.0/0` ekleyin (veya kendi IP'niz)
5. "Connect" butonuna tıklayıp connection string alın

### Seçenek B: Docker ile MongoDB

```bash
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

### Seçenek C: Local MongoDB

MongoDB'yi indirin ve kurun: https://www.mongodb.com/try/download/community

## 2️⃣ Environment Variables

`.env` dosyası oluşturun:

```bash
# MongoDB
MONGO_URL=mongodb://localhost:27017
# veya Atlas için:
# MONGO_URL=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority

DATABASE_NAME=imdb_sentiment_db

# JWT Secret (güvenli bir key oluşturun)
JWT_SECRET_KEY=super-secret-key-change-this-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:3001
```

## 3️⃣ API Başlatma

```bash
cd D:\Development\MasterSWE\film-sentiment-py
.venv\Scripts\activate
uvicorn api.main:app --reload
```

API: http://localhost:8000  
Docs: http://localhost:8000/docs

## 4️⃣ İlk Kullanıcı Oluşturma

### Swagger UI'dan (Kolay)

1. http://localhost:8000/docs adresine gidin
2. `POST /auth/register` endpoint'ini açın
3. "Try it out" butonuna tıklayın
4. Kullanıcı bilgilerini girin:

```json
{
  "username": "test_user",
  "email": "test@example.com",
  "password": "TestPass123",
  "full_name": "Test User",
  "role": "user"
}
```

5. "Execute" butonuna tıklayın

### cURL ile

```bash
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "test_user",
    "email": "test@example.com",
    "password": "TestPass123",
    "full_name": "Test User",
    "role": "user"
  }'
```

## 5️⃣ Giriş Yapma ve Token Alma

### Swagger UI'dan

1. `POST /auth/login` endpoint'ini açın
2. "Try it out" tıklayın
3. Bilgileri girin:
   - username: `test_user`
   - password: `TestPass123`
4. "Execute" tıklayın
5. Response'dan `access_token`'ı kopyalayın

### cURL ile

```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test_user&password=TestPass123"
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0X3VzZXIiLCJleHAiOjE3MDA3NjI0MDB9.xxx",
  "token_type": "bearer"
}
```

## 6️⃣ Token ile Prediction Yapma

### Swagger UI'dan

1. Sağ üstteki "Authorize" butonuna tıklayın
2. Token'ı yapıştırın (sadece token, "Bearer" yazmayın)
3. "Authorize" tıklayın
4. `POST /predict` endpoint'ini kullanın

### cURL ile

```bash
TOKEN="<ACCESS_TOKEN_BURAYA>"

curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"text": "This movie was absolutely fantastic!"}'
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.95,
  "prediction_time_ms": 25.3
}
```

## 7️⃣ İstatistiklerimi Görme

```bash
curl -X GET "http://localhost:8000/stats/me" \
  -H "Authorization: Bearer $TOKEN"
```

**Response:**
```json
{
  "total_predictions": 5,
  "positive_count": 3,
  "negative_count": 2,
  "average_confidence": 0.89,
  "average_prediction_time_ms": 28.3
}
```

## ✅ Tamamlandı!

Artık API'yi kullanmaya hazırsınız. Detaylı bilgi için:

- **Authentication Guide:** `docs/AUTHENTICATION_GUIDE.md`
- **API Documentation:** http://localhost:8000/docs
- **Environment Setup:** `ENV_SETUP.md`

## 🔧 Sorun Giderme

### MongoDB bağlanamıyor

```bash
# MongoDB çalışıyor mu kontrol et
# Windows
netstat -ano | findstr :27017

# Docker container'ı kontrol et
docker ps | findstr mongodb
```

### Token geçersiz

- Token'ın 24 saat geçerliliği var
- Yeni token için tekrar login yapın
- Token'ı kopyalarken boşluk bırakmayın

### CORS hatası

`.env` dosyasında `CORS_ORIGINS` değişkenini kontrol edin:
```
CORS_ORIGINS=http://localhost:3000,http://localhost:3001
```

## 📚 İleri Seviye

- MongoDB Atlas'ta index'ler otomatik oluşturulur
- Production'da güçlü JWT secret key kullanın
- HTTPS kullanın (production)
- Rate limiting ekleyin (gelecek versiyon)

