# Redis JWT Blacklist Sistemi

## 📋 İçindekiler
1. [Genel Bakış](#genel-bakış)
2. [Mimari](#mimari)
3. [Kurulum](#kurulum)
4. [Kullanım](#kullanım)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)
7. [Production Notları](#production-notları)

---

## Genel Bakış

### Neden Redis Blacklist?

JWT token'lar stateless'tır, yani server-side'da saklanmazlar. Bu nedenle:
- ❌ **Problem:** Normal logout yapılsa bile token hala geçerlidir
- ❌ **Problem:** Token çalınırsa expire olana kadar kullanılabilir
- ❌ **Problem:** Password değişince eski token'lar hala çalışır

✅ **Çözüm:** Redis blacklist sistemi ile logout yapılan veya revoke edilen token'lar kullanılamaz hale getirilir.

### Özellikler

- ✅ **Gerçek Logout:** Token blacklist'e eklenir ve tekrar kullanılamaz
- ✅ **Otomatik Temizleme:** Token expire olunca Redis'ten otomatik silinir (TTL)
- ✅ **Performans:** Redis in-memory cache olduğu için çok hızlı
- ✅ **Graceful Degradation:** Redis yoksa sistem uyarı verir ama çökmez
- ✅ **Unique Token ID:** Her token'a `jti` (JWT ID) eklenir

---

## Mimari

### Sistem Bileşenleri

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI App                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Login (auth.py)                                          │
│     └─> Token oluştur (jti ile)                             │
│                                                               │
│  2. Token Validation (dependencies.py)                       │
│     └─> Token'ı decode et                                    │
│     └─> Redis'te blacklist kontrolü ✓                       │
│     └─> İzin ver veya reddet                                │
│                                                               │
│  3. Logout (auth.py)                                         │
│     └─> Token'ı blacklist'e ekle (TTL ile)                  │
│     └─> Client'a başarı mesajı gönder                       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                          ↓ ↑
                    ┌─────────────┐
                    │    Redis    │
                    ├─────────────┤
                    │  blacklist: │
                    │   - jti_123 │
                    │   - jti_456 │
                    │   - jti_789 │
                    └─────────────┘
```

### Token Flow

```
1. LOGIN
   User ---login---> API
                      ↓
                  Generate JWT
                  (with jti claim)
                      ↓
                  Return token
                      ↓
   User <------------┘

2. AUTHENTICATED REQUEST
   User ---request + token---> API
                                ↓
                          Decode token
                                ↓
                        Get jti from token
                                ↓
                    Check Redis: blacklist:{jti}
                                ↓
                          ┌─────┴─────┐
                     Found?         Not Found?
                      ↓                  ↓
                 401 Unauthorized    Allow request
                      
3. LOGOUT
   User ---logout + token---> API
                               ↓
                         Get jti from token
                               ↓
                      Calculate TTL (time to expire)
                               ↓
                   Redis SET blacklist:{jti} = "logout"
                         (with TTL)
                               ↓
                        Return success
                               ↓
   User <---------------------┘
   
   (Token artık kullanılamaz, blacklist'te)
```

---

## Kurulum

### 1. Redis Kurulumu

#### Local (Development)

**Windows:**
```powershell
# Redis Windows için resmi build yok, Docker kullanın
docker run -d -p 6379:6379 --name redis-blacklist redis:7-alpine
```

**Mac (Homebrew):**
```bash
brew install redis
brew services start redis
```

**Linux (Ubuntu):**
```bash
sudo apt-get update
sudo apt-get install redis-server
sudo systemctl start redis
```

#### Redis Cloud (Production)

1. **Redis Cloud** veya **Upstash** hesabı oluşturun
2. Yeni database oluşturun
3. Connection string'i kopyalayın:
   ```
   redis://default:password@redis-xxxxx.cloud.redislabs.com:12345
   ```

### 2. Environment Variables

`.env` dosyasına ekleyin:

```bash
# Local Redis
REDIS_URL=redis://localhost:6379

# Redis Cloud (Production)
# REDIS_URL=redis://default:your-password@redis-xxxxx.cloud.redislabs.com:12345

# Redis Password (opsiyonel)
REDIS_PASSWORD=

# Redis Database Number (0-15)
REDIS_DB=0
```

### 3. Dependency Kurulumu

```bash
pip install redis>=5.0.0
```

`requirements.txt` zaten içeriyor.

---

## Kullanım

### API Endpoint'leri

#### 1. Login (Token Al)

```bash
POST /auth/login
Content-Type: application/x-www-form-urlencoded

username=emre_yilmaz&password=your_password
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

Token içinde `jti` (unique ID) var:
```json
{
  "sub": "emre_yilmaz",
  "exp": 1234567890,
  "jti": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"  ← Unique ID
}
```

#### 2. Authenticated Request

```bash
GET /predict
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Process:**
1. Token decode edilir
2. `jti` claim'i çıkarılır
3. Redis'te `blacklist:{jti}` key'i kontrol edilir
4. Yoksa → İzin verilir ✅
5. Varsa → 401 Unauthorized ❌

#### 3. Logout (Token İptal Et)

```bash
POST /auth/logout
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Response (Redis Available):**
```json
{
  "message": "Successfully logged out",
  "detail": "Token has been revoked and added to blacklist",
  "username": "emre_yilmaz"
}
```

**Response (Redis Unavailable):**
```json
{
  "message": "Logged out (client-side only)",
  "detail": "Please remove the token from client storage",
  "warning": "Server-side token revocation not available (Redis disconnected)"
}
```

**Process:**
1. Token'dan `jti` çıkarılır: `a1b2c3d4-e5f6-7890-abcd-ef1234567890`
2. Token'ın kalan geçerlilik süresi hesaplanır: `3600 saniye`
3. Redis'e eklenir:
   ```
   SET blacklist:a1b2c3d4-e5f6-7890-abcd-ef1234567890 "user_logout:2025-11-23T10:30:00"
   EXPIRE blacklist:a1b2c3d4-e5f6-7890-abcd-ef1234567890 3600
   ```
4. 3600 saniye sonra Redis otomatik siler (token zaten expired)

---

## Best Practices

### 1. TTL Kullanımı

✅ **DO:** Token'ın kalan geçerlilik süresini TTL olarak kullan
```python
ttl = token_exp - current_time  # Örn: 3600 saniye
await redis.setex(f"blacklist:{jti}", ttl, "logout")
```

❌ **DON'T:** Sabit TTL kullanma
```python
# YANLIŞ: Token 10 dakika sonra expire olacak ama Redis 1 gün tutuyor
await redis.setex(f"blacklist:{jti}", 86400, "logout")
```

### 2. Graceful Degradation

✅ **DO:** Redis yoksa uyarı ver ama servisi çöktürme
```python
if not is_redis_available():
    logger.warning("Redis unavailable, blacklist disabled")
    return False  # Devam et, sadece client-side logout
```

❌ **DON'T:** Redis hatası için 500 dönme
```python
if not redis:
    raise HTTPException(status_code=500, detail="Redis down")
```

### 3. Unique Token ID (jti)

✅ **DO:** Her token'a unique `jti` claim'i ekle
```python
token_data = {
    "sub": username,
    "jti": str(uuid.uuid4())  # Unique ID
}
```

❌ **DON'T:** Token string'ini direk blacklist'e ekleme
```python
# YANLIŞ: Token string çok uzun, Redis key olarak verimsiz
await redis.set(f"blacklist:{token}", "logout")
```

### 4. Blacklist Kontrolü

✅ **DO:** Her protected endpoint'te blacklist kontrolü yap
```python
async def get_current_user(token: str = Depends(oauth2_scheme)):
    # 1. Token decode
    # 2. Blacklist kontrolü ✓
    # 3. User bilgisi getir
```

❌ **DON'T:** Sadece logout'ta kontrol etme

### 5. Security Reasons

Token'ı farklı sebeplerle blacklist'e ekle:
```python
await add_token_to_blacklist(token, reason="user_logout")
await add_token_to_blacklist(token, reason="password_change")
await add_token_to_blacklist(token, reason="security_breach")
await add_token_to_blacklist(token, reason="admin_revoke")
```

---

## Troubleshooting

### Problem 1: Redis'e Bağlanamıyor

**Error:**
```
⚠ Redis bağlantı hatası: ConnectionError
⚠ Redis olmadan devam ediliyor. JWT blacklist özelliği devre dışı.
```

**Çözüm:**
1. Redis çalışıyor mu kontrol et:
   ```bash
   # Docker
   docker ps | grep redis
   
   # Local
   redis-cli ping  # PONG döner
   ```

2. Connection string doğru mu:
   ```bash
   echo $REDIS_URL
   # redis://localhost:6379 olmalı
   ```

3. Port açık mı:
   ```bash
   telnet localhost 6379
   ```

### Problem 2: Token Blacklist'te Ama Kullanılabiliyor

**Çözüm:**
1. `jti` claim'i var mı kontrol et:
   ```python
   # Token decode et, jti var mı bak
   payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
   print(payload.get("jti"))  # None olmamalı
   ```

2. Redis'te key var mı:
   ```bash
   redis-cli
   > SCAN 0 MATCH blacklist:* COUNT 100
   > GET blacklist:a1b2c3d4-e5f6-7890-abcd-ef1234567890
   ```

3. `dependencies.py`'de blacklist kontrolü var mı:
   ```python
   is_blacklisted = await is_token_blacklisted(token)
   if is_blacklisted:
       raise HTTPException(...)
   ```

### Problem 3: Redis Memory Doldu

**Çözüm:**
1. TTL'leri kontrol et:
   ```bash
   redis-cli
   > SCAN 0 MATCH blacklist:* COUNT 100
   > TTL blacklist:xxxx  # -1 ise TTL yok (KÖTÜ)
   ```

2. Max memory policy ayarla:
   ```bash
   # redis.conf
   maxmemory 256mb
   maxmemory-policy allkeys-lru  # En eski key'leri sil
   ```

3. Blacklist stats kontrol et:
   ```bash
   curl http://localhost:8000/health
   # redis_info kısmına bak
   ```

---

## Production Notları

### 1. Redis High Availability

**Redis Sentinel** veya **Redis Cluster** kullan:
```python
from redis.sentinel import Sentinel

sentinel = Sentinel([
    ('redis-sentinel-1', 26379),
    ('redis-sentinel-2', 26379),
    ('redis-sentinel-3', 26379)
])

redis_client = sentinel.master_for('mymaster')
```

### 2. Connection Pooling

```python
redis_client = aioredis.from_url(
    REDIS_URL,
    encoding="utf-8",
    decode_responses=True,
    max_connections=50,  # Connection pool
    socket_timeout=5,
    socket_connect_timeout=5
)
```

### 3. Monitoring

**Redis Key Sayısı:**
```bash
redis-cli
> INFO keyspace
# db0:keys=150,expires=150
```

**Blacklist Stats Endpoint:**
```python
@app.get("/admin/blacklist/stats")
async def get_blacklist_stats():
    stats = await blacklist.get_blacklist_stats()
    return stats
```

### 4. Rate Limiting

Redis ile rate limiting ekle:
```python
async def check_rate_limit(username: str):
    key = f"rate_limit:{username}"
    count = await redis.incr(key)
    
    if count == 1:
        await redis.expire(key, 60)  # 1 dakika
    
    if count > 100:  # 1 dakikada max 100 request
        raise HTTPException(status_code=429, detail="Too many requests")
```

### 5. Security

- ✅ Redis password kullan (production)
- ✅ Redis'i internal network'te tut (public expose etme)
- ✅ TLS/SSL kullan (Redis Cloud otomatik sağlıyor)
- ✅ Firewall rules ayarla (sadece API server erişebilsin)

**Redis Password:**
```bash
REDIS_URL=redis://:your-strong-password@localhost:6379
```

**Redis TLS:**
```python
redis_client = aioredis.from_url(
    "rediss://...",  # rediss:// (SSL)
    ssl_cert_reqs="required"
)
```

---

## Kod Örnekleri

### Manuel Blacklist Kontrolü

```python
from api.blacklist import is_token_blacklisted

token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

is_blacklisted = await is_token_blacklisted(token)

if is_blacklisted:
    print("❌ Token blacklist'te, kullanamaz")
else:
    print("✅ Token temiz, kullanabilir")
```

### Tüm Kullanıcı Token'larını İptal Et

```python
# TODO: User session tracking gerekli
# Şu anda implement edilmemiş, gelecek versiyon için

from api.blacklist import blacklist_all_user_tokens

count = await blacklist_all_user_tokens("emre_yilmaz")
print(f"{count} token iptal edildi")
```

### Token'ı Blacklist'ten Çıkar (Admin)

```python
from api.blacklist import remove_token_from_blacklist

token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

removed = await remove_token_from_blacklist(token)

if removed:
    print("✅ Token blacklist'ten çıkarıldı")
```

### Redis Health Check

```python
from api.redis_client import test_redis_connection, get_redis_info

# Bağlantı testi
is_ok = await test_redis_connection()
print(f"Redis OK: {is_ok}")

# Redis bilgileri
info = await get_redis_info()
print(f"Redis Version: {info.get('version')}")
print(f"Uptime: {info.get('uptime_seconds')} seconds")
```

---

## Test

### Unit Test

```python
import pytest
from api.blacklist import add_token_to_blacklist, is_token_blacklisted
from api.auth_utils import create_access_token

@pytest.mark.asyncio
async def test_blacklist():
    # Token oluştur
    token = create_access_token({"sub": "test_user"})
    
    # Blacklist'te olmamalı
    assert await is_token_blacklisted(token) == False
    
    # Blacklist'e ekle
    await add_token_to_blacklist(token)
    
    # Blacklist'te olmalı
    assert await is_token_blacklisted(token) == True
```

### Integration Test

```bash
# 1. Login
curl -X POST http://localhost:8000/auth/login \
  -d "username=emre_yilmaz&password=test123"

# Response: {"access_token": "..."}

# 2. Token ile request
curl http://localhost:8000/predict \
  -H "Authorization: Bearer TOKEN"

# Response: 200 OK

# 3. Logout
curl -X POST http://localhost:8000/auth/logout \
  -H "Authorization: Bearer TOKEN"

# Response: {"message": "Successfully logged out"}

# 4. Aynı token ile tekrar dene
curl http://localhost:8000/predict \
  -H "Authorization: Bearer TOKEN"

# Response: 401 Unauthorized (blacklist'te)
```

---

## Kaynaklar

- **Redis Docs:** https://redis.io/docs/
- **Redis Python Client:** https://redis.readthedocs.io/
- **JWT Best Practices:** https://tools.ietf.org/html/rfc8725
- **Redis Cloud:** https://redis.com/
- **Upstash Redis:** https://upstash.com/

---

## Revizyon Geçmişi

| Tarih | Versiyon | Değişiklik | Yazar |
|-------|----------|------------|-------|
| 2025-11-23 | 1.0 | İlk versiyon oluşturuldu | AI Yazılım Mühendisi: Emre Yılmaz |

---

**İletişim:**
- Proje: IMDB Sentiment Analysis API
- AI Yazılım Mühendisi: Emre Yılmaz
- Dokümantasyon: `docs/REDIS_BLACKLIST.md`

