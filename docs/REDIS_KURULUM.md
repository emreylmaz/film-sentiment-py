# Redis Kurulum Kılavuzu

## 📋 İçindekiler
1. [Redis Nedir?](#redis-nedir)
2. [Local Kurulum](#local-kurulum)
3. [Docker ile Kurulum](#docker-ile-kurulum)
4. [Cloud Redis (Production)](#cloud-redis-production)
5. [Test ve Doğrulama](#test-ve-doğrulama)
6. [Sorun Giderme](#sorun-giderme)

---

## Redis Nedir?

**Redis** (Remote Dictionary Server), in-memory key-value data store'dur.

### Projedeki Kullanım Alanları

1. **JWT Token Blacklist:** Logout yapılan token'ları saklar
2. **Session Management:** Kullanıcı oturumlarını yönetir
3. **Caching:** Sık kullanılan verileri cache'ler
4. **Rate Limiting:** API rate limit kontrolü (gelecek)

### Neden Redis?

- ⚡ **Çok Hızlı:** In-memory olduğu için millisaniye response
- 🔄 **TTL Desteği:** Veriler otomatik expire olabilir
- 📦 **Basit:** Key-value structure, öğrenmesi kolay
- 🚀 **Production-Ready:** Netflix, GitHub, Twitter kullanıyor

---

## Local Kurulum

### Windows

Redis'in resmi Windows build'i yok. **Docker kullanmanızı öneriyoruz.**

#### Docker ile (Önerilen)

```powershell
# Redis container'ı başlat
docker run -d `
  --name redis-blacklist `
  -p 6379:6379 `
  redis:7-alpine

# Çalışıyor mu kontrol et
docker ps | Select-String redis

# Redis'e bağlan (test)
docker exec -it redis-blacklist redis-cli ping
# PONG dönmeli
```

#### WSL ile (Alternatif)

```bash
# WSL Ubuntu'da
sudo apt-get update
sudo apt-get install redis-server

# Başlat
sudo service redis-server start

# Test
redis-cli ping  # PONG
```

### Mac

#### Homebrew ile (Önerilen)

```bash
# Redis yükle
brew install redis

# Başlat (autostart)
brew services start redis

# Veya manuel başlat
redis-server /usr/local/etc/redis.conf

# Test
redis-cli ping  # PONG
```

#### Docker ile

```bash
# Mac'te de Docker kullanabilirsiniz
docker run -d \
  --name redis-blacklist \
  -p 6379:6379 \
  redis:7-alpine
```

### Linux (Ubuntu/Debian)

```bash
# Redis yükle
sudo apt-get update
sudo apt-get install redis-server

# Başlat
sudo systemctl start redis

# Otomatik başlat (boot time)
sudo systemctl enable redis

# Durum kontrol
sudo systemctl status redis

# Test
redis-cli ping  # PONG
```

### Linux (CentOS/RHEL)

```bash
# EPEL repo ekle
sudo yum install epel-release

# Redis yükle
sudo yum install redis

# Başlat
sudo systemctl start redis

# Otomatik başlat
sudo systemctl enable redis

# Test
redis-cli ping  # PONG
```

---

## Docker ile Kurulum

### Temel Kullanım

```bash
# Redis başlat (alpine = küçük image)
docker run -d \
  --name redis-blacklist \
  -p 6379:6379 \
  redis:7-alpine

# Çalışıyor mu?
docker ps

# Logları izle
docker logs -f redis-blacklist

# Durdur
docker stop redis-blacklist

# Başlat
docker start redis-blacklist

# Kaldır
docker rm -f redis-blacklist
```

### Persistent Data (Data Kalıcılığı)

Default olarak Docker container silinince data kaybolur. Kalıcı data için volume kullan:

```bash
# Named volume oluştur
docker volume create redis-data

# Redis'i volume ile başlat
docker run -d \
  --name redis-blacklist \
  -p 6379:6379 \
  -v redis-data:/data \
  redis:7-alpine redis-server --appendonly yes

# Artık container silinse bile data kalır
```

### Password ile (Güvenlik)

```bash
# Password ile Redis başlat
docker run -d \
  --name redis-blacklist \
  -p 6379:6379 \
  redis:7-alpine redis-server --requirepass your-strong-password

# Bağlan (password ile)
docker exec -it redis-blacklist redis-cli
> AUTH your-strong-password
> PING  # PONG
```

### Docker Compose (Recommended)

Projede `docker-compose.yml` oluştur:

```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    container_name: redis-blacklist
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3

volumes:
  redis-data:
```

Başlat:

```bash
# Tüm servisleri başlat
docker-compose up -d

# Sadece Redis
docker-compose up -d redis

# Logları izle
docker-compose logs -f redis

# Durdur
docker-compose down

# Durdur + volume'leri sil
docker-compose down -v
```

---

## Cloud Redis (Production)

Local development için yukarıdaki yöntemler yeterli. Production'da cloud Redis kullanın.

### Redis Cloud (Önerilen)

**Avantajları:**
- ✅ Managed service (bakım yok)
- ✅ Auto backup
- ✅ High availability
- ✅ SSL/TLS encryption
- ✅ Free tier (30MB)

**Kurulum:**

1. https://redis.com/ git
2. "Try Free" ile hesap oluştur
3. "Create Database" seç:
   - Cloud: AWS/GCP/Azure
   - Region: Size yakın bir region
   - Plan: Free (development için yeterli)
4. Connection string'i kopyala:
   ```
   redis://default:password@redis-12345.c123.us-east-1-1.ec2.cloud.redislabs.com:12345
   ```

5. `.env` dosyasına ekle:
   ```bash
   REDIS_URL=redis://default:password@redis-12345...
   ```

### Upstash Redis

**Avantajları:**
- ✅ Serverless (pay-per-request)
- ✅ Free tier (10K commands/day)
- ✅ Global edge locations
- ✅ REST API (client gerekmez)

**Kurulum:**

1. https://upstash.com/ git
2. "Create Database" seç
3. Connection string'i kopyala
4. `.env`'ye ekle

### AWS ElastiCache

Production'da AWS kullanıyorsanız ElastiCache kullanın.

```bash
# AWS CLI ile oluştur
aws elasticache create-cache-cluster \
  --cache-cluster-id sentiment-api-redis \
  --engine redis \
  --cache-node-type cache.t2.micro \
  --num-cache-nodes 1
```

### Environment Variables

Cloud Redis kullanırken:

```bash
# .env
REDIS_URL=redis://default:password@your-redis-url:port
REDIS_PASSWORD=your-password  # Bazı provider'lar URL'de, bazıları ayrı
```

---

## Test ve Doğrulama

### 1. Redis CLI ile Test

```bash
# Bağlan
redis-cli

# Veya Docker'da
docker exec -it redis-blacklist redis-cli

# Test komutları
> PING
PONG

> SET test "hello"
OK

> GET test
"hello"

> DEL test
(integer) 1

> KEYS *
(empty array)

> EXIT
```

### 2. Python ile Test

Proje içinde:

```bash
# Redis client test scripti çalıştır
cd api
python redis_client.py
```

Çıktı:

```
============================================================
Redis Client Test
============================================================
✓ Redis bağlantısı başarılı

1. Setting key 'test:example' with 60s TTL...
   ✓ Set: True

2. Getting value...
   ✓ Value: hello_redis

3. Checking existence...
   ✓ Exists: True

4. Getting TTL...
   ✓ TTL: 59 seconds

5. Deleting key...
   ✓ Deleted: True

6. Checking after delete...
   ✓ Exists after delete: False

7. Redis info...
   ✓ Status: connected
   ✓ Version: 7.2.0

✓ Test tamamlandı!
```

### 3. API Health Check

FastAPI başlatıp health endpoint'ini kontrol et:

```bash
# API'yi başlat
uvicorn api.main:app --reload

# Health check
curl http://localhost:8000/health
```

Response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "database_connected": true,
  "redis_connected": true,
  "redis_info": {
    "available": true,
    "status": "connected",
    "version": "7.2.0"
  },
  "blacklist_stats": {
    "available": true,
    "total_blacklisted": 0
  }
}
```

### 4. Blacklist Test

```bash
# 1. Login
curl -X POST http://localhost:8000/auth/login \
  -d "username=test_user&password=test123"

# Token'ı kopyala
TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

# 2. Token ile request (çalışmalı)
curl http://localhost:8000/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie is great!"}'

# 3. Logout
curl -X POST http://localhost:8000/auth/logout \
  -H "Authorization: Bearer $TOKEN"

# 4. Aynı token ile tekrar (401 dönmeli)
curl http://localhost:8000/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie is great!"}'

# Response: 401 Unauthorized - Token has been revoked
```

### 5. Redis Monitoring

```bash
# Redis CLI'da
redis-cli

> MONITOR  # Tüm komutları canlı izle

# Başka terminalde API kullan, MONITOR'da görürsün

> INFO stats  # İstatistikler

> INFO memory  # Memory kullanımı

> DBSIZE  # Key sayısı

> SCAN 0 MATCH blacklist:* COUNT 100  # Blacklist key'leri
```

---

## Sorun Giderme

### Problem 1: Redis Başlamıyor

**Hata:**
```
Could not connect to Redis at 127.0.0.1:6379: Connection refused
```

**Çözüm:**

```bash
# Redis çalışıyor mu?
# Linux/Mac
ps aux | grep redis

# Docker
docker ps | grep redis

# Yoksa başlat
# Linux/Mac
redis-server
# veya
brew services start redis  # Mac
sudo systemctl start redis  # Linux

# Docker
docker start redis-blacklist
```

### Problem 2: Port 6379 Kullanımda

**Hata:**
```
Address already in use
```

**Çözüm:**

```bash
# Port'u kim kullanıyor?
# Linux/Mac
lsof -i :6379

# Windows
netstat -ano | findstr :6379

# Başka port kullan
redis-server --port 6380

# Environment variable'ı güncelle
REDIS_URL=redis://localhost:6380
```

### Problem 3: Redis Memory Doldu

**Hata:**
```
OOM command not allowed when used memory > 'maxmemory'
```

**Çözüm:**

```bash
# redis.conf düzenle
# Linux: /etc/redis/redis.conf
# Mac: /usr/local/etc/redis.conf

# Ayarlar
maxmemory 256mb
maxmemory-policy allkeys-lru  # En eski key'leri sil

# Redis'i restart et
sudo systemctl restart redis  # Linux
brew services restart redis   # Mac
```

### Problem 4: Python redis Modülü Bulunamadı

**Hata:**
```python
ModuleNotFoundError: No module named 'redis'
```

**Çözüm:**

```bash
# Virtual environment aktif mi?
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

# Redis yükle
pip install redis>=5.0.0

# Veya tüm dependencies
pip install -r requirements.txt
```

### Problem 5: Redis Password Hatası

**Hata:**
```
NOAUTH Authentication required
```

**Çözüm:**

```bash
# Password kullanıyorsan environment variable'da belirt
REDIS_URL=redis://:password@localhost:6379
# veya
REDIS_PASSWORD=your-password

# Redis CLI'da password ile bağlan
redis-cli -a your-password
# veya
redis-cli
> AUTH your-password
> PING
```

### Problem 6: Docker Container Duruyor

**Hata:**
```bash
docker ps  # Redis yok
```

**Çözüm:**

```bash
# Container'ın durumunu kontrol et
docker ps -a | grep redis

# Logları kontrol et
docker logs redis-blacklist

# Yeniden başlat
docker start redis-blacklist

# Çalışmıyorsa yeniden oluştur
docker rm redis-blacklist
docker run -d --name redis-blacklist -p 6379:6379 redis:7-alpine
```

---

## İleri Seviye

### Redis Persistence

Redis varsayılan olarak data'yı disk'e yazar ama crash durumunda son birkaç saniye kaybolabilir.

**RDB (Snapshot):**
```bash
# redis.conf
save 900 1      # 900 saniyede 1 değişiklik varsa kaydet
save 300 10     # 300 saniyede 10 değişiklik varsa kaydet
save 60 10000   # 60 saniyede 10000 değişiklik varsa kaydet
```

**AOF (Append Only File):**
```bash
# redis.conf
appendonly yes
appendfsync everysec  # Her saniye sync (güvenli + performanslı)
```

### Redis Replication

High availability için:

```bash
# Slave Redis (replica)
redis-server --port 6380 --replicaof 127.0.0.1 6379
```

Master çökerse slave'e geçersin.

### Redis Sentinel

Otomatik failover için:

```bash
# sentinel.conf
sentinel monitor mymaster 127.0.0.1 6379 2
sentinel down-after-milliseconds mymaster 5000

# Başlat
redis-sentinel sentinel.conf
```

### Redis Cluster

Horizontal scaling için:

```bash
# 3 master + 3 slave cluster
redis-cli --cluster create \
  127.0.0.1:7000 127.0.0.1:7001 127.0.0.1:7002 \
  127.0.0.1:7003 127.0.0.1:7004 127.0.0.1:7005 \
  --cluster-replicas 1
```

---

## Kaynaklar

- **Redis Official:** https://redis.io/
- **Redis Quick Start:** https://redis.io/topics/quickstart
- **Redis Commands:** https://redis.io/commands
- **Redis Python Client:** https://redis.readthedocs.io/
- **Docker Hub Redis:** https://hub.docker.com/_/redis
- **Redis Cloud:** https://redis.com/
- **Upstash:** https://upstash.com/

---

## Revizyon Geçmişi

| Tarih | Versiyon | Değişiklik | Yazar |
|-------|----------|------------|-------|
| 2025-11-23 | 1.0 | İlk versiyon oluşturuldu | AI Yazılım Mühendisi: Emre Yılmaz |

---

**Not:** Redis opsiyoneldir. Redis yoksa API çalışır ama JWT blacklist özelliği devre dışı kalır.

