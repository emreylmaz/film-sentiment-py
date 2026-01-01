"""
Redis bağlantı yönetimi.

Bu modül, Redis ile async bağlantı kurar ve JWT blacklist
için gerekli operasyonları sağlar.
"""

import os
from typing import Optional
from redis import asyncio as aioredis
from src.utils.logger import setup_logger
from datetime import datetime

# Logger'ı ayarla
logger = setup_logger(__name__, f"logs/redis_{datetime.now().strftime('%Y%m%d')}.log")

# Global Redis client
redis_client: Optional[aioredis.Redis] = None


def get_redis_url() -> str:
    """
    Redis connection URL'ini config'den alır.
    
    Öncelik: Environment Variable > config.yaml > Default
    
    Returns:
        str: Redis connection URL
    """
    # Lazy import to avoid circular dependency
    from api.config import settings
    return settings.REDIS_URL


def get_redis_password() -> Optional[str]:
    """
    Redis password'ünü config'den alır.
    
    Öncelik: Environment Variable > config.yaml > Default
    
    Returns:
        Optional[str]: Redis password (yoksa None)
    """
    # Lazy import to avoid circular dependency
    from api.config import settings
    return settings.redis.password


async def connect_to_redis():
    """
    Redis'e asenkron bağlantı kurar.
    Uygulama başlangıcında (startup event) çağrılmalıdır.
    
    Redis yoksa veya bağlantı başarısız olursa, sistem uyarı verir
    ancak çökmez. Bu sayede Redis opsiyonel kalır.
    """
    global redis_client
    
    try:
        redis_url = get_redis_url()
        redis_password = get_redis_password()
        
        # Redis client oluştur
        redis_client = aioredis.from_url(
            redis_url,
            password=redis_password,
            encoding="utf-8",
            decode_responses=True,  # String olarak dön
            socket_timeout=5,
            socket_connect_timeout=5
        )
        
        # Bağlantıyı test et
        await redis_client.ping()
        logger.info(f"✓ Redis'e başarıyla bağlanıldı: {redis_url}")
        
    except Exception as e:
        logger.warning(f"⚠ Redis bağlantı hatası: {e}")
        logger.warning("⚠ Redis olmadan devam ediliyor. JWT blacklist özelliği devre dışı.")
        redis_client = None


async def close_redis_connection():
    """
    Redis bağlantısını kapatır.
    Uygulama kapanışında (shutdown event) çağrılmalıdır.
    """
    global redis_client
    
    if redis_client:
        try:
            await redis_client.aclose()
            logger.info("✓ Redis bağlantısı kapatıldı")
        except Exception as e:
            logger.error(f"Redis kapatma hatası: {e}")
    
    redis_client = None


def get_redis_client() -> Optional[aioredis.Redis]:
    """
    Redis client instance'ını döner.
    
    Returns:
        Optional[aioredis.Redis]: Redis client (bağlantı yoksa None)
        
    Usage:
        ```python
        redis = get_redis_client()
        if redis:
            await redis.set("key", "value")
        ```
    """
    return redis_client


def is_redis_available() -> bool:
    """
    Redis'in kullanılabilir olup olmadığını kontrol eder.
    
    Returns:
        bool: Redis bağlantısı varsa True, yoksa False
    """
    return redis_client is not None


async def test_redis_connection() -> bool:
    """
    Redis bağlantısını test eder.
    
    Returns:
        bool: Bağlantı başarılıysa True, değilse False
    """
    if not redis_client:
        return False
    
    try:
        await redis_client.ping()
        return True
    except Exception as e:
        logger.error(f"Redis ping hatası: {e}")
        return False


# ============================================================================
# Redis Key Helper Functions
# ============================================================================

def get_blacklist_key(token_jti: str) -> str:
    """
    Blacklist için Redis key oluşturur.
    
    Args:
        token_jti (str): JWT token'ın unique ID'si (jti claim)
        
    Returns:
        str: Redis key (örn: "blacklist:abc123")
    """
    return f"blacklist:{token_jti}"


def get_user_session_key(username: str) -> str:
    """
    User session için Redis key oluşturur.
    
    Args:
        username (str): Kullanıcı adı
        
    Returns:
        str: Redis key (örn: "session:emre_yilmaz")
    """
    return f"session:{username}"


# ============================================================================
# Redis Operations (Best Practices)
# ============================================================================

async def set_with_ttl(key: str, value: str, ttl_seconds: int) -> bool:
    """
    Redis'e TTL (Time To Live) ile değer yazar.
    
    Args:
        key (str): Redis key
        value (str): Değer
        ttl_seconds (int): Geçerlilik süresi (saniye)
        
    Returns:
        bool: İşlem başarılıysa True
    """
    if not redis_client:
        return False
    
    try:
        await redis_client.setex(key, ttl_seconds, value)
        return True
    except Exception as e:
        logger.error(f"Redis set hatası: {e}")
        return False


async def get_value(key: str) -> Optional[str]:
    """
    Redis'ten değer okur.
    
    Args:
        key (str): Redis key
        
    Returns:
        Optional[str]: Değer (yoksa None)
    """
    if not redis_client:
        return None
    
    try:
        return await redis_client.get(key)
    except Exception as e:
        logger.error(f"Redis get hatası: {e}")
        return None


async def delete_key(key: str) -> bool:
    """
    Redis'ten key'i siler.
    
    Args:
        key (str): Redis key
        
    Returns:
        bool: İşlem başarılıysa True
    """
    if not redis_client:
        return False
    
    try:
        await redis_client.delete(key)
        return True
    except Exception as e:
        logger.error(f"Redis delete hatası: {e}")
        return False


async def key_exists(key: str) -> bool:
    """
    Redis'te key'in varlığını kontrol eder.
    
    Args:
        key (str): Redis key
        
    Returns:
        bool: Key varsa True
    """
    if not redis_client:
        return False
    
    try:
        return await redis_client.exists(key) > 0
    except Exception as e:
        logger.error(f"Redis exists hatası: {e}")
        return False


async def get_ttl(key: str) -> int:
    """
    Key'in kalan TTL'ini (saniye) döner.
    
    Args:
        key (str): Redis key
        
    Returns:
        int: Kalan süre (saniye), key yoksa -2, TTL yoksa -1
    """
    if not redis_client:
        return -2
    
    try:
        return await redis_client.ttl(key)
    except Exception as e:
        logger.error(f"Redis TTL hatası: {e}")
        return -2


# ============================================================================
# Health Check
# ============================================================================

async def get_redis_info() -> dict:
    """
    Redis durumu hakkında bilgi döner.
    
    Returns:
        dict: Redis durum bilgileri
    """
    if not redis_client:
        return {
            "available": False,
            "status": "disconnected",
            "error": "Redis client not initialized"
        }
    
    try:
        await redis_client.ping()
        info = await redis_client.info("server")
        
        return {
            "available": True,
            "status": "connected",
            "version": info.get("redis_version", "unknown"),
            "uptime_seconds": info.get("uptime_in_seconds", 0)
        }
    except Exception as e:
        return {
            "available": False,
            "status": "error",
            "error": str(e)
        }


# ============================================================================
# Örnek Kullanım ve Test
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def test_redis_operations():
        """Redis operasyonlarını test eder."""
        print("=" * 60)
        print("Redis Client Test")
        print("=" * 60)
        
        try:
            # Bağlan
            await connect_to_redis()
            
            if not is_redis_available():
                print("❌ Redis bağlantısı yok")
                return
            
            print("✓ Redis bağlantısı başarılı")
            
            # Test key-value
            test_key = "test:example"
            test_value = "hello_redis"
            
            # Set with TTL
            print(f"\n1. Setting key '{test_key}' with 60s TTL...")
            success = await set_with_ttl(test_key, test_value, 60)
            print(f"   {'✓' if success else '✗'} Set: {success}")
            
            # Get value
            print(f"\n2. Getting value...")
            value = await get_value(test_key)
            print(f"   ✓ Value: {value}")
            
            # Check existence
            print(f"\n3. Checking existence...")
            exists = await key_exists(test_key)
            print(f"   ✓ Exists: {exists}")
            
            # Get TTL
            print(f"\n4. Getting TTL...")
            ttl = await get_ttl(test_key)
            print(f"   ✓ TTL: {ttl} seconds")
            
            # Delete
            print(f"\n5. Deleting key...")
            deleted = await delete_key(test_key)
            print(f"   {'✓' if deleted else '✗'} Deleted: {deleted}")
            
            # Check again
            print(f"\n6. Checking after delete...")
            exists_after = await key_exists(test_key)
            print(f"   ✓ Exists after delete: {exists_after}")
            
            # Redis info
            print(f"\n7. Redis info...")
            info = await get_redis_info()
            print(f"   ✓ Status: {info['status']}")
            if info['available']:
                print(f"   ✓ Version: {info.get('version')}")
            
            # Bağlantıyı kapat
            await close_redis_connection()
            print("\n✓ Test tamamlandı!")
            
        except Exception as e:
            print(f"\n❌ Test hatası: {e}")
    
    # Test'i çalıştır
    asyncio.run(test_redis_operations())

