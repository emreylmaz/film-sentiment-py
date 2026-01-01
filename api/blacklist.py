"""
JWT Token Blacklist Service.

Bu modül, logout yapılan token'ları blacklist'e ekler ve
token validation sırasında blacklist kontrolü yapar.

Best Practices:
- Redis TTL kullanarak token expiry ile senkronize
- Token'ın jti (JWT ID) claim'ini kullanarak unique ID
- Graceful degradation: Redis yoksa uyarı ver ama çökme
"""

from typing import Optional
from datetime import datetime, timedelta
from jose import jwt, JWTError
from api.redis_client import (
    get_redis_client,
    is_redis_available,
    set_with_ttl,
    key_exists,
    get_blacklist_key,
    delete_key
)
from api.auth_utils import SECRET_KEY, ALGORITHM
from src.utils.logger import setup_logger

# Logger'ı ayarla
logger = setup_logger(__name__, f"logs/blacklist_{datetime.now().strftime('%Y%m%d')}.log")


def get_token_jti(token: str) -> Optional[str]:
    """
    JWT token'dan jti (JWT ID) claim'ini çıkarır.
    
    jti, token'ın unique identifier'ıdır. Bu sayede aynı kullanıcının
    farklı token'larını birbirinden ayırabiliriz.
    
    Args:
        token (str): JWT token
        
    Returns:
        Optional[str]: Token'ın jti'si (yoksa None)
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # jti claim'i varsa kullan
        if "jti" in payload:
            return payload["jti"]
        
        # jti yoksa, token'ın kendisinin hash'ini kullan (fallback)
        # Bu ideal değil ama geriye dönük uyumluluk için
        import hashlib
        token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
        logger.warning("Token'da jti claim'i yok, token hash kullanılıyor")
        return token_hash
        
    except JWTError as e:
        logger.error(f"Token decode hatası: {e}")
        return None


def get_token_exp(token: str) -> Optional[int]:
    """
    JWT token'ın expiration timestamp'ini döner.
    
    Args:
        token (str): JWT token
        
    Returns:
        Optional[int]: Expiration timestamp (unix epoch), yoksa None
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("exp")
    except JWTError as e:
        logger.error(f"Token decode hatası: {e}")
        return None


def calculate_ttl(token: str) -> int:
    """
    Token'ın kalan geçerlilik süresini hesaplar (saniye).
    
    Bu, Redis TTL için kullanılır. Token expire olduğunda
    Redis'te otomatik olarak silinir.
    
    Args:
        token (str): JWT token
        
    Returns:
        int: Kalan süre (saniye), hata durumunda 3600 (1 saat)
    """
    try:
        exp = get_token_exp(token)
        
        if exp is None:
            logger.warning("Token'da exp claim'i yok, default TTL kullanılıyor")
            return 3600  # Default: 1 saat
        
        now = datetime.utcnow().timestamp()
        remaining = int(exp - now)
        
        # Negatif veya çok küçük TTL kontrolü
        if remaining <= 0:
            logger.warning("Token zaten expired, minimum TTL kullanılıyor")
            return 60  # Minimum 1 dakika
        
        logger.debug(f"Token TTL: {remaining} saniye")
        return remaining
        
    except Exception as e:
        logger.error(f"TTL hesaplama hatası: {e}")
        return 3600  # Default: 1 saat


async def add_token_to_blacklist(
    token: str,
    reason: str = "logout"
) -> bool:
    """
    Token'ı blacklist'e ekler.
    
    Best Practices:
    - Token'ın jti'sini kullan (unique ID)
    - TTL kullan (token expire olunca otomatik temizlensin)
    - Reason bilgisi kaydet (debugging için)
    
    Args:
        token (str): Blacklist'e eklenecek JWT token
        reason (str): Blacklist sebebi (örn: "logout", "password_change", "security")
        
    Returns:
        bool: İşlem başarılıysa True
        
    Raises:
        ValueError: Token geçersizse
    """
    # Redis kontrolü
    if not is_redis_available():
        logger.warning("⚠ Redis yok, blacklist yapılamıyor")
        return False
    
    # Token jti'sini al
    jti = get_token_jti(token)
    
    if not jti:
        raise ValueError("Token geçersiz veya jti alınamadı")
    
    # TTL hesapla
    ttl = calculate_ttl(token)
    
    # Blacklist key oluştur
    blacklist_key = get_blacklist_key(jti)
    
    # Redis'e ekle (TTL ile)
    blacklist_data = f"{reason}:{datetime.utcnow().isoformat()}"
    success = await set_with_ttl(blacklist_key, blacklist_data, ttl)
    
    if success:
        logger.info(f"✓ Token blacklist'e eklendi: {jti[:8]}... (reason: {reason}, TTL: {ttl}s)")
    else:
        logger.error(f"✗ Token blacklist'e eklenemedi: {jti[:8]}...")
    
    return success


async def is_token_blacklisted(token: str) -> bool:
    """
    Token'ın blacklist'te olup olmadığını kontrol eder.
    
    Bu fonksiyon, her token validation'da çağrılır.
    Performance-critical olduğu için optimize edilmiştir.
    
    Args:
        token (str): Kontrol edilecek JWT token
        
    Returns:
        bool: Token blacklist'teyse True, değilse False
        
    Note:
        Redis yoksa veya hata durumunda False döner (güvenli varsayılan).
        Bu sayede Redis çökerse sistem çalışmaya devam eder.
    """
    # Redis kontrolü
    if not is_redis_available():
        # Redis yoksa blacklist kontrolü yapılamaz
        # Güvenlik vs. availability trade-off'u
        # Burada availability'i seçiyoruz
        return False
    
    # Token jti'sini al
    jti = get_token_jti(token)
    
    if not jti:
        logger.warning("Token jti alınamadı, blacklist kontrolü yapılamıyor")
        return False
    
    # Blacklist key oluştur
    blacklist_key = get_blacklist_key(jti)
    
    # Redis'te var mı kontrol et
    is_blacklisted = await key_exists(blacklist_key)
    
    if is_blacklisted:
        logger.warning(f"⚠ Blacklisted token kullanım denemesi: {jti[:8]}...")
    
    return is_blacklisted


async def remove_token_from_blacklist(token: str) -> bool:
    """
    Token'ı blacklist'ten çıkarır.
    
    Not: Normal kullanımda gerekmez (TTL otomatik siler),
    ancak admin operasyonları veya test için kullanılabilir.
    
    Args:
        token (str): Çıkarılacak token
        
    Returns:
        bool: İşlem başarılıysa True
    """
    if not is_redis_available():
        return False
    
    jti = get_token_jti(token)
    
    if not jti:
        return False
    
    blacklist_key = get_blacklist_key(jti)
    success = await delete_key(blacklist_key)
    
    if success:
        logger.info(f"✓ Token blacklist'ten çıkarıldı: {jti[:8]}...")
    
    return success


async def blacklist_all_user_tokens(username: str) -> int:
    """
    Bir kullanıcının tüm token'larını blacklist'e ekler.
    
    Use case: Password değişikliği, hesap güvenliği ihlali.
    
    Not: Bu özellik için tüm token'ları track etmek gerekir.
    Şu anda implement edilmemiş, gelecek versiyon için placeholder.
    
    Args:
        username (str): Kullanıcı adı
        
    Returns:
        int: Blacklist'e eklenen token sayısı
        
    TODO: Implement user session tracking
    """
    logger.info(f"TODO: Kullanıcının tüm token'ları blacklist'lenecek: {username}")
    # Bu özellik için Redis'te user:tokens:username key'inde
    # aktif token listesi tutmak gerekir
    return 0


# ============================================================================
# Helper Functions
# ============================================================================

async def get_blacklist_stats() -> dict:
    """
    Blacklist istatistiklerini döner.
    
    Returns:
        dict: Blacklist istatistikleri
    """
    redis = get_redis_client()
    
    if not redis:
        return {
            "available": False,
            "total_blacklisted": 0,
            "error": "Redis not available"
        }
    
    try:
        # Blacklist key'lerini say (SCAN kullan, KEYS çok maliyetli)
        cursor = 0
        count = 0
        
        while True:
            cursor, keys = await redis.scan(
                cursor=cursor,
                match="blacklist:*",
                count=100
            )
            count += len(keys)
            
            if cursor == 0:
                break
        
        return {
            "available": True,
            "total_blacklisted": count,
            "redis_connected": True
        }
        
    except Exception as e:
        logger.error(f"Blacklist stats hatası: {e}")
        return {
            "available": False,
            "total_blacklisted": 0,
            "error": str(e)
        }


# ============================================================================
# Test ve Örnek Kullanım
# ============================================================================

if __name__ == "__main__":
    import asyncio
    from api.redis_client import connect_to_redis, close_redis_connection
    from api.auth_utils import create_access_token
    from datetime import timedelta
    
    async def test_blacklist():
        """Blacklist fonksiyonlarını test eder."""
        print("=" * 60)
        print("JWT Blacklist Test")
        print("=" * 60)
        
        try:
            # Redis'e bağlan
            await connect_to_redis()
            
            if not is_redis_available():
                print("❌ Redis yok, blacklist test edilemez")
                return
            
            print("✓ Redis bağlantısı başarılı\n")
            
            # Test token oluştur
            print("1. Test token oluşturuluyor...")
            test_token = create_access_token(
                data={"sub": "test_user", "jti": "test_jti_123"},
                expires_delta=timedelta(minutes=30)
            )
            print(f"   ✓ Token: {test_token[:30]}...\n")
            
            # Blacklist kontrolü (önce)
            print("2. Blacklist kontrolü (token temiz olmalı)...")
            is_bl_before = await is_token_blacklisted(test_token)
            print(f"   {'✗ HATA' if is_bl_before else '✓'} Blacklisted: {is_bl_before}\n")
            
            # Blacklist'e ekle
            print("3. Token blacklist'e ekleniyor...")
            added = await add_token_to_blacklist(test_token, reason="test_logout")
            print(f"   {'✓' if added else '✗'} Eklendi: {added}\n")
            
            # Blacklist kontrolü (sonra)
            print("4. Blacklist kontrolü (token blacklisted olmalı)...")
            is_bl_after = await is_token_blacklisted(test_token)
            print(f"   {'✓' if is_bl_after else '✗ HATA'} Blacklisted: {is_bl_after}\n")
            
            # Stats
            print("5. Blacklist istatistikleri...")
            stats = await get_blacklist_stats()
            print(f"   ✓ Total blacklisted: {stats['total_blacklisted']}\n")
            
            # Temizle
            print("6. Test token temizleniyor...")
            removed = await remove_token_from_blacklist(test_token)
            print(f"   {'✓' if removed else '✗'} Temizlendi: {removed}\n")
            
            # Redis'i kapat
            await close_redis_connection()
            print("✓ Test tamamlandı!")
            
        except Exception as e:
            print(f"\n❌ Test hatası: {e}")
            import traceback
            traceback.print_exc()
    
    # Test'i çalıştır
    asyncio.run(test_blacklist())

