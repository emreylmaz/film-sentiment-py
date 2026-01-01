"""
FastAPI dependency fonksiyonları.

Bu modül, endpoint'lerde kullanılacak dependency injection 
fonksiyonlarını içerir (authentication, database, etc.).
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from typing import Optional
from api.auth_utils import decode_access_token
from api.models import UserInDB
from api.crud import get_user_by_username
from api.database import get_database
from api.blacklist import is_token_blacklisted  # JWT Blacklist kontrolü
from motor.motor_asyncio import AsyncIOMotorDatabase
from src.utils.logger import setup_logger
from datetime import datetime

# Logger'ı ayarla
logger = setup_logger(__name__, f"logs/dependencies_{datetime.now().strftime('%Y%m%d')}.log")

# OAuth2 scheme tanımla
# tokenUrl, login endpoint'inin path'idir
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncIOMotorDatabase = Depends(get_database)
) -> UserInDB:
    """
    JWT token'dan current user'ı çıkarır ve döner.
    Protected endpoint'lerde dependency olarak kullanılır.
    
    Args:
        token (str): JWT access token (Authorization header'dan otomatik alınır)
        db (AsyncIOMotorDatabase): MongoDB database instance
        
    Returns:
        UserInDB: Authenticated kullanıcı bilgileri
        
    Raises:
        HTTPException: Token geçersiz veya kullanıcı bulunamazsa 401 Unauthorized
        
    Example:
        ```python
        @app.get("/protected")
        async def protected_route(current_user: UserInDB = Depends(get_current_user)):
            return {"message": f"Hello {current_user.username}"}
        ```
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # 1. Token'ı decode et
        payload = decode_access_token(token)
        
        if payload is None:
            logger.warning("Token decode edildi ama payload None")
            raise credentials_exception
        
        # 2. Token blacklist'te mi kontrol et (Redis)
        is_blacklisted = await is_token_blacklisted(token)
        
        if is_blacklisted:
            logger.warning("Blacklisted token kullanım denemesi!")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked (logged out)",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # 3. Username'i payload'dan al
        username: Optional[str] = payload.get("sub")
        
        if username is None:
            logger.warning("Token payload'ında 'sub' field'ı bulunamadı")
            raise credentials_exception
        
        logger.info(f"Token doğrulandı: {username}")
    
    except HTTPException:
        # HTTPException'ları (blacklist) olduğu gibi geçir
        raise
    except JWTError as e:
        logger.error(f"JWT doğrulama hatası: {e}")
        raise credentials_exception
    
    # Kullanıcıyı veritabanından getir
    user = await get_user_by_username(db, username)
    
    if user is None:
        logger.warning(f"Token geçerli ama kullanıcı bulunamadı: {username}")
        raise credentials_exception
    
    # Kullanıcı aktif mi kontrol et
    if not user.is_active:
        logger.warning(f"Deaktive edilmiş kullanıcı erişim denemesi: {username}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated"
        )
    
    logger.info(f"Kullanıcı authenticated: {username}")
    return user


async def get_current_active_user(
    current_user: UserInDB = Depends(get_current_user)
) -> UserInDB:
    """
    Current user'ın aktif olduğunu doğrular.
    
    Bu dependency, get_current_user'ın üzerine ekstra bir doğrulama katmanı ekler.
    Şu anda get_current_user zaten aktiflik kontrolü yapıyor, ancak
    gelecekte farklı doğrulama mantıkları eklenebilir.
    
    Args:
        current_user (UserInDB): get_current_user dependency'sinden gelen kullanıcı
        
    Returns:
        UserInDB: Aktif kullanıcı
        
    Raises:
        HTTPException: Kullanıcı aktif değilse 403 Forbidden
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


async def get_current_admin_user(
    current_user: UserInDB = Depends(get_current_user)
) -> UserInDB:
    """
    Current user'ın admin olduğunu doğrular.
    Admin-only endpoint'ler için kullanılır.
    
    Args:
        current_user (UserInDB): get_current_user dependency'sinden gelen kullanıcı
        
    Returns:
        UserInDB: Admin kullanıcı
        
    Raises:
        HTTPException: Kullanıcı admin değilse 403 Forbidden
    """
    if current_user.role != "admin":
        logger.warning(f"Admin endpoint'e yetkisiz erişim: {current_user.username} (role: {current_user.role})")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    
    logger.info(f"Admin erişimi: {current_user.username}")
    return current_user


async def get_current_analyst_or_admin(
    current_user: UserInDB = Depends(get_current_user)
) -> UserInDB:
    """
    Current user'ın analyst veya admin olduğunu doğrular.
    İstatistik ve analiz endpoint'leri için kullanılır.
    
    Args:
        current_user (UserInDB): get_current_user dependency'sinden gelen kullanıcı
        
    Returns:
        UserInDB: Analyst veya admin kullanıcı
        
    Raises:
        HTTPException: Kullanıcı analyst veya admin değilse 403 Forbidden
    """
    allowed_roles = ["analyst", "admin"]
    
    if current_user.role not in allowed_roles:
        logger.warning(f"Analyst endpoint'e yetkisiz erişim: {current_user.username} (role: {current_user.role})")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Analyst or admin privileges required"
        )
    
    logger.info(f"Analyst erişimi: {current_user.username} (role: {current_user.role})")
    return current_user


# Optional: Rate limiting dependency (gelecek için)
class RateLimiter:
    """
    Rate limiting için dependency class.
    Belirli bir süre içinde maksimum istek sayısını sınırlar.
    
    Not: Şu anda basit bir placeholder. Production'da Redis veya 
    memory cache ile implement edilmelidir.
    """
    def __init__(self, times: int = 10, seconds: int = 60):
        """
        Args:
            times (int): İzin verilen maksimum istek sayısı
            seconds (int): Süre (saniye)
        """
        self.times = times
        self.seconds = seconds
    
    async def __call__(self, current_user: UserInDB = Depends(get_current_user)):
        """
        Rate limiting kontrolü yapar.
        
        Şu anda sadece log atar, gerçek rate limiting uygulanmamıştır.
        Gelecekte Redis ile implement edilebilir.
        """
        # TODO: Redis ile rate limiting implement et
        logger.debug(f"Rate limit check: {current_user.username} ({self.times} requests / {self.seconds}s)")
        return current_user


# Örnek kullanım
if __name__ == "__main__":
    print("=== Dependencies Module ===")
    print("Bu modül FastAPI dependency fonksiyonlarını içerir.")
    print("\nKullanım örnekleri:")
    print("""
    # 1. Protected endpoint (sadece authenticated kullanıcılar)
    @app.get("/protected")
    async def protected_route(current_user: UserInDB = Depends(get_current_user)):
        return {"message": f"Hello {current_user.username}"}
    
    # 2. Admin-only endpoint
    @app.get("/admin/users")
    async def admin_users(admin: UserInDB = Depends(get_current_admin_user)):
        return {"message": "Admin area"}
    
    # 3. Analyst veya admin endpoint
    @app.get("/analytics/stats")
    async def get_stats(user: UserInDB = Depends(get_current_analyst_or_admin)):
        return {"message": "Analytics data"}
    
    # 4. Rate limiting ile protected endpoint
    rate_limiter = RateLimiter(times=10, seconds=60)
    
    @app.get("/limited")
    async def limited_route(user: UserInDB = Depends(rate_limiter)):
        return {"message": "Rate limited endpoint"}
    """)
    print("\n✓ Dependencies hazır!")

