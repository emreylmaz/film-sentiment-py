"""
Authentication endpoint'leri.

Bu modül, kullanıcı kayıt, giriş ve profil yönetimi için 
API endpoint'lerini içerir.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from motor.motor_asyncio import AsyncIOMotorDatabase
from api.models import UserCreate, UserResponse, Token, UserInDB
from api.crud import create_user, get_user_by_username, get_user_by_email
from api.auth_utils import verify_password, create_access_token
from api.database import get_database
from api.dependencies import get_current_user, oauth2_scheme
from api.blacklist import add_token_to_blacklist, is_redis_available  # Blacklist import
from src.utils.logger import setup_logger
from datetime import datetime, timedelta

# Logger'ı ayarla
logger = setup_logger(__name__, f"logs/auth_{datetime.now().strftime('%Y%m%d')}.log")

# Router oluştur
router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
    responses={
        401: {"description": "Unauthorized - Invalid credentials"},
        403: {"description": "Forbidden - Insufficient permissions"},
        422: {"description": "Validation Error"}
    }
)


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user: UserCreate,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Yeni kullanıcı kaydı oluşturur.
    
    **İşlem Adımları:**
    1. Username ve email'in daha önce kullanılmadığını kontrol eder
    2. Password'ü hash'ler
    3. Kullanıcıyı veritabanına ekler
    4. Hassas bilgiler olmadan kullanıcı bilgilerini döner
    
    **Gereksinimler:**
    - Username: 3-50 karakter, sadece harf, rakam ve underscore
    - Email: Geçerli email formatı
    - Password: Minimum 8 karakter, en az 1 harf ve 1 rakam içermeli
    - Full name: 2-100 karakter
    
    **Dönüş:**
    - 201 Created: Kullanıcı başarıyla oluşturuldu
    - 400 Bad Request: Username veya email zaten kullanımda
    - 422 Unprocessable Entity: Validation hatası
    """
    try:
        # Username kontrolü
        existing_user = await get_user_by_username(db, user.username)
        if existing_user:
            logger.warning(f"Kayıt denemesi - Username zaten var: {user.username}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Username '{user.username}' is already registered"
            )
        
        # Email kontrolü
        existing_email = await get_user_by_email(db, user.email)
        if existing_email:
            logger.warning(f"Kayıt denemesi - Email zaten var: {user.email}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Email '{user.email}' is already registered"
            )
        
        # Yeni kullanıcı oluştur
        created_user = await create_user(db, user)
        
        logger.info(f"Yeni kullanıcı kaydedildi: {created_user.username} (ID: {created_user.id})")
        
        # Response dönerken hassas bilgileri çıkar
        return UserResponse(
            id=created_user.id,
            username=created_user.username,
            email=created_user.email,
            full_name=created_user.full_name,
            organization=created_user.organization,
            role=created_user.role,
            created_at=created_user.created_at,
            is_active=created_user.is_active
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Kayıt hatası: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during registration"
        )


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Kullanıcı girişi yapar ve JWT token döner.
    
    **OAuth2 Uyumlu Endpoint:**
    Bu endpoint OAuth2 password flow standardına uygundur.
    Form data olarak username ve password bekler.
    
    **İşlem Adımları:**
    1. Username ile kullanıcıyı veritabanından getirir
    2. Password'ü doğrular
    3. JWT access token oluşturur
    4. Token'ı döner
    
    **Dönüş:**
    - 200 OK: Başarılı giriş, access token döner
    - 401 Unauthorized: Yanlış username veya password
    - 403 Forbidden: Hesap deaktive edilmiş
    
    **Token Kullanımı:**
    ```
    Authorization: Bearer <access_token>
    ```
    
    **Token Geçerlilik Süresi:** 24 saat (1440 dakika)
    """
    try:
        # Kullanıcıyı getir
        user = await get_user_by_username(db, form_data.username)
        
        # Kullanıcı kontrolü ve password doğrulama
        if not user or not verify_password(form_data.password, user.hashed_password):
            logger.warning(f"Başarısız giriş denemesi: {form_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Hesap aktif mi kontrol et
        if not user.is_active:
            logger.warning(f"Deaktive hesaba giriş denemesi: {form_data.username}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is deactivated. Please contact support."
            )
        
        # Access token oluştur
        access_token_expires = timedelta(minutes=1440)  # 24 saat
        access_token = create_access_token(
            data={"sub": user.username},
            expires_delta=access_token_expires
        )
        
        logger.info(f"Başarılı giriş: {user.username}")
        
        return Token(access_token=access_token, token_type="bearer")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Giriş hatası: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during login"
        )


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: UserInDB = Depends(get_current_user)):
    """
    Mevcut authenticated kullanıcının bilgilerini döner.
    
    **Protected Endpoint:**
    Bu endpoint'e erişmek için geçerli bir JWT token gereklidir.
    
    **Authorization Header:**
    ```
    Authorization: Bearer <access_token>
    ```
    
    **Dönüş:**
    - 200 OK: Kullanıcı bilgileri
    - 401 Unauthorized: Token geçersiz veya süresi dolmuş
    - 403 Forbidden: Hesap deaktive edilmiş
    
    **Kullanım Amacı:**
    - Kullanıcı profil sayfası
    - Token doğrulama
    - Kullanıcı bilgilerini güncelleme öncesi mevcut bilgileri alma
    """
    logger.info(f"Profil bilgisi istendi: {current_user.username}")
    
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        organization=current_user.organization,
        role=current_user.role,
        created_at=current_user.created_at,
        is_active=current_user.is_active
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(current_user: UserInDB = Depends(get_current_user)):
    """
    Mevcut token ile yeni bir token oluşturur (token refresh).
    
    **Protected Endpoint:**
    Mevcut geçerli bir token ile yeni bir token alabilirsiniz.
    
    **Kullanım Amacı:**
    - Token süresi dolmadan önce yeni token almak
    - Sürekli aktif kullanıcılar için kesintisiz erişim
    
    **Dönüş:**
    - 200 OK: Yeni access token
    - 401 Unauthorized: Mevcut token geçersiz
    
    **Not:** 
    Mevcut token'ın süresi dolmadan önce yeni token alınmalıdır.
    Token dolduğunda `/auth/login` ile tekrar giriş yapılmalıdır.
    """
    try:
        # Yeni token oluştur
        access_token_expires = timedelta(minutes=1440)  # 24 saat
        access_token = create_access_token(
            data={"sub": current_user.username},
            expires_delta=access_token_expires
        )
        
        logger.info(f"Token yenilendi: {current_user.username}")
        
        return Token(access_token=access_token, token_type="bearer")
    
    except Exception as e:
        logger.error(f"Token yenileme hatası: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during token refresh"
        )


@router.post("/logout")
async def logout(
    current_user: UserInDB = Depends(get_current_user),
    token: str = Depends(oauth2_scheme)
):
    """
    Kullanıcı çıkışı yapar ve token'ı blacklist'e ekler.
    
    **Best Practice: Redis Blacklist**
    Token, Redis'e TTL ile eklenir. Token expire olduğunda otomatik silinir.
    Blacklist'teki token'lar kullanılamaz ve 401 Unauthorized döner.
    
    **Process:**
    1. Token'ı decode et ve jti (unique ID) al
    2. Token'ı Redis blacklist'e ekle (TTL = token expiry)
    3. Client-side'da token'ı sil
    4. Başarı mesajı dön
    
    **Redis Yoksa:**
    - Uyarı loglanır
    - Client-side logout yine de yapılır
    - Ancak token tekrar kullanılabilir (güvenlik riski)
    
    **Client-side yapılması gerekenler:**
    1. Local storage'dan token'ı sil
    2. Authorization header'ını kaldır
    3. Login sayfasına yönlendir
    
    **Dönüş:**
    - 200 OK: Logout başarılı mesajı
    - 401 Unauthorized: Token geçersiz
    - 500 Internal Server Error: Blacklist hatası (nadiren)
    """
    try:
        # Redis kontrolü
        if not is_redis_available():
            logger.warning(f"⚠ Redis yok, token blacklist yapılamıyor: {current_user.username}")
            return {
                "message": "Logged out (client-side only)",
                "detail": "Please remove the token from client storage",
                "warning": "Server-side token revocation not available (Redis disconnected)"
            }
        
        # Token'ı blacklist'e ekle
        blacklisted = await add_token_to_blacklist(token, reason="user_logout")
        
        if blacklisted:
            logger.info(f"✓ Logout başarılı (blacklist): {current_user.username}")
            return {
                "message": "Successfully logged out",
                "detail": "Token has been revoked and added to blacklist",
                "username": current_user.username
            }
        else:
            # Blacklist başarısız ama logout devam etsin
            logger.error(f"✗ Token blacklist başarısız: {current_user.username}")
            return {
                "message": "Logged out (client-side only)",
                "detail": "Please remove the token from client storage",
                "warning": "Server-side token revocation failed"
            }
    
    except Exception as e:
        logger.error(f"Logout hatası: {e}")
        # Logout'u engelleme, en azından client-side logout olsun
        return {
            "message": "Logged out (client-side only)",
            "detail": "Please remove the token from client storage",
            "error": "Server-side logout encountered an error"
        }


# Örnek kullanım ve test endpoint'i (development için)
@router.get("/test")
async def test_auth():
    """
    Authentication modülünün çalıştığını test eder.
    
    **Public Endpoint:**
    Bu endpoint authentication gerektirmez.
    
    **Dönüş:**
    - Modül durumu ve kullanılabilir endpoint'ler
    """
    return {
        "status": "ok",
        "module": "authentication",
        "endpoints": {
            "register": "POST /auth/register",
            "login": "POST /auth/login",
            "me": "GET /auth/me (protected)",
            "refresh": "POST /auth/refresh (protected)",
            "logout": "POST /auth/logout (protected)"
        },
        "info": "Authentication module is working properly"
    }


# Örnek kullanım
if __name__ == "__main__":
    print("=== Authentication Routes ===")
    print("\nEndpoint'ler:")
    print("1. POST /auth/register - Yeni kullanıcı kaydı")
    print("2. POST /auth/login - Kullanıcı girişi (JWT token döner)")
    print("3. GET /auth/me - Mevcut kullanıcı bilgileri (protected)")
    print("4. POST /auth/refresh - Token yenileme (protected)")
    print("5. POST /auth/logout - Çıkış yapma (protected)")
    print("6. GET /auth/test - Modül test endpoint'i (public)")
    print("\n✓ Authentication routes hazır!")

