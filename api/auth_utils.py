"""
Authentication utility fonksiyonları.

Bu modül, JWT token oluşturma/doğrulama ve password hashing 
işlemleri için fonksiyonlar içerir.
"""

from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
import os
from src.utils.logger import setup_logger

# Logger'ı ayarla
logger = setup_logger(__name__, f"logs/auth_{datetime.now().strftime('%Y%m%d')}.log")

# JWT ayarları - config'den yüklenir
# Öncelik: Environment Variable > config.yaml > Default
def _get_jwt_settings():
    """JWT ayarlarını config'den yükler (lazy loading)."""
    from api.config import settings
    return settings.jwt

# Bu değerler modül yüklenirken set edilir
# Ama config.py'deki settings singleton olduğu için her zaman güncel kalır
SECRET_KEY = None
ALGORITHM = None
ACCESS_TOKEN_EXPIRE_MINUTES = None

def _init_jwt_settings():
    """JWT ayarlarını initialize eder."""
    global SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
    jwt_config = _get_jwt_settings()
    SECRET_KEY = jwt_config.secret_key
    ALGORITHM = jwt_config.algorithm
    ACCESS_TOKEN_EXPIRE_MINUTES = jwt_config.access_token_expire_minutes

# İlk yükleme
_init_jwt_settings()

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Düz metin şifreyi hash'lenmiş şifre ile karşılaştırır.
    
    Args:
        plain_password (str): Kullanıcının girdiği şifre
        hashed_password (str): Veritabanında saklanan hash'lenmiş şifre
        
    Returns:
        bool: Şifreler eşleşiyorsa True, değilse False
    """
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.error(f"Password verification hatası: {e}")
        return False


def get_password_hash(password: str) -> str:
    """
    Düz metin şifreyi bcrypt ile hash'ler.
    
    Args:
        password (str): Hash'lenecek şifre
        
    Returns:
        str: Hash'lenmiş şifre
    """
    try:
        return pwd_context.hash(password)
    except Exception as e:
        logger.error(f"Password hashing hatası: {e}")
        raise


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    JWT access token oluşturur.
    
    **Best Practice: jti (JWT ID) Ekleme**
    Token'a unique ID (jti) eklenir. Bu sayede:
    - Token'lar birbirinden ayırt edilebilir
    - Blacklist sistemi çalışabilir (logout için)
    - Aynı kullanıcının farklı session'ları track edilebilir
    
    Args:
        data (dict): Token'a eklenecek payload verisi (örn: {"sub": "username"})
        expires_delta (Optional[timedelta]): Token geçerlilik süresi (None ise default kullanılır)
        
    Returns:
        str: JWT token string
        
    Example:
        >>> token = create_access_token(data={"sub": "emre_yilmaz"})
        >>> print(token)
        'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...'
    """
    import uuid
    
    to_encode = data.copy()
    
    # Token expiration time hesapla
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    # Expiration time ve jti'yi payload'a ekle
    to_encode.update({
        "exp": expire,
        "jti": str(uuid.uuid4())  # Unique JWT ID (blacklist için gerekli)
    })
    
    try:
        # Token'ı encode et
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        logger.info(f"JWT token oluşturuldu. Kullanıcı: {data.get('sub')}, jti: {to_encode['jti'][:8]}..., Geçerlilik: {expire}")
        return encoded_jwt
    except Exception as e:
        logger.error(f"JWT token oluşturma hatası: {e}")
        raise


def decode_access_token(token: str) -> Optional[dict]:
    """
    JWT token'ı decode eder ve payload'ı döner.
    
    Args:
        token (str): Decode edilecek JWT token
        
    Returns:
        Optional[dict]: Token geçerliyse payload, değilse None
        
    Raises:
        JWTError: Token geçersiz veya süresi dolmuşsa
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        
        if username is None:
            logger.warning("Token payload'ında 'sub' field'ı bulunamadı")
            return None
        
        logger.info(f"JWT token başarıyla decode edildi. Kullanıcı: {username}")
        return payload
    
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token'ın süresi dolmuş")
        raise
    except JWTError as e:
        logger.error(f"JWT decode hatası: {e}")
        raise


def verify_token(token: str) -> Optional[str]:
    """
    Token'ı doğrular ve kullanıcı adını döner.
    
    Args:
        token (str): Doğrulanacak JWT token
        
    Returns:
        Optional[str]: Token geçerliyse username, değilse None
    """
    try:
        payload = decode_access_token(token)
        if payload is None:
            return None
        return payload.get("sub")
    except Exception:
        return None


def generate_password_reset_token(email: str, expires_delta: timedelta = timedelta(hours=1)) -> str:
    """
    Şifre sıfırlama için token oluşturur.
    
    Args:
        email (str): Kullanıcı email adresi
        expires_delta (timedelta): Token geçerlilik süresi (default: 1 saat)
        
    Returns:
        str: Password reset token
        
    Note:
        Bu fonksiyon gelecekte şifre sıfırlama özelliği için kullanılabilir.
    """
    data = {"sub": email, "type": "password_reset"}
    expire = datetime.utcnow() + expires_delta
    data.update({"exp": expire})
    
    try:
        token = jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)
        logger.info(f"Password reset token oluşturuldu: {email}")
        return token
    except Exception as e:
        logger.error(f"Password reset token oluşturma hatası: {e}")
        raise


def verify_password_reset_token(token: str) -> Optional[str]:
    """
    Password reset token'ı doğrular ve email'i döner.
    
    Args:
        token (str): Password reset token
        
    Returns:
        Optional[str]: Token geçerliyse email, değilse None
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        token_type: str = payload.get("type")
        
        if email is None or token_type != "password_reset":
            logger.warning("Geçersiz password reset token payload")
            return None
        
        return email
    except jwt.ExpiredSignatureError:
        logger.warning("Password reset token'ın süresi dolmuş")
        return None
    except JWTError as e:
        logger.error(f"Password reset token doğrulama hatası: {e}")
        return None


# Örnek kullanım ve test
if __name__ == "__main__":
    print("=== Authentication Utils Test ===\n")
    
    # 1. Password Hashing Test
    print("1. Password Hashing Test:")
    test_password = "SecurePassword123"
    hashed = get_password_hash(test_password)
    print(f"   Original: {test_password}")
    print(f"   Hashed: {hashed[:50]}...")
    
    # 2. Password Verification Test
    print("\n2. Password Verification Test:")
    is_correct = verify_password(test_password, hashed)
    print(f"   Correct password: {is_correct}")
    is_wrong = verify_password("WrongPassword", hashed)
    print(f"   Wrong password: {is_wrong}")
    
    # 3. JWT Token Creation Test
    print("\n3. JWT Token Creation Test:")
    token = create_access_token(data={"sub": "emre_yilmaz"})
    print(f"   Token: {token[:50]}...")
    
    # 4. JWT Token Decode Test
    print("\n4. JWT Token Decode Test:")
    try:
        payload = decode_access_token(token)
        print(f"   Payload: {payload}")
        print(f"   Username: {payload.get('sub')}")
    except Exception as e:
        print(f"   Hata: {e}")
    
    # 5. Token Verification Test
    print("\n5. Token Verification Test:")
    username = verify_token(token)
    print(f"   Verified username: {username}")
    
    # 6. Expired Token Test (1 saniye geçerli)
    print("\n6. Short-lived Token Test (2 seconds):")
    short_token = create_access_token(
        data={"sub": "test_user"},
        expires_delta=timedelta(seconds=2)
    )
    print(f"   Token created: {short_token[:50]}...")
    print(f"   Verifying immediately: {verify_token(short_token)}")
    
    # 7. Password Reset Token Test
    print("\n7. Password Reset Token Test:")
    reset_token = generate_password_reset_token("test@example.com")
    print(f"   Reset token: {reset_token[:50]}...")
    email = verify_password_reset_token(reset_token)
    print(f"   Verified email: {email}")
    
    print("\n✓ Tüm testler tamamlandı!")

