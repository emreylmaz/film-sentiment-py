"""
Merkezi Konfigürasyon Yönetimi

Bu modül, config.yaml ve environment variables'ı birleştirir.
Öncelik sırası:
1. Environment Variables (.env veya sistem)
2. config.yaml dosyası
3. Default değerler

Kullanım:
    from api.config import settings
    
    print(settings.MONGO_URL)
    print(settings.REDIS_URL)
    print(settings.jwt.secret_key)
"""

import os
import yaml
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Basit logger (circular import önlemek için)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Proje kök dizini
PROJECT_ROOT = Path(__file__).parent.parent


def load_yaml_config(config_path: str = "config.yaml") -> dict:
    """
    config.yaml dosyasını yükler.
    
    Args:
        config_path: config.yaml dosyasının yolu
        
    Returns:
        dict: YAML içeriği, dosya yoksa boş dict
    """
    full_path = PROJECT_ROOT / config_path
    
    if not full_path.exists():
        logger.warning(f"config.yaml bulunamadı: {full_path}")
        return {}
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
            logger.info(f"✓ config.yaml yüklendi: {full_path}")
            return config
    except Exception as e:
        logger.error(f"config.yaml okuma hatası: {e}")
        return {}


def get_env_or_yaml(
    env_key: str,
    yaml_value: Optional[str] = None,
    default: Optional[str] = None
) -> Optional[str]:
    """
    Önce environment variable, sonra yaml, en son default değeri döner.
    
    Args:
        env_key: Environment variable adı
        yaml_value: config.yaml'dan gelen değer
        default: Default değer
        
    Returns:
        Değer (öncelik: env > yaml > default)
    """
    # 1. Environment variable (en yüksek öncelik)
    env_value = os.getenv(env_key)
    if env_value is not None:
        return env_value
    
    # 2. YAML değeri
    if yaml_value is not None:
        return yaml_value
    
    # 3. Default değer
    return default


# ============================================================================
# Konfigürasyon Dataclass'ları
# ============================================================================

@dataclass
class DatabaseConfig:
    """MongoDB konfigürasyonu."""
    url: str = "mongodb://localhost:27017"
    name: str = "imdb_sentiment_db"
    
    def __post_init__(self):
        logger.debug(f"MongoDB URL: {self.url[:30]}...")


@dataclass
class RedisConfig:
    """Redis konfigürasyonu."""
    url: str = "redis://localhost:6379"
    password: Optional[str] = None
    db: int = 0
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    
    def __post_init__(self):
        logger.debug(f"Redis URL: {self.url}")


@dataclass
class JWTConfig:
    """JWT authentication konfigürasyonu."""
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440  # 24 saat
    
    def __post_init__(self):
        if self.secret_key == "your-secret-key-change-in-production":
            logger.warning("⚠ JWT SECRET_KEY default değerde! Production'da değiştirin!")


@dataclass
class CORSConfig:
    """CORS konfigürasyonu."""
    allowed_origins: List[str] = field(default_factory=lambda: ["http://localhost:3000"])
    allow_credentials: bool = True
    allow_methods: List[str] = field(default_factory=lambda: ["*"])
    allow_headers: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class ModelConfig:
    """Model konfigürasyonu."""
    model_path: str = "models/model.pkl"
    vectorizer_path: str = "models/vectorizer.pkl"
    metadata_path: str = "models/metadata.json"


@dataclass
class APIConfig:
    """API konfigürasyonu."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    log_level: str = "INFO"


@dataclass
class DataConfig:
    """Veri konfigürasyonu."""
    raw_data_path: str = "data/IMDB Dataset.csv"
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class PreprocessingConfig:
    """Preprocessing konfigürasyonu."""
    max_features: int = 5000
    ngram_range: tuple = (1, 2)
    min_df: int = 5
    max_df: float = 0.8


# ============================================================================
# Ana Settings Sınıfı
# ============================================================================

class Settings:
    """
    Merkezi konfigürasyon sınıfı.
    
    config.yaml ve environment variables'ı birleştirir.
    Environment variables her zaman önceliklidir.
    
    Kullanım:
        from api.config import settings
        
        # Doğrudan erişim
        print(settings.MONGO_URL)
        
        # Nested config
        print(settings.jwt.secret_key)
        print(settings.redis.url)
    """
    
    def __init__(self):
        """Konfigürasyonu yükler."""
        # YAML config'i yükle
        self._yaml_config = load_yaml_config()
        
        # Alt konfigürasyonları oluştur
        self._load_database_config()
        self._load_redis_config()
        self._load_jwt_config()
        self._load_cors_config()
        self._load_model_config()
        self._load_api_config()
        self._load_data_config()
        self._load_preprocessing_config()
        
        logger.info("✓ Tüm konfigürasyonlar yüklendi")
    
    def _load_database_config(self):
        """MongoDB konfigürasyonunu yükler."""
        yaml_db = self._yaml_config.get("database", {})
        
        self.database = DatabaseConfig(
            url=get_env_or_yaml(
                "MONGO_URL",
                yaml_db.get("url"),
                "mongodb://localhost:27017"
            ),
            name=get_env_or_yaml(
                "DATABASE_NAME",
                yaml_db.get("name"),
                "imdb_sentiment_db"
            )
        )
        
        # Shortcut
        self.MONGO_URL = self.database.url
        self.DATABASE_NAME = self.database.name
    
    def _load_redis_config(self):
        """Redis konfigürasyonunu yükler."""
        yaml_redis = self._yaml_config.get("redis", {})
        
        self.redis = RedisConfig(
            url=get_env_or_yaml(
                "REDIS_URL",
                yaml_redis.get("url"),
                "redis://localhost:6379"
            ),
            password=get_env_or_yaml(
                "REDIS_PASSWORD",
                yaml_redis.get("password"),
                None
            ),
            db=int(get_env_or_yaml(
                "REDIS_DB",
                str(yaml_redis.get("db", 0)),
                "0"
            )),
            socket_timeout=yaml_redis.get("socket_timeout", 5),
            socket_connect_timeout=yaml_redis.get("socket_connect_timeout", 5)
        )
        
        # Shortcut
        self.REDIS_URL = self.redis.url
    
    def _load_jwt_config(self):
        """JWT konfigürasyonunu yükler."""
        yaml_jwt = self._yaml_config.get("jwt", {})
        
        self.jwt = JWTConfig(
            secret_key=get_env_or_yaml(
                "JWT_SECRET_KEY",
                yaml_jwt.get("secret_key"),
                "your-secret-key-change-in-production"
            ),
            algorithm=get_env_or_yaml(
                "JWT_ALGORITHM",
                yaml_jwt.get("algorithm"),
                "HS256"
            ),
            access_token_expire_minutes=int(get_env_or_yaml(
                "ACCESS_TOKEN_EXPIRE_MINUTES",
                str(yaml_jwt.get("access_token_expire_minutes", 1440)),
                "1440"
            ))
        )
        
        # Shortcuts
        self.SECRET_KEY = self.jwt.secret_key
        self.JWT_ALGORITHM = self.jwt.algorithm
        self.ACCESS_TOKEN_EXPIRE_MINUTES = self.jwt.access_token_expire_minutes
    
    def _load_cors_config(self):
        """CORS konfigürasyonunu yükler."""
        yaml_cors = self._yaml_config.get("cors", {})
        
        # Environment'tan virgülle ayrılmış liste
        env_origins = os.getenv("CORS_ORIGINS")
        if env_origins:
            origins = [o.strip() for o in env_origins.split(",")]
        else:
            origins = yaml_cors.get("allowed_origins", ["http://localhost:3000"])
        
        self.cors = CORSConfig(
            allowed_origins=origins,
            allow_credentials=yaml_cors.get("allow_credentials", True),
            allow_methods=yaml_cors.get("allow_methods", ["*"]),
            allow_headers=yaml_cors.get("allow_headers", ["*"])
        )
        
        # Shortcut
        self.CORS_ORIGINS = self.cors.allowed_origins
    
    def _load_model_config(self):
        """Model konfigürasyonunu yükler."""
        yaml_model = self._yaml_config.get("model", {})
        yaml_paths = yaml_model.get("paths", {})
        
        self.model = ModelConfig(
            model_path=yaml_paths.get("model", "models/model.pkl"),
            vectorizer_path=yaml_paths.get("vectorizer", "models/vectorizer.pkl"),
            metadata_path=yaml_paths.get("metadata", "models/metadata.json")
        )
    
    def _load_api_config(self):
        """API konfigürasyonunu yükler."""
        yaml_api = self._yaml_config.get("api", {})
        
        self.api = APIConfig(
            host=get_env_or_yaml("API_HOST", yaml_api.get("host"), "0.0.0.0"),
            port=int(get_env_or_yaml("API_PORT", str(yaml_api.get("port", 8000)), "8000")),
            debug=get_env_or_yaml("DEBUG", str(yaml_api.get("debug", False)), "False").lower() == "true",
            log_level=get_env_or_yaml("LOG_LEVEL", yaml_api.get("log_level"), "INFO")
        )
    
    def _load_data_config(self):
        """Veri konfigürasyonunu yükler."""
        yaml_data = self._yaml_config.get("data", {})
        
        self.data = DataConfig(
            raw_data_path=yaml_data.get("raw_data_path", "data/IMDB Dataset.csv"),
            test_size=yaml_data.get("test_size", 0.2),
            random_state=yaml_data.get("random_state", 42)
        )
    
    def _load_preprocessing_config(self):
        """Preprocessing konfigürasyonunu yükler."""
        yaml_prep = self._yaml_config.get("preprocessing", {})
        yaml_tfidf = yaml_prep.get("tfidf", {})
        
        ngram = yaml_tfidf.get("ngram_range", [1, 2])
        
        self.preprocessing = PreprocessingConfig(
            max_features=yaml_tfidf.get("max_features", 5000),
            ngram_range=tuple(ngram) if isinstance(ngram, list) else (1, 2),
            min_df=yaml_tfidf.get("min_df", 5),
            max_df=yaml_tfidf.get("max_df", 0.8)
        )
    
    def reload(self):
        """Konfigürasyonu yeniden yükler."""
        logger.info("Konfigürasyon yeniden yükleniyor...")
        self.__init__()
    
    def to_dict(self) -> dict:
        """Tüm konfigürasyonu dictionary olarak döner (hassas veriler maskeli)."""
        return {
            "database": {
                "url": self.MONGO_URL[:30] + "..." if len(self.MONGO_URL) > 30 else self.MONGO_URL,
                "name": self.DATABASE_NAME
            },
            "redis": {
                "url": self.REDIS_URL,
                "db": self.redis.db
            },
            "jwt": {
                "algorithm": self.JWT_ALGORITHM,
                "expire_minutes": self.ACCESS_TOKEN_EXPIRE_MINUTES,
                "secret_key": "***MASKED***"
            },
            "cors": {
                "origins": self.CORS_ORIGINS
            },
            "api": {
                "host": self.api.host,
                "port": self.api.port,
                "debug": self.api.debug
            }
        }
    
    def __repr__(self):
        return f"<Settings mongo={self.MONGO_URL[:20]}... redis={self.REDIS_URL}>"


# ============================================================================
# Singleton Instance
# ============================================================================

# Global settings instance (singleton pattern)
settings = Settings()


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Configuration Test")
    print("=" * 60)
    
    print(f"\n[MongoDB]")
    print(f"   URL: {settings.MONGO_URL}")
    print(f"   Database: {settings.DATABASE_NAME}")
    
    print(f"\n[Redis]")
    print(f"   URL: {settings.REDIS_URL}")
    print(f"   DB: {settings.redis.db}")
    
    print(f"\n[JWT]")
    print(f"   Algorithm: {settings.JWT_ALGORITHM}")
    print(f"   Expire: {settings.ACCESS_TOKEN_EXPIRE_MINUTES} minutes")
    print(f"   Secret: ***MASKED***")
    
    print(f"\n[CORS]")
    print(f"   Origins: {settings.CORS_ORIGINS}")
    
    print(f"\n[API]")
    print(f"   Host: {settings.api.host}")
    print(f"   Port: {settings.api.port}")
    print(f"   Debug: {settings.api.debug}")
    
    print(f"\n[Model]")
    print(f"   Path: {settings.model.model_path}")
    
    print(f"\n[Data]")
    print(f"   Path: {settings.data.raw_data_path}")
    
    print("\n" + "=" * 60)
    print("[OK] Konfigurasyon testi basarili!")
    print("=" * 60)

