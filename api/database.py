"""
MongoDB bağlantı yönetimi modülü.

Bu modül, FastAPI uygulaması için async MongoDB bağlantısını yönetir.
Motor (async MongoDB driver) kullanılır.
"""

from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional
import os
from src.utils.logger import setup_logger
from datetime import datetime

# Logger'ı ayarla
logger = setup_logger(__name__, f"logs/database_{datetime.now().strftime('%Y%m%d')}.log")

# Global MongoDB client
client: Optional[AsyncIOMotorClient] = None


def get_mongo_url() -> str:
    """
    MongoDB connection URL'ini config'den alır.
    
    Öncelik: Environment Variable > config.yaml > Default
    
    Returns:
        str: MongoDB connection URL
    """
    # Lazy import to avoid circular dependency
    from api.config import settings
    return settings.MONGO_URL


def get_database_name() -> str:
    """
    Veritabanı adını config'den alır.
    
    Öncelik: Environment Variable > config.yaml > Default
    
    Returns:
        str: Database adı
    """
    # Lazy import to avoid circular dependency
    from api.config import settings
    return settings.DATABASE_NAME


async def connect_to_mongo():
    """
    MongoDB'ye asenkron bağlantı kurar.
    Uygulama başlangıcında (startup event) çağrılmalıdır.
    """
    global client
    try:
        mongo_url = get_mongo_url()
        client = AsyncIOMotorClient(mongo_url)
        # Bağlantıyı test et
        await client.admin.command('ping')
        logger.info(f"MongoDB'ye başarıyla bağlanıldı: {get_database_name()}")
    except Exception as e:
        logger.error(f"MongoDB bağlantı hatası: {e}")
        raise


async def close_mongo_connection():
    """
    MongoDB bağlantısını kapatır.
    Uygulama kapanışında (shutdown event) çağrılmalıdır.
    """
    global client
    if client:
        client.close()
        logger.info("MongoDB bağlantısı kapatıldı.")


def get_database():
    """
    MongoDB database instance'ını döner.
    FastAPI dependency olarak kullanılabilir.
    
    Returns:
        AsyncIOMotorDatabase: MongoDB database instance
        
    Raises:
        RuntimeError: Client başlatılmamışsa
    """
    if client is None:
        logger.error("MongoDB client başlatılmamış!")
        raise RuntimeError("MongoDB client is not initialized. Call connect_to_mongo() first.")
    return client[get_database_name()]


async def create_indexes():
    """
    Veritabanı index'lerini oluşturur.
    Performans optimizasyonu için gerekli index'ler tanımlanır.
    """
    try:
        db = get_database()
        
        # Users collection indexes
        await db.users.create_index("username", unique=True)
        await db.users.create_index("email", unique=True)
        await db.users.create_index("created_at")
        
        # Prompt logs collection indexes
        await db.prompt_logs.create_index("user_id")
        await db.prompt_logs.create_index("username")
        await db.prompt_logs.create_index("timestamp")
        await db.prompt_logs.create_index([("user_id", 1), ("timestamp", -1)])
        
        logger.info("MongoDB index'leri başarıyla oluşturuldu.")
    except Exception as e:
        logger.error(f"Index oluşturma hatası: {e}")
        # Index oluşturma hatası kritik değil, uygulama çalışmaya devam edebilir


# Örnek kullanım ve test
if __name__ == "__main__":
    import asyncio
    
    async def test_connection():
        """MongoDB bağlantısını test eder."""
        try:
            await connect_to_mongo()
            db = get_database()
            
            # Koleksiyonları listele
            collections = await db.list_collection_names()
            logger.info(f"Mevcut koleksiyonlar: {collections}")
            
            # Index'leri oluştur
            await create_indexes()
            
            # Bağlantıyı kapat
            await close_mongo_connection()
            logger.info("Test başarılı!")
        except Exception as e:
            logger.error(f"Test başarısız: {e}")
    
    # Test'i çalıştır
    asyncio.run(test_connection())

