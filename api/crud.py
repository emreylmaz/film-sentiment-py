"""
CRUD (Create, Read, Update, Delete) operasyonları.

Bu modül, MongoDB ile etkileşim için veritabanı operasyonlarını içerir.
Kullanıcı yönetimi ve prompt logging için CRUD fonksiyonları.
"""

from motor.motor_asyncio import AsyncIOMotorDatabase
from api.models import UserCreate, UserInDB, PromptLogCreate, PromptLogInDB
from api.auth_utils import get_password_hash
from datetime import datetime
from typing import Optional, List
from bson import ObjectId
from src.utils.logger import setup_logger

# Logger'ı ayarla
logger = setup_logger(__name__, f"logs/crud_{datetime.now().strftime('%Y%m%d')}.log")


# ==================== User CRUD ====================

async def create_user(db: AsyncIOMotorDatabase, user: UserCreate) -> UserInDB:
    """
    Yeni kullanıcı oluşturur.
    
    Args:
        db (AsyncIOMotorDatabase): MongoDB database instance
        user (UserCreate): Oluşturulacak kullanıcı bilgileri
        
    Returns:
        UserInDB: Oluşturulan kullanıcı (ID ve metadata ile)
        
    Raises:
        Exception: Veritabanı hatası durumunda
    """
    try:
        # User modelini dict'e çevir ve password'ı hash'le
        user_dict = user.model_dump()
        user_dict["hashed_password"] = get_password_hash(user_dict.pop("password"))
        user_dict["created_at"] = datetime.utcnow()
        user_dict["is_active"] = True
        
        # Veritabanına ekle
        result = await db.users.insert_one(user_dict)
        user_dict["id"] = str(result.inserted_id)
        
        logger.info(f"Yeni kullanıcı oluşturuldu: {user.username} (ID: {user_dict['id']})")
        return UserInDB(**user_dict)
    
    except Exception as e:
        logger.error(f"Kullanıcı oluşturma hatası: {e}")
        raise


async def get_user_by_username(db: AsyncIOMotorDatabase, username: str) -> Optional[UserInDB]:
    """
    Kullanıcı adına göre kullanıcı getirir.
    
    Args:
        db (AsyncIOMotorDatabase): MongoDB database instance
        username (str): Aranacak kullanıcı adı
        
    Returns:
        Optional[UserInDB]: Kullanıcı bulunursa UserInDB, bulunamazsa None
    """
    try:
        user = await db.users.find_one({"username": username.lower()})
        
        if user:
            user["id"] = str(user.pop("_id"))
            logger.info(f"Kullanıcı bulundu: {username}")
            return UserInDB(**user)
        
        logger.warning(f"Kullanıcı bulunamadı: {username}")
        return None
    
    except Exception as e:
        logger.error(f"Kullanıcı getirme hatası: {e}")
        raise


async def get_user_by_email(db: AsyncIOMotorDatabase, email: str) -> Optional[UserInDB]:
    """
    Email adresine göre kullanıcı getirir.
    
    Args:
        db (AsyncIOMotorDatabase): MongoDB database instance
        email (str): Aranacak email adresi
        
    Returns:
        Optional[UserInDB]: Kullanıcı bulunursa UserInDB, bulunamazsa None
    """
    try:
        user = await db.users.find_one({"email": email.lower()})
        
        if user:
            user["id"] = str(user.pop("_id"))
            logger.info(f"Email ile kullanıcı bulundu: {email}")
            return UserInDB(**user)
        
        return None
    
    except Exception as e:
        logger.error(f"Email ile kullanıcı getirme hatası: {e}")
        raise


async def get_user_by_id(db: AsyncIOMotorDatabase, user_id: str) -> Optional[UserInDB]:
    """
    ID'ye göre kullanıcı getirir.
    
    Args:
        db (AsyncIOMotorDatabase): MongoDB database instance
        user_id (str): MongoDB ObjectId (string format)
        
    Returns:
        Optional[UserInDB]: Kullanıcı bulunursa UserInDB, bulunamazsa None
    """
    try:
        if not ObjectId.is_valid(user_id):
            logger.warning(f"Geçersiz ObjectId: {user_id}")
            return None
        
        user = await db.users.find_one({"_id": ObjectId(user_id)})
        
        if user:
            user["id"] = str(user.pop("_id"))
            return UserInDB(**user)
        
        return None
    
    except Exception as e:
        logger.error(f"ID ile kullanıcı getirme hatası: {e}")
        raise


async def update_user(db: AsyncIOMotorDatabase, user_id: str, update_data: dict) -> bool:
    """
    Kullanıcı bilgilerini günceller.
    
    Args:
        db (AsyncIOMotorDatabase): MongoDB database instance
        user_id (str): Güncellenecek kullanıcının ID'si
        update_data (dict): Güncellenecek alanlar
        
    Returns:
        bool: Güncelleme başarılıysa True, değilse False
    """
    try:
        if not ObjectId.is_valid(user_id):
            logger.warning(f"Geçersiz ObjectId: {user_id}")
            return False
        
        result = await db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": update_data}
        )
        
        if result.modified_count > 0:
            logger.info(f"Kullanıcı güncellendi: {user_id}")
            return True
        
        logger.warning(f"Kullanıcı güncellenemedi veya değişiklik yok: {user_id}")
        return False
    
    except Exception as e:
        logger.error(f"Kullanıcı güncelleme hatası: {e}")
        raise


async def delete_user(db: AsyncIOMotorDatabase, user_id: str) -> bool:
    """
    Kullanıcıyı siler (soft delete - is_active=False).
    
    Args:
        db (AsyncIOMotorDatabase): MongoDB database instance
        user_id (str): Silinecek kullanıcının ID'si
        
    Returns:
        bool: Silme başarılıysa True, değilse False
    """
    try:
        if not ObjectId.is_valid(user_id):
            return False
        
        # Soft delete - is_active'i False yap
        result = await db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"is_active": False}}
        )
        
        if result.modified_count > 0:
            logger.info(f"Kullanıcı deaktive edildi: {user_id}")
            return True
        
        return False
    
    except Exception as e:
        logger.error(f"Kullanıcı silme hatası: {e}")
        raise


# ==================== Prompt Log CRUD ====================

async def create_prompt_log(
    db: AsyncIOMotorDatabase,
    log: PromptLogCreate,
    user_id: str,
    username: str,
    ip_address: Optional[str] = None
) -> str:
    """
    Yeni prompt log kaydı oluşturur.
    
    Args:
        db (AsyncIOMotorDatabase): MongoDB database instance
        log (PromptLogCreate): Log bilgileri (text, sentiment, confidence, etc.)
        user_id (str): Kullanıcı ID
        username (str): Kullanıcı adı
        ip_address (Optional[str]): İstek yapan IP adresi
        
    Returns:
        str: Oluşturulan log'un ID'si
        
    Raises:
        Exception: Veritabanı hatası durumunda
    """
    try:
        # Log modelini dict'e çevir ve metadata ekle
        log_dict = log.model_dump()
        log_dict["user_id"] = user_id
        log_dict["username"] = username
        log_dict["timestamp"] = datetime.utcnow()
        log_dict["ip_address"] = ip_address
        
        # Veritabanına ekle
        result = await db.prompt_logs.insert_one(log_dict)
        log_id = str(result.inserted_id)
        
        logger.info(f"Prompt log kaydedildi: User={username}, Sentiment={log.sentiment}, ID={log_id}")
        return log_id
    
    except Exception as e:
        logger.error(f"Prompt log oluşturma hatası: {e}")
        raise


async def get_user_prompt_logs(
    db: AsyncIOMotorDatabase,
    user_id: str,
    limit: int = 50,
    skip: int = 0
) -> List[PromptLogInDB]:
    """
    Belirli bir kullanıcının prompt log'larını getirir.
    
    Args:
        db (AsyncIOMotorDatabase): MongoDB database instance
        user_id (str): Kullanıcı ID
        limit (int): Maksimum getir ilecek log sayısı (default: 50)
        skip (int): Atlanacak log sayısı (pagination için, default: 0)
        
    Returns:
        List[PromptLogInDB]: Log listesi (en yeniden eskiye sıralı)
    """
    try:
        cursor = db.prompt_logs.find({"user_id": user_id}) \
            .sort("timestamp", -1) \
            .skip(skip) \
            .limit(limit)
        
        logs = []
        async for log in cursor:
            log["id"] = str(log.pop("_id"))
            logs.append(PromptLogInDB(**log))
        
        logger.info(f"Kullanıcı log'ları getirildi: {user_id}, Sayı: {len(logs)}")
        return logs
    
    except Exception as e:
        logger.error(f"Kullanıcı log'ları getirme hatası: {e}")
        raise


async def get_prompt_log_by_id(db: AsyncIOMotorDatabase, log_id: str) -> Optional[PromptLogInDB]:
    """
    ID'ye göre prompt log getirir.
    
    Args:
        db (AsyncIOMotorDatabase): MongoDB database instance
        log_id (str): Log ID
        
    Returns:
        Optional[PromptLogInDB]: Log bulunursa PromptLogInDB, bulunamazsa None
    """
    try:
        if not ObjectId.is_valid(log_id):
            return None
        
        log = await db.prompt_logs.find_one({"_id": ObjectId(log_id)})
        
        if log:
            log["id"] = str(log.pop("_id"))
            return PromptLogInDB(**log)
        
        return None
    
    except Exception as e:
        logger.error(f"Log getirme hatası: {e}")
        raise


async def get_user_statistics(db: AsyncIOMotorDatabase, user_id: str) -> dict:
    """
    Kullanıcının sentiment analizi istatistiklerini getirir.
    
    Args:
        db (AsyncIOMotorDatabase): MongoDB database instance
        user_id (str): Kullanıcı ID
        
    Returns:
        dict: İstatistik bilgileri (total_predictions, positive_count, etc.)
    """
    try:
        pipeline = [
            {"$match": {"user_id": user_id}},
            {
                "$group": {
                    "_id": None,
                    "total_predictions": {"$sum": 1},
                    "positive_count": {
                        "$sum": {"$cond": [{"$eq": ["$sentiment", "positive"]}, 1, 0]}
                    },
                    "negative_count": {
                        "$sum": {"$cond": [{"$eq": ["$sentiment", "negative"]}, 1, 0]}
                    },
                    "average_confidence": {"$avg": "$confidence"},
                    "average_prediction_time_ms": {"$avg": "$prediction_time_ms"}
                }
            }
        ]
        
        result = await db.prompt_logs.aggregate(pipeline).to_list(length=1)
        
        if result:
            stats = result[0]
            stats.pop("_id")
            logger.info(f"Kullanıcı istatistikleri: {user_id}, Total: {stats['total_predictions']}")
            return stats
        
        # Hiç log yoksa default değerler dön
        return {
            "total_predictions": 0,
            "positive_count": 0,
            "negative_count": 0,
            "average_confidence": 0.0,
            "average_prediction_time_ms": 0.0
        }
    
    except Exception as e:
        logger.error(f"İstatistik getirme hatası: {e}")
        raise


async def delete_prompt_log(db: AsyncIOMotorDatabase, log_id: str) -> bool:
    """
    Prompt log'u siler.
    
    Args:
        db (AsyncIOMotorDatabase): MongoDB database instance
        log_id (str): Silinecek log ID
        
    Returns:
        bool: Silme başarılıysa True, değilse False
    """
    try:
        if not ObjectId.is_valid(log_id):
            return False
        
        result = await db.prompt_logs.delete_one({"_id": ObjectId(log_id)})
        
        if result.deleted_count > 0:
            logger.info(f"Prompt log silindi: {log_id}")
            return True
        
        return False
    
    except Exception as e:
        logger.error(f"Log silme hatası: {e}")
        raise


# Örnek kullanım (test amaçlı)
if __name__ == "__main__":
    import asyncio
    from api.database import connect_to_mongo, get_database, close_mongo_connection
    
    async def test_crud_operations():
        """CRUD operasyonlarını test eder."""
        try:
            # MongoDB'ye bağlan
            await connect_to_mongo()
            db = get_database()
            
            # Test kullanıcısı oluştur
            test_user = UserCreate(
                username="test_user_crud",
                email="test_crud@example.com",
                password="TestPass123",
                full_name="Test CRUD User",
                organization="Test Org",
                role="user"
            )
            
            print("1. Kullanıcı oluşturuluyor...")
            created_user = await create_user(db, test_user)
            print(f"   ✓ Kullanıcı oluşturuldu: {created_user.username} (ID: {created_user.id})")
            
            print("\n2. Kullanıcı getiriliyor (username)...")
            fetched_user = await get_user_by_username(db, test_user.username)
            print(f"   ✓ Kullanıcı bulundu: {fetched_user.username if fetched_user else 'Bulunamadı'}")
            
            print("\n3. Prompt log oluşturuluyor...")
            test_log = PromptLogCreate(
                text="This is a test movie review",
                sentiment="positive",
                confidence=0.95,
                prediction_time_ms=25.0
            )
            log_id = await create_prompt_log(db, test_log, created_user.id, created_user.username, "127.0.0.1")
            print(f"   ✓ Log oluşturuldu: {log_id}")
            
            print("\n4. Kullanıcı log'ları getiriliyor...")
            logs = await get_user_prompt_logs(db, created_user.id)
            print(f"   ✓ {len(logs)} log bulundu")
            
            print("\n5. Kullanıcı istatistikleri getiriliyor...")
            stats = await get_user_statistics(db, created_user.id)
            print(f"   ✓ İstatistikler: {stats}")
            
            print("\n6. Test verilerini temizleme...")
            await delete_user(db, created_user.id)
            await delete_prompt_log(db, log_id)
            print("   ✓ Test verileri temizlendi")
            
            # Bağlantıyı kapat
            await close_mongo_connection()
            print("\n✓ Tüm testler başarılı!")
            
        except Exception as e:
            print(f"\n✗ Test hatası: {e}")
            await close_mongo_connection()
    
    # Test'i çalıştır
    asyncio.run(test_crud_operations())

