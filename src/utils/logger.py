"""
Loglama Sistemi

Bu modül, proje genelinde kullanılacak loglama altyapısını sağlar.
Hem dosyaya hem de konsola log yazma desteği sunar.
"""

import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Yapılandırılmış logger oluşturur.
    
    Args:
        name: Logger ismi (genellikle modül adı)
        log_file: Log dosyası ismi (None ise otomatik oluşturulur)
        level: Log seviyesi (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Log dosyalarının kaydedileceği klasör
        
    Returns:
        Yapılandırılmış Logger nesnesi
        
    Örnek:
        >>> logger = setup_logger("train_model")
        >>> logger.info("Model eğitimi başladı")
    """
    # Logger oluştur
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Eğer daha önce handler eklenmişse tekrar ekleme
    if logger.handlers:
        return logger
    
    # Formatter oluştur - Türkçe karakter desteği ile
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (eğer log_file belirtilmişse)
    if log_file or log_dir:
        # Log klasörünü oluştur
        os.makedirs(log_dir, exist_ok=True)
        
        # Log dosya adı oluştur
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            log_file = f"{name}_{timestamp}.log"
        
        log_path = os.path.join(log_dir, log_file)
        
        # UTF-8 encoding ile file handler
        file_handler = logging.FileHandler(
            log_path, 
            mode='a',
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Mevcut bir logger'ı getirir veya yeni oluşturur.
    
    Args:
        name: Logger ismi
        
    Returns:
        Logger nesnesi
    """
    logger = logging.getLogger(name)
    
    # Eğer logger henüz yapılandırılmamışsa, varsayılan ayarlarla oluştur
    if not logger.handlers:
        return setup_logger(name)
    
    return logger


class LoggerMixin:
    """
    Sınıflara logger özelliği ekleyen mixin.
    
    Kullanım:
        class MyClass(LoggerMixin):
            def my_method(self):
                self.logger.info("İşlem başarılı")
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Sınıf için logger döndürür."""
        name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return get_logger(name)


# Test fonksiyonu
if __name__ == "__main__":
    # Test logger
    test_logger = setup_logger("test", level=logging.DEBUG)
    
    test_logger.debug("Bu bir debug mesajı")
    test_logger.info("Bu bir info mesajı")
    test_logger.warning("Bu bir uyarı mesajı")
    test_logger.error("Bu bir hata mesajı")
    test_logger.critical("Bu kritik bir mesaj")
    
    print("\n✓ Logger başarıyla test edildi!")
    print(f"✓ Log dosyası: logs/test_{datetime.now().strftime('%Y%m%d')}.log")


