"""
Metin Ön İşleme ve Vektörizasyon Modülü

Bu modül, film yorumlarını temizler ve TF-IDF vektörlerine dönüştürür.
"""

import re
import pickle
from typing import List, Optional, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.logger import setup_logger

# Logger oluştur
logger = setup_logger(__name__)


class TextPreprocessor:
    """
    Metin ön işleme ve vektörizasyon sınıfı.
    
    Bu sınıf, HTML taglerini temizleme, küçük harfe çevirme,
    noktalama işaretlerini kaldırma ve TF-IDF vektörizasyonu gibi
    işlemleri gerçekleştirir.
    
    Özellikler:
        - HTML tag temizleme
        - Küçük harfe çevirme
        - Noktalama işareti temizleme
        - TF-IDF vektörizasyon
        - Fit/transform ayrımı
    """
    
    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple = (1, 2),
        min_df: int = 5,
        max_df: float = 0.8,
        stop_words: str = 'english'
    ):
        """
        TextPreprocessor sınıfını başlatır.
        
        Args:
            max_features: Maksimum özellik sayısı (kelime sayısı)
            ngram_range: N-gram aralığı (unigram, bigram vb.)
            min_df: Minimum doküman frekansı
            max_df: Maksimum doküman frekansı (0-1 arası oran)
            stop_words: Stop words dili ('english', 'turkish' veya None)
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.stop_words = stop_words
        
        # Vectorizer'ı başlat
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words=stop_words,
            lowercase=True,  # Otomatik küçük harfe çevirme
            strip_accents='unicode'  # Aksan işaretlerini kaldır
        )
        
        self.is_fitted = False
        
        logger.info(f"TextPreprocessor oluşturuldu: max_features={max_features}, "
                   f"ngram_range={ngram_range}, min_df={min_df}, max_df={max_df}")
    
    @staticmethod
    def clean_html(text: str) -> str:
        """
        HTML taglerini temizler.
        
        Args:
            text: Temizlenecek metin
            
        Returns:
            HTML tagleri kaldırılmış metin
        """
        # HTML taglerini kaldır
        text = re.sub(r'<[^>]+>', ' ', text)
        # <br />, <br/>, <br> gibi satır sonlarını kaldır
        text = re.sub(r'<br\s*/?\s*>', ' ', text, flags=re.IGNORECASE)
        return text
    
    @staticmethod
    def clean_special_chars(text: str, keep_apostrophe: bool = True) -> str:
        """
        Özel karakterleri ve fazla boşlukları temizler.
        
        Args:
            text: Temizlenecek metin
            keep_apostrophe: Apostrof karakterini koru (örn: don't, it's)
            
        Returns:
            Temizlenmiş metin
        """
        # Sadece harf, rakam ve (opsiyonel) apostrof bırak
        if keep_apostrophe:
            text = re.sub(r"[^a-zA-Z0-9\s']", ' ', text)
        else:
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Fazla boşlukları tek boşluğa indir
        text = re.sub(r'\s+', ' ', text)
        
        # Baş ve sondaki boşlukları kaldır
        text = text.strip()
        
        return text
    
    def clean_text(self, text: str) -> str:
        """
        Metni tamamen temizler (HTML, özel karakterler vb.).
        
        Args:
            text: Ham metin
            
        Returns:
            Temizlenmiş metin
        """
        if not isinstance(text, str):
            text = str(text)
        
        # HTML taglerini temizle
        text = self.clean_html(text)
        
        # Küçük harfe çevir
        text = text.lower()
        
        # Özel karakterleri temizle
        text = self.clean_special_chars(text)
        
        return text
    
    def clean_texts(self, texts: List[str]) -> List[str]:
        """
        Birden fazla metni temizler.
        
        Args:
            texts: Metin listesi
            
        Returns:
            Temizlenmiş metin listesi
        """
        logger.info(f"Metinler temizleniyor: {len(texts)} örnek")
        cleaned = [self.clean_text(text) for text in texts]
        logger.info("✓ Metin temizleme tamamlandı")
        return cleaned
    
    def fit(self, texts: List[str]) -> 'TextPreprocessor':
        """
        Vectorizer'ı eğitir (vocabulary oluşturur).
        
        Args:
            texts: Eğitim metinleri
            
        Returns:
            self (method chaining için)
        """
        logger.info(f"Vectorizer eğitiliyor: {len(texts)} örnek")
        
        # Metinleri temizle
        cleaned_texts = self.clean_texts(texts)
        
        # Vectorizer'ı fit et
        self.vectorizer.fit(cleaned_texts)
        self.is_fitted = True
        
        # Vocabulary bilgisi
        vocab_size = len(self.vectorizer.vocabulary_)
        logger.info(f"✓ Vectorizer eğitildi: {vocab_size} kelime vocabulary")
        
        # En önemli kelimelerin bir kısmını göster
        if vocab_size > 0:
            feature_names = self.vectorizer.get_feature_names_out()
            sample_words = list(feature_names[:10])
            logger.info(f"Örnek kelimeler: {sample_words}")
        
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Metinleri TF-IDF vektörlerine dönüştürür.
        
        Args:
            texts: Dönüştürülecek metinler
            
        Returns:
            TF-IDF vektör matrisi
            
        Raises:
            ValueError: Eğer vectorizer henüz fit edilmemişse
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer henüz eğitilmedi! Önce fit() metodunu çağırın.")
        
        logger.info(f"Metinler vektörlere dönüştürülüyor: {len(texts)} örnek")
        
        # Metinleri temizle
        cleaned_texts = self.clean_texts(texts)
        
        # Transform
        vectors = self.vectorizer.transform(cleaned_texts)
        
        logger.info(f"✓ Vektörizasyon tamamlandı: shape={vectors.shape}")
        
        return vectors
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit ve transform işlemlerini birlikte yapar.
        
        Args:
            texts: Metinler
            
        Returns:
            TF-IDF vektör matrisi
        """
        self.fit(texts)
        return self.transform(texts)
    
    def save(self, filepath: str) -> None:
        """
        Preprocessor'ı dosyaya kaydeder.
        
        Args:
            filepath: Kayıt yolu (örn: models/vectorizer.pkl)
        """
        logger.info(f"Preprocessor kaydediliyor: {filepath}")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        logger.info(f"✓ Preprocessor kaydedildi: {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'TextPreprocessor':
        """
        Preprocessor'ı dosyadan yükler.
        
        Args:
            filepath: Yükleme yolu
            
        Returns:
            Yüklenmiş TextPreprocessor nesnesi
        """
        logger.info(f"Preprocessor yükleniyor: {filepath}")
        
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        
        logger.info(f"✓ Preprocessor yüklendi: {filepath}")
        return preprocessor
    
    def get_feature_names(self) -> List[str]:
        """
        Vocabulary'deki özellik isimlerini döndürür.
        
        Returns:
            Özellik isimleri listesi
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer henüz eğitilmedi!")
        
        return list(self.vectorizer.get_feature_names_out())
    
    def get_config(self) -> dict:
        """
        Preprocessor konfigürasyonunu döndürür.
        
        Returns:
            Konfigürasyon dictionary'si
        """
        return {
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'stop_words': self.stop_words,
            'is_fitted': self.is_fitted,
            'vocabulary_size': len(self.vectorizer.vocabulary_) if self.is_fitted else 0
        }


# Test ve örnek kullanım
if __name__ == "__main__":
    # Test metinleri
    sample_texts = [
        "<br />This movie was absolutely <b>fantastic</b>! Great acting.",
        "Terrible film, waste of time and money!!!",
        "The plot was interesting, but the ending was disappointing.",
        "Amazing cinematography and a wonderful story. Highly recommend!",
        "I fell asleep during this boring movie. Not recommended."
    ]
    
    print("=== Test Metinleri ===")
    for i, text in enumerate(sample_texts, 1):
        print(f"{i}. {text}")
    
    # Preprocessor oluştur
    preprocessor = TextPreprocessor(
        max_features=50,  # Test için küçük değer
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0
    )
    
    # Fit ve transform
    print("\n=== Fit ve Transform ===")
    vectors = preprocessor.fit_transform(sample_texts)
    
    print(f"\nVektör shape: {vectors.shape}")
    print(f"Vocabulary boyutu: {len(preprocessor.get_feature_names())}")
    
    # Örnek kelimeler
    feature_names = preprocessor.get_feature_names()[:20]
    print(f"\nİlk 20 özellik: {feature_names}")
    
    # Config
    print("\n=== Konfigürasyon ===")
    config = preprocessor.get_config()
    for key, value in config.items():
        print(f"{key}: {value}")
    
    print("\n✓ Preprocessor başarıyla test edildi!")


