"""
Veri Yükleme ve Hazırlama Modülü

Bu modül, IMDB dataset'ini yükler ve train/test setlerine ayırır.
"""

import pandas as pd
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from src.utils.logger import setup_logger

# Logger oluştur
logger = setup_logger(__name__)


def load_data(file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """
    CSV dosyasından veri yükler.
    
    Args:
        file_path: CSV dosyasının yolu
        encoding: Dosya encoding'i (varsayılan: utf-8)
        
    Returns:
        Yüklenmiş pandas DataFrame
        
    Raises:
        FileNotFoundError: Dosya bulunamazsa
        pd.errors.EmptyDataError: Dosya boşsa
        
    Örnek:
        >>> df = load_data("data/IMDB Dataset.csv")
        >>> print(df.shape)
        (50000, 2)
    """
    try:
        logger.info(f"Veri yükleniyor: {file_path}")
        df = pd.read_csv(file_path, encoding=encoding)
        
        logger.info(f"✓ Veri başarıyla yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
        logger.info(f"Sütunlar: {list(df.columns)}")
        
        return df
        
    except FileNotFoundError:
        logger.error(f"✗ Dosya bulunamadı: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"✗ Dosya boş: {file_path}")
        raise
    except Exception as e:
        logger.error(f"✗ Veri yükleme hatası: {str(e)}")
        raise


def validate_data(df: pd.DataFrame, required_columns: list = None) -> bool:
    """
    DataFrame'in geçerliliğini kontrol eder.
    
    Args:
        df: Kontrol edilecek DataFrame
        required_columns: Zorunlu sütun isimleri (None ise ['review', 'sentiment'])
        
    Returns:
        True eğer veri geçerliyse
        
    Raises:
        ValueError: Veri geçerli değilse
    """
    if required_columns is None:
        required_columns = ['review', 'sentiment']
    
    # Boş DataFrame kontrolü
    if df.empty:
        raise ValueError("DataFrame boş!")
    
    # Sütun kontrolü
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Eksik sütunlar: {missing_columns}")
    
    # Null değer kontrolü
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        logger.warning(f"⚠ Null değerler bulundu:\n{null_counts[null_counts > 0]}")
    
    # Sentiment değerleri kontrolü
    if 'sentiment' in df.columns:
        unique_sentiments = df['sentiment'].unique()
        logger.info(f"Sentiment sınıfları: {unique_sentiments}")
        
        # Sınıf dağılımı
        sentiment_dist = df['sentiment'].value_counts()
        logger.info(f"Sınıf dağılımı:\n{sentiment_dist}")
    
    logger.info("✓ Veri validasyonu başarılı")
    return True


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_column: Optional[str] = 'sentiment'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Veriyi train ve test setlerine ayırır.
    
    Args:
        df: Bölünecek DataFrame
        test_size: Test seti oranı (0-1 arası)
        random_state: Rastgelelik seed'i (tekrarlanabilirlik için)
        stratify_column: Stratified split için kullanılacak sütun
        
    Returns:
        (train_df, test_df) tuple'ı
        
    Örnek:
        >>> df = load_data("data/IMDB Dataset.csv")
        >>> train_df, test_df = split_data(df, test_size=0.2)
        >>> print(f"Train: {len(train_df)}, Test: {len(test_df)}")
        Train: 40000, Test: 10000
    """
    logger.info(f"Veri bölünüyor: test_size={test_size}, random_state={random_state}")
    
    # Stratify parametresi
    stratify = df[stratify_column] if stratify_column and stratify_column in df.columns else None
    
    # Train-test split
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    
    logger.info(f"✓ Train seti: {len(train_df)} örnek ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"✓ Test seti: {len(test_df)} örnek ({len(test_df)/len(df)*100:.1f}%)")
    
    # Stratified split kontrolü
    if stratify is not None:
        logger.info("Sınıf dağılımları:")
        logger.info(f"  Train: {train_df[stratify_column].value_counts(normalize=True).to_dict()}")
        logger.info(f"  Test:  {test_df[stratify_column].value_counts(normalize=True).to_dict()}")
    
    return train_df, test_df


def get_basic_stats(df: pd.DataFrame) -> dict:
    """
    DataFrame hakkında temel istatistikler döndürür.
    
    Args:
        df: Analiz edilecek DataFrame
        
    Returns:
        İstatistikler içeren dictionary
    """
    stats = {
        'total_samples': len(df),
        'columns': list(df.columns),
        'null_values': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.to_dict()
    }
    
    # Review uzunluk istatistikleri
    if 'review' in df.columns:
        df['review_length'] = df['review'].astype(str).str.len()
        stats['review_length_stats'] = {
            'mean': df['review_length'].mean(),
            'median': df['review_length'].median(),
            'min': df['review_length'].min(),
            'max': df['review_length'].max()
        }
    
    # Sentiment dağılımı
    if 'sentiment' in df.columns:
        stats['sentiment_distribution'] = df['sentiment'].value_counts().to_dict()
    
    return stats


# Test ve örnek kullanım
if __name__ == "__main__":
    import yaml
    
    # Config'i yükle
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Veriyi yükle
    df = load_data(config['data']['raw_path'])
    
    # Validasyon
    validate_data(df)
    
    # İstatistikler
    stats = get_basic_stats(df)
    print("\n=== Veri İstatistikleri ===")
    print(f"Toplam örnekler: {stats['total_samples']}")
    print(f"Sütunlar: {stats['columns']}")
    
    if 'review_length_stats' in stats:
        print(f"\nYorum uzunluk istatistikleri:")
        for key, value in stats['review_length_stats'].items():
            print(f"  {key}: {value:.0f}")
    
    # Train-test split
    train_df, test_df = split_data(
        df,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )
    
    print(f"\n✓ Data loader başarıyla test edildi!")

