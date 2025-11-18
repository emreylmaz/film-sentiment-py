"""
Model Eğitim Modülü

Bu modül, sentiment analizi için makine öğrenmesi modellerini eğitir.
Logistic Regression ve Random Forest modelleri desteklenir.
"""

import os
import json
import pickle
import time
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import yaml
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import load_data, split_data, validate_data
from src.preprocessor import TextPreprocessor
from src.evaluate_model import ModelEvaluator, evaluate_model
from src.utils.logger import setup_logger

# Logger oluştur
logger = setup_logger(__name__)


class SentimentModelTrainer:
    """
    Sentiment analizi model eğitim sınıfı.
    
    Veri yükleme, ön işleme, model eğitimi ve değerlendirme
    işlemlerini koordine eder.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Trainer'ı başlatır.
        
        Args:
            config_path: Konfigürasyon dosyası yolu
        """
        # Config yükle - proje root dizinine göre path ayarla
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        if not os.path.isabs(config_path):
            # Relative path ise, proje root'una göre ayarla
            config_path = os.path.join(project_root, config_path)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Config içindeki relative path'leri project root'a göre düzelt
        
        # Data path'ini düzelt
        if not os.path.isabs(self.config['data']['raw_path']):
            self.config['data']['raw_path'] = os.path.join(
                project_root, 
                self.config['data']['raw_path']
            )
        
        # Model save path'ini düzelt
        if not os.path.isabs(self.config['training']['model_save_path']):
            self.config['training']['model_save_path'] = os.path.join(
                project_root,
                self.config['training']['model_save_path']
            )
        
        # Log path'ini düzelt
        if not os.path.isabs(self.config['training']['log_path']):
            self.config['training']['log_path'] = os.path.join(
                project_root,
                self.config['training']['log_path']
            )
        
        logger.info("SentimentModelTrainer başlatıldı")
        logger.info(f"Konfigürasyon: {config_path}")
        logger.info(f"Veri path: {self.config['data']['raw_path']}")
        
        self.preprocessor = None
        self.models = {}
        self.metrics = {}
    
    def load_and_prepare_data(self) -> Tuple:
        """
        Veriyi yükler ve train/test setlerine ayırır.
        
        Returns:
            (train_df, test_df) tuple'ı
        """
        logger.info("=" * 60)
        logger.info("Veri Yükleme ve Hazırlama")
        logger.info("=" * 60)
        
        # Veriyi yükle
        df = load_data(self.config['data']['raw_path'])
        
        # Validasyon
        validate_data(df)
        
        # Train-test split
        train_df, test_df = split_data(
            df,
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state']
        )
        
        return train_df, test_df
    
    def create_preprocessor(self) -> TextPreprocessor:
        """
        Preprocessor oluşturur.
        
        Returns:
            TextPreprocessor nesnesi
        """
        logger.info("=" * 60)
        logger.info("Preprocessor Oluşturma")
        logger.info("=" * 60)
        
        preprocessor = TextPreprocessor(
            max_features=self.config['preprocessing']['max_features'],
            ngram_range=tuple(self.config['preprocessing']['ngram_range']),
            min_df=self.config['preprocessing']['min_df'],
            max_df=self.config['preprocessing']['max_df'],
            stop_words='english'
        )
        
        return preprocessor
    
    def train_logistic_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> LogisticRegression:
        """
        Logistic Regression modeli eğitir.
        
        Args:
            X_train: Eğitim özellikleri
            y_train: Eğitim etiketleri
            
        Returns:
            Eğitilmiş model
        """
        logger.info("=" * 60)
        logger.info("Logistic Regression Eğitimi")
        logger.info("=" * 60)
        
        # Model parametreleri
        params = self.config['models']['logistic_regression']
        logger.info(f"Parametreler: {params}")
        
        # Model oluştur
        model = LogisticRegression(**params)
        
        # Eğitim
        start_time = time.time()
        logger.info("Eğitim başladı...")
        
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        logger.info(f"✓ Eğitim tamamlandı: {training_time:.2f} saniye")
        
        return model
    
    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> RandomForestClassifier:
        """
        Random Forest modeli eğitir.
        
        Args:
            X_train: Eğitim özellikleri
            y_train: Eğitim etiketleri
            
        Returns:
            Eğitilmiş model
        """
        logger.info("=" * 60)
        logger.info("Random Forest Eğitimi")
        logger.info("=" * 60)
        
        # Model parametreleri
        params = self.config['models']['random_forest']
        logger.info(f"Parametreler: {params}")
        
        # Model oluştur
        model = RandomForestClassifier(**params)
        
        # Eğitim
        start_time = time.time()
        logger.info("Eğitim başladı...")
        
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        logger.info(f"✓ Eğitim tamamlandı: {training_time:.2f} saniye")
        
        return model
    
    def train_all_models(self) -> Dict[str, Any]:
        """
        Tüm modelleri eğitir ve değerlendirir.
        
        Returns:
            Model metrikleri dictionary'si
        """
        # Veri hazırlama
        train_df, test_df = self.load_and_prepare_data()
        
        # Preprocessor oluştur ve eğit
        self.preprocessor = self.create_preprocessor()
        
        logger.info("=" * 60)
        logger.info("Vektörizasyon")
        logger.info("=" * 60)
        
        # Train verisi için fit_transform
        X_train = self.preprocessor.fit_transform(train_df['review'].tolist())
        y_train = train_df['sentiment'].values
        
        # Test verisi için sadece transform
        X_test = self.preprocessor.transform(test_df['review'].tolist())
        y_test = test_df['sentiment'].values
        
        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"X_test shape: {X_test.shape}")
        
        # Logistic Regression
        lr_model = self.train_logistic_regression(X_train, y_train)
        lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
        self.models['logistic_regression'] = lr_model
        self.metrics['logistic_regression'] = lr_metrics
        
        # Random Forest
        rf_model = self.train_random_forest(X_train, y_train)
        rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
        self.models['random_forest'] = rf_model
        self.metrics['random_forest'] = rf_metrics
        
        # En iyi modeli seç
        logger.info("=" * 60)
        logger.info("Model Karşılaştırma")
        logger.info("=" * 60)
        
        evaluator = ModelEvaluator()
        best_model_name, best_score = evaluator.compare_models(
            [lr_metrics, rf_metrics],
            metric_name='f1_score'
        )
        
        return {
            'logistic_regression': lr_metrics,
            'random_forest': rf_metrics,
            'best_model': best_model_name,
            'best_score': best_score
        }
    
    def save_model(
        self,
        model_name: str = 'logistic_regression',
        model_filename: str = 'model.pkl'
    ) -> None:
        """
        Modeli ve preprocessor'ı kaydeder.
        
        Args:
            model_name: Kaydedilecek model adı
            model_filename: Model dosya adı
        """
        logger.info("=" * 60)
        logger.info("Model Kaydetme")
        logger.info("=" * 60)
        
        save_path = self.config['training']['model_save_path']
        os.makedirs(save_path, exist_ok=True)
        
        # Model kaydet
        model_path = os.path.join(save_path, model_filename)
        model = self.models[model_name]
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"✓ Model kaydedildi: {model_path}")
        
        # Preprocessor kaydet
        preprocessor_path = os.path.join(save_path, 'vectorizer.pkl')
        self.preprocessor.save(preprocessor_path)
        
        # Metadata kaydet
        metadata = {
            'model_name': model_name,
            'model_type': type(model).__name__,
            'version': '1.0.0',
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': self.metrics[model_name],
            'config': {
                'preprocessing': self.config['preprocessing'],
                'model_params': self.config['models'][model_name]
            },
            'vocabulary_size': len(self.preprocessor.get_feature_names())
        }
        
        metadata_path = os.path.join(save_path, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Metadata kaydedildi: {metadata_path}")
        
        logger.info(f"✓ Tüm dosyalar kaydedildi: {save_path}")


def main():
    """Ana eğitim fonksiyonu."""
    logger.info("=" * 60)
    logger.info("IMDB Sentiment Analizi - Model Eğitimi")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    try:
        # Trainer oluştur
        trainer = SentimentModelTrainer()
        
        # Tüm modelleri eğit
        results = trainer.train_all_models()
        
        # En iyi modeli kaydet
        # Model isim mapping'i (display name -> key name)
        model_name_mapping = {
            'Logistic Regression': 'logistic_regression',
            'Random Forest': 'random_forest',
            'Unknown': 'logistic_regression'  # Fallback
        }
        
        best_model_display_name = results['best_model']
        best_model_key = model_name_mapping.get(
            best_model_display_name,
            best_model_display_name.lower().replace(' ', '_')
        )
        
        trainer.save_model(model_name=best_model_key)
        
        # Özet
        total_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("Eğitim Özeti")
        logger.info("=" * 60)
        logger.info(f"Toplam süre: {total_time:.2f} saniye")
        logger.info(f"En iyi model: {results['best_model']}")
        logger.info(f"F1 Skoru: {results['best_score']:.4f}")
        
        print("\n" + "=" * 60)
        print("✓ MODEL EĞİTİMİ BAŞARIYLA TAMAMLANDI!")
        print("=" * 60)
        print(f"En iyi model: {results['best_model']}")
        print(f"F1 Skoru: {results['best_score']:.4f}")
        print(f"Toplam süre: {total_time:.2f} saniye")
        print(f"\nModel dosyaları:")
        print(f"  - models/model.pkl")
        print(f"  - models/vectorizer.pkl")
        print(f"  - models/metadata.json")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"✗ Eğitim sırasında hata oluştu: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()


