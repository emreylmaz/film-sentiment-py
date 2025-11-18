"""
Model Değerlendirme Modülü

Bu modül, eğitilmiş modellerin performansını değerlendirir ve
detaylı metrikler üretir.
"""

import json
from typing import Dict, Any, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from src.utils.logger import setup_logger

# Logger oluştur
logger = setup_logger(__name__)


class ModelEvaluator:
    """
    Model performans değerlendirme sınıfı.
    
    Çeşitli sınıflandırma metriklerini hesaplar ve raporlar.
    """
    
    def __init__(self, model_name: str = "Model"):
        """
        ModelEvaluator'ı başlatır.
        
        Args:
            model_name: Model ismi (raporlama için)
        """
        self.model_name = model_name
        self.metrics = {}
        
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Tüm performans metriklerini hesaplar.
        
        Args:
            y_true: Gerçek etiketler
            y_pred: Tahmin edilen etiketler
            y_pred_proba: Tahmin olasılıkları (ROC-AUC için)
            
        Returns:
            Metrikler içeren dictionary
        """
        logger.info(f"{self.model_name} için metrikler hesaplanıyor...")
        
        metrics = {}
        
        # Temel metrikler
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary', pos_label='positive')
        metrics['recall'] = recall_score(y_true, y_pred, average='binary', pos_label='positive')
        metrics['f1_score'] = f1_score(y_true, y_pred, average='binary', pos_label='positive')
        
        # ROC-AUC (eğer olasılıklar verilmişse)
        if y_pred_proba is not None:
            # Binary classification için ikinci sınıfın olasılığını al
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
                y_proba = y_pred_proba[:, 1]
            else:
                y_proba = y_pred_proba
            
            # Label encoding (positive -> 1, negative -> 0)
            y_true_binary = np.array([1 if label == 'positive' else 0 for label in y_true])
            metrics['roc_auc'] = roc_auc_score(y_true_binary, y_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=['negative', 'positive'])
        metrics['confusion_matrix'] = cm.tolist()
        
        # True/False Positives/Negatives
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Toplam örnek sayısı
        metrics['total_samples'] = len(y_true)
        
        self.metrics = metrics
        
        logger.info(f"✓ Metrikler hesaplandı:")
        logger.info(f"  Doğruluk (Accuracy): {metrics['accuracy']:.4f}")
        logger.info(f"  Kesinlik (Precision): {metrics['precision']:.4f}")
        logger.info(f"  Duyarlılık (Recall): {metrics['recall']:.4f}")
        logger.info(f"  F1 Skoru: {metrics['f1_score']:.4f}")
        if 'roc_auc' in metrics:
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """
        Detaylı sınıflandırma raporu üretir.
        
        Args:
            y_true: Gerçek etiketler
            y_pred: Tahmin edilen etiketler
            
        Returns:
            Formatlı sınıflandırma raporu
        """
        report = classification_report(
            y_true,
            y_pred,
            labels=['negative', 'positive'],
            target_names=['Negatif', 'Pozitif'],
            digits=4
        )
        
        return report
    
    def print_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> None:
        """
        Confusion matrix'i konsola yazdırır.
        
        Args:
            y_true: Gerçek etiketler
            y_pred: Tahmin edilen etiketler
        """
        cm = confusion_matrix(y_true, y_pred, labels=['negative', 'positive'])
        
        print("\n=== Confusion Matrix ===")
        print("                 Tahmin")
        print("               Neg    Pos")
        print(f"Gerçek  Neg   {cm[0,0]:5d}  {cm[0,1]:5d}")
        print(f"        Pos   {cm[1,0]:5d}  {cm[1,1]:5d}")
        print()
        
        # Yüzdeli gösterim
        total = cm.sum()
        print("Yüzdeli Dağılım:")
        print(f"  True Negatives:  {cm[0,0]/total*100:5.1f}%")
        print(f"  False Positives: {cm[0,1]/total*100:5.1f}%")
        print(f"  False Negatives: {cm[1,0]/total*100:5.1f}%")
        print(f"  True Positives:  {cm[1,1]/total*100:5.1f}%")
    
    def save_metrics(self, filepath: str) -> None:
        """
        Metrikleri JSON dosyasına kaydeder.
        
        Args:
            filepath: Kayıt yolu (örn: models/metrics.json)
        """
        if not self.metrics:
            logger.warning("Henüz hesaplanmış metrik yok!")
            return
        
        logger.info(f"Metrikler kaydediliyor: {filepath}")
        
        # JSON serializable hale getir
        metrics_to_save = {
            'model_name': self.model_name,
            **self.metrics
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics_to_save, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Metrikler kaydedildi: {filepath}")
    
    @staticmethod
    def load_metrics(filepath: str) -> Dict[str, Any]:
        """
        Metrikleri JSON dosyasından yükler.
        
        Args:
            filepath: Yükleme yolu
            
        Returns:
            Metrikler dictionary'si
        """
        logger.info(f"Metrikler yükleniyor: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        logger.info(f"✓ Metrikler yüklendi: {filepath}")
        return metrics
    
    def compare_models(
        self,
        metrics_list: list,
        metric_name: str = 'f1_score'
    ) -> Tuple[str, float]:
        """
        Birden fazla modeli karşılaştırır ve en iyisini seçer.
        
        Args:
            metrics_list: Model metrikleri listesi
            metric_name: Karşılaştırma için kullanılacak metrik
            
        Returns:
            (en_iyi_model_adi, en_iyi_skor) tuple'ı
        """
        logger.info(f"Modeller karşılaştırılıyor (metrik: {metric_name})...")
        
        best_model = None
        best_score = -1
        
        for metrics in metrics_list:
            model_name = metrics.get('model_name', 'Unknown')
            score = metrics.get(metric_name, 0)
            
            logger.info(f"  {model_name}: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        logger.info(f"✓ En iyi model: {best_model} ({best_score:.4f})")
        
        return best_model, best_score


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Modeli değerlendirir ve metrikleri döndürür.
    
    Args:
        model: Değerlendirilecek model
        X_test: Test özellikleri
        y_test: Test etiketleri
        model_name: Model ismi
        
    Returns:
        Metrikler dictionary'si
    """
    evaluator = ModelEvaluator(model_name=model_name)
    
    # Tahminler
    y_pred = model.predict(X_test)
    
    # Olasılıklar (eğer destekleniyorsa)
    y_pred_proba = None
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)
    elif hasattr(model, 'decision_function'):
        # Decision function'ı olasılığa çevir (sigmoid)
        decision = model.decision_function(X_test)
        y_pred_proba = 1 / (1 + np.exp(-decision))
    
    # Metrikleri hesapla
    metrics = evaluator.calculate_metrics(y_test, y_pred, y_pred_proba)
    
    # Model adını metrics'e ekle
    metrics['model_name'] = model_name
    
    # Classification report
    print(f"\n=== {model_name} - Sınıflandırma Raporu ===")
    print(evaluator.get_classification_report(y_test, y_pred))
    
    # Confusion matrix
    evaluator.print_confusion_matrix(y_test, y_pred)
    
    return metrics


# Test ve örnek kullanım
if __name__ == "__main__":
    # Örnek test verileri
    np.random.seed(42)
    
    # 100 örnek
    y_true = np.random.choice(['positive', 'negative'], size=100)
    
    # %85 doğrulukla tahmin simülasyonu
    y_pred = y_true.copy()
    random_indices = np.random.choice(100, size=15, replace=False)
    for idx in random_indices:
        y_pred[idx] = 'negative' if y_pred[idx] == 'positive' else 'positive'
    
    # Olasılıklar (simüle)
    y_pred_proba = np.random.rand(100, 2)
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)
    
    print("=== Model Değerlendirme Testi ===\n")
    
    # Evaluator oluştur
    evaluator = ModelEvaluator(model_name="Test Model")
    
    # Metrikleri hesapla
    metrics = evaluator.calculate_metrics(y_true, y_pred, y_pred_proba)
    
    # Classification report
    print("\n" + evaluator.get_classification_report(y_true, y_pred))
    
    # Confusion matrix
    evaluator.print_confusion_matrix(y_true, y_pred)
    
    # Metrikleri kaydet
    evaluator.save_metrics("test_metrics.json")
    
    # Metrikleri yükle
    loaded_metrics = ModelEvaluator.load_metrics("test_metrics.json")
    print(f"\n✓ Yüklenen metrikler: {loaded_metrics}")
    
    print("\n✓ ModelEvaluator başarıyla test edildi!")
    
    # Test dosyasını temizle
    import os
    if os.path.exists("test_metrics.json"):
        os.remove("test_metrics.json")


