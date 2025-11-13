# Feature Dokümantasyonu
# IMDB Sentiment Analizi Projesi

**Versiyon:** 1.0.0  
**Son Güncelleme:** 5 Kasım 2025

---

> **⚠️ GÜNCELLEME HATIRLATMASI**
> 
> Bu dosya yeni feature eklendiğinde veya mevcut feature'lar değiştiğinde MUTLAKA güncellenmelidir!
> 
> **Yeni Feature Eklerken:**
> 1. ✅ Feature ID belirle (F007, F008, ...)
> 2. ✅ Template'i kullanarak dokümante et
> 3. ✅ Feature durumu tablosunu güncelle
> 4. ✅ İlgili bölümleri (bağımlılıklar, genişletme) ekle
> 
> **Diğer Güncellenecek Dosyalar:**
> - docs/PROJECT_PLAN.md (Bölüm 10.3: Feature listesi)
> - docs/TODO_TRACKING.md (yeni task'lar)
> - README.md (eğer önemli feature ise)

---

## Amaç

Bu doküman, projedeki tüm feature'ların detaylı açıklamalarını içerir. Her feature için:
- Tanım ve amaç
- İlgili dosyalar
- Input/Output
- Kullanım örnekleri
- Genişletme noktaları

---

## Feature İndeksi

- [F001: Veri Yükleme ve Hazırlama](#f001-veri-yükleme-ve-hazırlama)
- [F002: Metin Ön İşleme ve Vektörizasyon](#f002-metin-ön-işleme-ve-vektörizasyon)
- [F003: Model Eğitimi](#f003-model-eğitimi)
- [F004: Model Değerlendirme](#f004-model-değerlendirme)
- [F005: FastAPI Servisi](#f005-fastapi-servisi)
- [F006: Docker Deployment](#f006-docker-deployment)

---

## F001: Veri Yükleme ve Hazırlama

### Tanım
IMDB CSV dataset'ini yükler, validate eder ve train/test setlerine ayırır.

### İlgili Dosyalar
- `src/data_loader.py`

### Fonksiyonlar

#### `load_data(file_path: str) -> pd.DataFrame`

**Açıklama:** CSV dosyasından veriyi yükler.

**Input:**
- `file_path`: CSV dosya yolu (string)

**Output:**
- `pd.DataFrame`: Yüklenmiş veri

**Örnek:**
```python
from src.data_loader import load_data

df = load_data("data/IMDB Dataset.csv")
print(df.shape)  # (50000, 2)
print(df.columns)  # ['review', 'sentiment']
```

#### `validate_data(df: pd.DataFrame) -> bool`

**Açıklama:** DataFrame'in geçerliliğini kontrol eder.

**Validasyon Kontrolleri:**
- Boş DataFrame kontrolü
- Gerekli sütunların varlığı
- Null değer kontrolü
- Sentiment sınıf dağılımı

#### `split_data(df, test_size=0.2, random_state=42) -> Tuple`

**Açıklama:** Veriyi stratified split ile train/test ayırır.

**Input:**
- `df`: DataFrame
- `test_size`: Test oranı (0-1)
- `random_state`: Seed

**Output:**
- `(train_df, test_df)`: Tuple

**Örnek:**
```python
from src.data_loader import split_data

train_df, test_df = split_data(df, test_size=0.2, random_state=42)
# Train: 40000 örnekler, Test: 10000 örnekler
```

### Bağımlılıklar
- pandas
- scikit-learn (train_test_split)

### Genişletme Noktaları
1. **Farklı veri formatları:** JSON, Excel desteği
2. **Veri augmentation:** Backtranslation, synonym replacement
3. **Balanced sampling:** Dengesiz dataset'ler için
4. **Cross-validation split:** K-fold desteği

### Test Dosyası
`tests/test_data_loader.py` (oluşturulacak)

---

## F002: Metin Ön İşleme ve Vektörizasyon

### Tanım
Film yorumlarını temizler ve TF-IDF vektörlerine dönüştürür.

### İlgili Dosyalar
- `src/preprocessor.py`

### Sınıf: `TextPreprocessor`

#### Başlatma

```python
from src.preprocessor import TextPreprocessor

preprocessor = TextPreprocessor(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.8,
    stop_words='english'
)
```

#### Metodlar

##### `clean_text(text: str) -> str`

**Açıklama:** Tek bir metni temizler.

**İşlemler:**
1. HTML tag temizleme
2. Küçük harfe çevirme
3. Özel karakter temizleme
4. Fazla boşluk kaldırma

**Örnek:**
```python
raw_text = "<br />This movie was AMAZING!!! <b>Great</b> acting."
clean_text = preprocessor.clean_text(raw_text)
# Output: "this movie was amazing great acting"
```

##### `fit(texts: List[str]) -> TextPreprocessor`

**Açıklama:** Vocabulary oluşturur (sadece train data ile).

##### `transform(texts: List[str]) -> np.ndarray`

**Açıklama:** Metinleri TF-IDF vektörlerine dönüştürür.

##### `fit_transform(texts: List[str]) -> np.ndarray`

**Açıklama:** Fit ve transform işlemlerini birlikte yapar.

**Örnek:**
```python
# Train data için
X_train = preprocessor.fit_transform(train_texts)
# X_train shape: (40000, 5000)

# Test data için (sadece transform!)
X_test = preprocessor.transform(test_texts)
# X_test shape: (10000, 5000)
```

##### `save(filepath: str)` / `load(filepath: str)`

**Açıklama:** Preprocessor'ı kaydet/yükle (pickle).

**Örnek:**
```python
# Kaydet
preprocessor.save("models/vectorizer.pkl")

# Yükle
from src.preprocessor import TextPreprocessor
preprocessor = TextPreprocessor.load("models/vectorizer.pkl")
```

### Parametreler

| Parametre | Açıklama | Default | Önerilen Aralık |
|-----------|----------|---------|-----------------|
| `max_features` | Maksimum kelime sayısı | 5000 | 1000-10000 |
| `ngram_range` | N-gram aralığı | (1, 2) | (1, 1) veya (1, 2) |
| `min_df` | Minimum doküman frekansı | 5 | 2-10 |
| `max_df` | Maksimum doküman frekansı | 0.8 | 0.7-0.9 |
| `stop_words` | Stop words dili | 'english' | 'english', None |

### Bağımlılıklar
- scikit-learn (TfidfVectorizer)
- re (regex)
- pickle

### Genişletme Noktaları
1. **Lemmatization:** NLTK WordNetLemmatizer
2. **Stemming:** Porter Stemmer
3. **Custom stop words:** Domain-specific stop words
4. **Word embeddings:** Word2Vec, GloVe
5. **Contextual embeddings:** BERT tokenizer

### Test Dosyası
`tests/test_preprocessor.py` (oluşturulacak)

---

## F003: Model Eğitimi

### Tanım
Sentiment analizi için makine öğrenmesi modellerini eğitir ve karşılaştırır.

### İlgili Dosyalar
- `src/train_model.py`

### Sınıf: `SentimentModelTrainer`

#### Başlatma

```python
from src.train_model import SentimentModelTrainer

trainer = SentimentModelTrainer(config_path="config.yaml")
```

#### Ana İşlem Akışı

```python
# 1. Tüm modelleri eğit
results = trainer.train_all_models()

# 2. En iyi modeli kaydet
best_model = results['best_model'].lower().replace(' ', '_')
trainer.save_model(model_name=best_model)
```

#### Desteklenen Modeller

##### Logistic Regression

**Avantajlar:**
- Hızlı eğitim ve inference
- İyi yorumlanabilirlik
- Az parametre

**Hiperparametreler:**
```yaml
C: 1.0          # Regularization strength
max_iter: 1000  # Max iterations
solver: lbfgs   # Optimization algorithm
```

##### Random Forest

**Avantajlar:**
- Non-linear patterns
- Feature importance
- Robust to outliers

**Hiperparametreler:**
```yaml
n_estimators: 100   # Number of trees
max_depth: 50       # Max tree depth
min_samples_split: 2
random_state: 42
```

#### Model Karşılaştırma

**Kriter:** F1 Score (varsayılan)

Eğer F1 skorları %2 içindeyse, daha hızlı model seçilir.

### CLI Kullanımı

```bash
# Model eğitimi
python src/train_model.py

# Çıktılar:
# - models/model.pkl
# - models/vectorizer.pkl
# - models/metadata.json
# - logs/train_model_YYYYMMDD.log
```

### Metadata Formatı

```json
{
  "model_name": "logistic_regression",
  "model_type": "LogisticRegression",
  "version": "1.0.0",
  "training_date": "2025-11-05 14:30:00",
  "metrics": {
    "accuracy": 0.88,
    "precision": 0.88,
    "recall": 0.88,
    "f1_score": 0.88,
    "roc_auc": 0.95
  },
  "config": {
    "preprocessing": {...},
    "model_params": {...}
  },
  "vocabulary_size": 5000
}
```

### Bağımlılıklar
- scikit-learn (LogisticRegression, RandomForestClassifier)
- yaml (config loading)
- pickle (model saving)

### Genişletme Noktaları
1. **Yeni modeller:** SVM, XGBoost, LightGBM
2. **Hyperparameter tuning:** GridSearchCV, RandomizedSearchCV
3. **Ensemble methods:** Voting, Stacking
4. **Neural networks:** LSTM, BERT
5. **AutoML:** TPOT, Auto-sklearn

---

## F004: Model Değerlendirme

### Tanım
Eğitilmiş modellerin performansını çeşitli metriklerle değerlendirir.

### İlgili Dosyalar
- `src/evaluate_model.py`

### Sınıf: `ModelEvaluator`

#### Metrikler

##### Classification Metrics

| Metrik | Açıklama | İyi Değer |
|--------|----------|-----------|
| **Accuracy** | Doğru tahmin oranı | >0.85 |
| **Precision** | Pozitif dediğimizin doğruluğu | >0.85 |
| **Recall** | Pozitif olanları bulma oranı | >0.85 |
| **F1 Score** | Precision ve Recall harmonik ortalaması | >0.85 |
| **ROC-AUC** | Sınıflandırma eşik performansı | >0.90 |

##### Confusion Matrix

```
                 Tahmin
               Neg    Pos
Gerçek  Neg    TN     FP
        Pos    FN     TP
```

- **TN (True Negative):** Doğru negatif tahmin
- **FP (False Positive):** Yanlış pozitif tahmin (Type I Error)
- **FN (False Negative):** Yanlış negatif tahmin (Type II Error)
- **TP (True Positive):** Doğru pozitif tahmin

#### Kullanım

```python
from src.evaluate_model import ModelEvaluator, evaluate_model

# Kolay kullanım
metrics = evaluate_model(
    model=trained_model,
    X_test=X_test,
    y_test=y_test,
    model_name="Logistic Regression"
)

# Veya manuel
evaluator = ModelEvaluator(model_name="My Model")
metrics = evaluator.calculate_metrics(y_true, y_pred, y_pred_proba)
evaluator.save_metrics("metrics.json")
```

#### Model Karşılaştırma

```python
# Birden fazla modeli karşılaştır
evaluator = ModelEvaluator()
best_model, best_score = evaluator.compare_models(
    [lr_metrics, rf_metrics],
    metric_name='f1_score'
)
```

### Bağımlılıklar
- scikit-learn (metrics)
- numpy
- json

### Genişletme Noktaları
1. **Visualization:** ROC curve, PR curve plotting
2. **Error analysis:** Hatalı örnekleri analiz et
3. **Calibration:** Probability calibration
4. **Feature importance:** SHAP, LIME

---

## F005: FastAPI Servisi

### Tanım
Eğitilmiş modeli REST API olarak sunar.

### İlgili Dosyalar
- `api/main.py`

### Endpoints

#### POST /predict

**Açıklama:** Sentiment tahmini yapar.

**Request:**
```json
{
  "text": "This movie was absolutely fantastic!"
}
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.92,
  "prediction_time_ms": 23
}
```

**Error Responses:**
- `400`: Geçersiz input
- `422`: Validation error
- `500`: Internal server error
- `503`: Service unavailable

#### GET /health

**Açıklama:** Servis sağlık kontrolü.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0",
  "model_type": "LogisticRegression"
}
```

#### GET /model/info

**Açıklama:** Model detayları.

**Response:**
```json
{
  "model_name": "logistic_regression",
  "model_type": "LogisticRegression",
  "version": "1.0.0",
  "training_date": "2025-11-05 14:30:00",
  "metrics": {...},
  "vocabulary_size": 5000
}
```

#### GET /docs

**Açıklama:** Swagger UI otomatik dokümantasyon.

#### GET /redoc

**Açıklama:** ReDoc alternatif dokümantasyon.

### Pydantic Models

#### PredictionRequest

```python
class PredictionRequest(BaseModel):
    text: str = Field(min_length=10, max_length=5000)
```

#### PredictionResponse

```python
class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float  # 0.0-1.0
    prediction_time_ms: int
```

### Model Yönetimi

**Singleton Pattern:** Model bir kez yüklenir, tüm isteklerde paylaşılır.

```python
class ModelManager:
    _instance = None
    _model = None
    _preprocessor = None
    
    # Startup'ta yüklenir
    def load_model(self, path="models/model.pkl")
    def load_preprocessor(self, path="models/vectorizer.pkl")
    
    # Tahmin
    def predict(self, text: str) -> dict
```

### Başlatma

```bash
# Development
uvicorn api.main:app --reload

# Production
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Bağımlılıklar
- FastAPI
- Uvicorn
- Pydantic

### Genişletme Noktaları
1. **Batch prediction:** Toplu tahmin endpoint'i
2. **Authentication:** API key, OAuth2
3. **Rate limiting:** Redis-based rate limiter
4. **Caching:** Redis ile response caching
5. **Monitoring:** Prometheus metrics
6. **Async processing:** Celery ile queue system

---

## F006: Docker Deployment

### Tanım
Uygulamayı Docker container olarak paketler.

### İlgili Dosyalar
- `Dockerfile`
- `.dockerignore`

### Dockerfile Yapısı

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import nltk; nltk.download('stopwords')"
COPY . .
EXPOSE 8000
HEALTHCHECK --interval=30s CMD python -c "import requests..."
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kullanım

```bash
# Build
docker build -t imdb-sentiment-api .

# Run
docker run -d -p 8000:8000 --name sentiment-api imdb-sentiment-api

# Stop
docker stop sentiment-api

# Logs
docker logs sentiment-api

# Remove
docker rm sentiment-api
```

### Environment Variables

```bash
docker run -d \
  -p 8000:8000 \
  -e PORT=8000 \
  -e LOG_LEVEL=INFO \
  imdb-sentiment-api
```

### Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PORT=8000
      - LOG_LEVEL=INFO
    restart: unless-stopped
```

### Bağımlılıklar
- Docker
- Python base image

### Genişletme Noktaları
1. **Multi-stage build:** Image boyutunu küçült
2. **Volume mounting:** Model hotswap
3. **Docker Compose:** Multi-container setup
4. **Kubernetes:** Production orchestration
5. **CI/CD:** GitHub Actions ile otomatik build

---

## Feature Durumu

| Feature | Durum | Versiyon | Son Güncelleme |
|---------|-------|----------|----------------|
| F001 | ✅ Tamamlandı | 1.0.0 | 2025-11-05 |
| F002 | ✅ Tamamlandı | 1.0.0 | 2025-11-05 |
| F003 | ✅ Tamamlandı | 1.0.0 | 2025-11-05 |
| F004 | ✅ Tamamlandı | 1.0.0 | 2025-11-05 |
| F005 | ✅ Tamamlandı | 1.0.0 | 2025-11-05 |
| F006 | ✅ Tamamlandı | 1.0.0 | 2025-11-05 |

---

## Gelecek Feature'lar

### F007: Batch Prediction API (Planlanıyor)

**Tanım:** Birden fazla metin için toplu tahmin.

**Endpoint:** `POST /predict/batch`

**Request:**
```json
{
  "texts": ["Text 1", "Text 2", "Text 3"]
}
```

### F008: Model Retraining Pipeline (Planlanıyor)

**Tanım:** Yeni veri ile otomatik model güncelleme.

### F009: Multi-language Support (Planlanıyor)

**Tanım:** Türkçe, İspanyolca vb. dil desteği.

---

**Doküman Hazırlayan:** AI Yazılım Mühendisi  
**Son Güncelleme:** 5 Kasım 2025


