# 🎬 IMDB Film Sentiment Analizi Projesi

Film yorumları üzerinde yapay zeka destekli sentiment analizi yapan, **FastAPI** ile servis edilen ve **Docker** ile dağıtılabilen kapsamlı bir makine öğrenmesi projesi.

## 📋 Proje Özeti

Bu proje, 50,000 IMDB film yorumu üzerinde sentiment analizi (pozitif/negatif sınıflandırma) gerçekleştirir. TF-IDF vektörizasyonu ve makine öğrenmesi modelleri (Logistic Regression, Random Forest) kullanılarak %85+ doğruluk oranı hedeflenmiştir.

### 🎯 Özellikler

- ✅ 50,000 IMDB film yorumu sentiment analizi
- ✅ TF-IDF ile metin vektörizasyonu
- ✅ Multiple model karşılaştırma (Logistic Regression, Random Forest)
- ✅ FastAPI ile REST API servisi
- ✅ Docker containerization
- ✅ Kapsamlı test coverage
- ✅ Türkçe dokümantasyon
- ✅ Agent-friendly proje yapısı

## 🏗️ Proje Yapısı

```
film-sentiment-py/
├── data/                      # Veri dosyaları
│   └── IMDB Dataset.csv
├── src/                       # Kaynak kod
│   ├── data_loader.py         # Veri yükleme
│   ├── preprocessor.py        # Metin ön işleme
│   ├── train_model.py         # Model eğitimi
│   ├── evaluate_model.py      # Model değerlendirme
│   └── utils/
│       └── logger.py          # Loglama sistemi
├── api/                       # FastAPI servisi
│   └── main.py
├── models/                    # Eğitilmiş modeller
│   ├── model.pkl
│   ├── vectorizer.pkl
│   └── metadata.json
├── tests/                     # Testler
│   └── test_api.py
├── notebooks/                 # Jupyter notebooks
│   ├── 01_veri_analizi.ipynb
│   └── 02_model_karsilastirma.ipynb
├── reports/                   # Raporlar
│   ├── BRD.md
│   ├── SRS.md
│   └── model_rapor.md
├── docs/                      # Dokümantasyon
│   ├── ARCHITECTURE.md
│   ├── FEATURES.md
│   ├── TODO_TRACKING.md
│   ├── API_DOCUMENTATION.md
│   ├── DEVELOPMENT_GUIDE.md
│   └── CHANGELOG.md
├── config.yaml                # Konfigürasyon
├── requirements.txt           # Python bağımlılıklar
├── Dockerfile                 # Docker image tanımı
└── README.md                  # Bu dosya
```

## 🚀 Hızlı Başlangıç

### Gereksinimler

- Python 3.10+
- pip
- (Opsiyonel) Docker

### 1. Kurulum

```bash
# Projeyi klonla
git clone <repo-url>
cd film-sentiment-py

# Virtual environment oluştur
python -m venv venv

# Aktive et (Windows)
venv\Scripts\activate

# Aktive et (Linux/Mac)
source venv/bin/activate

# Bağımlılıkları yükle
pip install -r requirements.txt

# NLTK stopwords indir
python -c "import nltk; nltk.download('stopwords')"
```

### 2. Model Eğitimi

```bash
# Modeli eğit
python src/train_model.py
```

Bu komut:
- IMDB dataset'ini yükler
- Veriyi train/test olarak ayırır (%80/%20)
- TF-IDF vektörizasyonu yapar
- Logistic Regression ve Random Forest modellerini eğitir
- En iyi modeli `models/` klasörüne kaydeder

**Çıktı:**
```
models/
├── model.pkl           # Eğitilmiş model
├── vectorizer.pkl      # TF-IDF vectorizer
└── metadata.json       # Model metrikleri
```

### 3. API Servisi Başlatma

```bash
# FastAPI servisini başlat
uvicorn api.main:app --reload
```

API şu adreste çalışacak: `http://localhost:8000`

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### 4. API Kullanımı

#### Python ile

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "This movie was absolutely fantastic! Great acting and plot."}
)

print(response.json())
# {"sentiment": "positive", "confidence": 0.92, "prediction_time_ms": 23}
```

#### cURL ile

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Terrible movie, waste of time!"}'
```

#### Sağlık Kontrolü

```bash
curl http://localhost:8000/health
```

## 🐳 Docker ile Çalıştırma

### Image Oluşturma

```bash
docker build -t imdb-sentiment-api .
```

### Container Başlatma

```bash
docker run -d -p 8000:8000 --name sentiment-api imdb-sentiment-api
```

### Test

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Amazing movie!"}'
```

## 📊 Model Performansı

| Model                | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---------------------|----------|-----------|---------|----------|---------|
| Logistic Regression | ~0.88    | ~0.88     | ~0.88   | ~0.88    | ~0.95   |
| Random Forest       | ~0.85    | ~0.85     | ~0.85   | ~0.85    | ~0.92   |

*Not: Gerçek metrikler model eğitimi sonrası `models/metadata.json` dosyasında bulunur.*

## 🧪 Testler

```bash
# Tüm testleri çalıştır
pytest tests/ -v

# Sadece API testleri
pytest tests/test_api.py -v

# Coverage ile
pytest tests/ --cov=src --cov=api
```

## 📚 Dokümantasyon

> **⚠️ Dokümantasyon Güncellemeleri Hakkında**
> 
> Proje değişikliklerinde ilgili dokümanları **MUTLAKA** güncelleyin!
> - Yeni feature → `docs/FEATURES.md` + `docs/PROJECT_PLAN.md`
> - API değişikliği → `docs/API_DOCUMENTATION.md`
> - Mimari değişiklik → `docs/ARCHITECTURE.md` + `docs/PROJECT_PLAN.md`
> - Versiyon → `docs/CHANGELOG.md` + `docs/PROJECT_PLAN.md`
> - Task tamamlama → `docs/TODO_TRACKING.md`
> 
> Detaylı güncelleme kuralları: `docs/PROJECT_PLAN.md` Bölüm 17

Detaylı dokümantasyon için `docs/` klasörüne bakın:

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Sistem mimarisi
- **[FEATURES.md](docs/FEATURES.md)** - Feature açıklamaları
- **[API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md)** - API kullanım kılavuzu
- **[DEVELOPMENT_GUIDE.md](docs/DEVELOPMENT_GUIDE.md)** - Geliştirici rehberi
- **[TODO_TRACKING.md](docs/TODO_TRACKING.md)** - İlerleme takibi
- **[PROJECT_PLAN.md](docs/PROJECT_PLAN.md)** - Master planlama dokümanı
- **[CHANGELOG.md](docs/CHANGELOG.md)** - Versiyon geçmişi

### Raporlar

- **[BRD.md](reports/BRD.md)** - Business Requirements Document
- **[SRS.md](reports/SRS.md)** - Software Requirements Specification
- **[model_rapor.md](reports/model_rapor.md)** - Model performans raporu

## 🔧 Konfigürasyon

`config.yaml` dosyasında proje ayarlarını değiştirebilirsiniz:

```yaml
data:
  raw_path: "data/IMDB Dataset.csv"
  test_size: 0.2
  random_state: 42

preprocessing:
  max_features: 5000
  ngram_range: [1, 2]
  min_df: 5
  max_df: 0.8

models:
  logistic_regression:
    C: 1.0
    max_iter: 1000
  
  random_forest:
    n_estimators: 100
    max_depth: 50
```

## 📖 API Endpoints

### POST /predict
Film yorumu için sentiment tahmini yapar.

**Request:**
```json
{
  "text": "This movie was great!"
}
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.89,
  "prediction_time_ms": 15
}
```

### GET /health
Servis sağlık kontrolü.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0",
  "model_type": "LogisticRegression"
}
```

### GET /model/info
Model detayları.

**Response:**
```json
{
  "model_name": "logistic_regression",
  "model_type": "LogisticRegression",
  "version": "1.0.0",
  "training_date": "2025-11-05 14:30:00",
  "metrics": {
    "accuracy": 0.88,
    "f1_score": 0.88
  },
  "vocabulary_size": 5000
}
```

## 🌐 Deployment

### Render

1. GitHub'a push yapın
2. Render dashboard'da "New Web Service" seçin
3. Repository'yi bağlayın
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

### Heroku

```bash
# Procfile oluştur
echo "web: uvicorn api.main:app --host 0.0.0.0 --port \$PORT" > Procfile

# Deploy
heroku create imdb-sentiment-api
git push heroku main
```

## 🔮 Gelecek Geliştirmeler

- [ ] BERT/RoBERTa transformer modelleri
- [ ] Çok sınıflı sentiment (1-5 yıldız)
- [ ] Türkçe film yorumu desteği
- [ ] Batch prediction endpoint
- [ ] Redis caching
- [ ] Prometheus monitoring
- [ ] A/B testing altyapısı

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Commit yapın (`git commit -am 'feat: yeni özellik eklendi'`)
4. Branch'i push edin (`git push origin feature/yeni-ozellik`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje akademik amaçlı geliştirilmiştir.

## 👥 İletişim

**Proje Sahibi:** AI Yazılım Mühendisi: Emre Yılmaz

---

**⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın!**


