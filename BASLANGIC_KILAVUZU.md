# 🚀 IMDB Sentiment Analizi - Başlangıç Kılavuzu

**Hoş geldiniz!** Bu kılavuz size projeyi nasıl çalıştıracağınızı **adım adım** anlatır.

---

## ✅ Hazırlık Durumu

Şu ana kadar yapılanlar:
- ✅ Proje yapısı oluşturuldu (30+ dosya)
- ✅ Tüm kod yazıldı (8 modül)
- ✅ Dokümantasyon hazır (30+ sayfa)
- ✅ Virtual environment kuruldu
- ✅ Paketler yüklendi (pandas, scikit-learn, fastapi, vs.)
- ✅ NLTK data indirildi

**Şimdi yapmanız gerekenler:** Sadece 3 adım! 🎉

---

## 📋 Yapılacaklar Listesi

### 🔴 ADIM 1: Model Eğitimi (ZORUNLU!)

**Ne yapacak?**  
50,000 IMDB film yorumunu kullanarak sentiment analizi modeli eğitecek.

**Komut:**
```bash
python src/train_model.py
```

**Süre:** 10-20 dakika ⏱️

**Ne olacak?**
- Ekranda log mesajları göreceksiniz
- 2 model eğitilecek: Logistic Regression ve Random Forest
- En iyi model otomatik seçilecek
- Sonuçlar `models/` klasörüne kaydedilecek

**Çıktılar:**
```
models/
├── model.pkl          ← Eğitilmiş model
├── vectorizer.pkl     ← Metin işleyici
└── metadata.json      ← Performans metrikleri
```

**Örnek ekran çıktısı:**
```
============================================================
IMDB Sentiment Analizi - Model Eğitimi
============================================================
2025-11-05 18:00:00 - INFO - Veri yükleniyor...
2025-11-05 18:00:05 - INFO - ✓ Train: 40000 örnek
2025-11-05 18:00:05 - INFO - ✓ Test: 10000 örnek
2025-11-05 18:00:10 - INFO - Vectorizer eğitiliyor...
2025-11-05 18:05:00 - INFO - Logistic Regression eğitimi...
2025-11-05 18:08:00 - INFO - ✓ Accuracy: 0.88, F1: 0.88
...
============================================================
✓ MODEL EĞİTİMİ BAŞARIYLA TAMAMLANDI!
============================================================
```

**Sorun mu var?**
- Eğer "FileNotFoundError: data/IMDB Dataset.csv" hatası alırsanız:
  → Dataset'in `data/` klasöründe olduğundan emin olun

---

### 🟡 ADIM 2: API Servisini Başlatın

**Ne yapacak?**  
Eğittiğiniz modeli REST API olarak çalıştıracak.

**Komut:**
```bash
uvicorn api.main:app --reload
```

**Ne olacak?**
- API http://localhost:8000 adresinde çalışmaya başlayacak
- Ekranda şöyle bir çıktı göreceksiniz:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**API Erişimi:**
- **Ana Sayfa:** http://localhost:8000
- **Swagger Docs:** http://localhost:8000/docs ← 👈 Buradan test edebilirsiniz!
- **ReDoc:** http://localhost:8000/redoc

**Swagger UI'da Test:**
1. http://localhost:8000/docs adresine gidin
2. `/predict` endpoint'ini açın
3. "Try it out" butonuna tıklayın
4. Text alanına bir yorum yazın (örn: "This movie was great!")
5. "Execute" butonuna tıklayın
6. Sonucu görün! 🎉

**Manuel Test (Terminal'den):**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"This movie was absolutely fantastic!\"}"
```

**Beklenen yanıt:**
```json
{
  "sentiment": "positive",
  "confidence": 0.92,
  "prediction_time_ms": 23
}
```

**Durdurmak için:** Terminal'de `CTRL+C` tuşuna basın

---

### 🟢 ADIM 3: Test Edin (Opsiyonel ama önerilen)

**Ne yapacak?**  
API'nizin doğru çalıştığını test edecek.

**Komut:**
```bash
pytest tests/test_api.py -v
```

**Ne olacak?**
- 15+ test çalışacak
- Her testin sonucu gösterilecek (✓ PASSED veya ✗ FAILED)

**Örnek çıktı:**
```
tests/test_api.py::TestPredictionEndpoint::test_predict_positive_sentiment PASSED
tests/test_api.py::TestPredictionEndpoint::test_predict_negative_sentiment PASSED
tests/test_api.py::TestHealthEndpoint::test_health_check PASSED
...
==================== 15 passed in 2.5s ====================
```

---

## 🎉 Tebrikler! Artık Kullanabilirsiniz

### Python ile Kullanım

```python
import requests

# Tahmin yap
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Amazing film! Highly recommended."}
)

result = response.json()
print(f"Sentiment: {result['sentiment']}")
print(f"Güven: {result['confidence']:.2%}")
```

### Farklı Örnekler Deneyin

```python
# Pozitif yorum
test_reviews = [
    "This movie was absolutely fantastic!",
    "Great acting and wonderful story",
    "Best film I've seen this year!",
]

# Negatif yorum
test_reviews = [
    "Terrible movie, complete waste of time",
    "Very disappointing and boring",
    "I want my money back!",
]

for review in test_reviews:
    response = requests.post(
        "http://localhost:8000/predict",
        json={"text": review}
    )
    result = response.json()
    print(f"'{review[:30]}...' → {result['sentiment']}")
```

---

## 📊 Opsiyonel: Jupyter Notebooks

Veri analizi yapmak isterseniz:

```bash
# Jupyter Lab'ı yükleyin (eğer yoksa)
pip install jupyterlab

# Jupyter'ı başlatın
jupyter lab
```

**Notebook'lar:**
1. `notebooks/01_veri_analizi.ipynb` - Veri keşfi (EDA)
2. `notebooks/02_model_karsilastirma.ipynb` - Model analizi

---

## 🐳 Opsiyonel: Docker ile Çalıştırma

**Docker varsa:**

```bash
# 1. Image oluştur
docker build -t imdb-sentiment-api .

# 2. Container başlat
docker run -d -p 8000:8000 --name sentiment-api imdb-sentiment-api

# 3. Logları görüntüle
docker logs -f sentiment-api

# 4. Test et
curl http://localhost:8000/health

# 5. Durdur ve sil
docker stop sentiment-api
docker rm sentiment-api
```

---

## ❓ Sık Karşılaşılan Sorunlar

### 1. "Model dosyası bulunamadı" hatası

**Sorun:** API başladı ama `/predict` endpoint'i 503 hatası veriyor

**Çözüm:** Model henüz eğitilmemiş!
```bash
python src/train_model.py
```

### 2. "Port 8000 kullanımda" hatası

**Sorun:** Başka bir program 8000 portunu kullanıyor

**Çözüm:** Farklı port kullanın:
```bash
uvicorn api.main:app --reload --port 8001
```

### 3. "ModuleNotFoundError" hatası

**Sorun:** Virtual environment aktif değil

**Çözüm:** Virtual environment'ı aktive edin:
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 4. Eğitim çok yavaş

**Normal!** 50,000 veri ile eğitim 10-20 dakika sürebilir. ☕ Kahve molası verin!

---

## 📚 Daha Fazla Bilgi

### Detaylı Dokümantasyon

- **README.md** - Genel proje bilgisi
- **docs/API_DOCUMENTATION.md** - API detayları
- **docs/ARCHITECTURE.md** - Sistem mimarisi
- **docs/FEATURES.md** - Feature açıklamaları
- **docs/DEVELOPMENT_GUIDE.md** - Geliştirici rehberi
- **docs/PROJECT_PLAN.md** - Master plan

### Raporlar

- **reports/BRD.md** - İş gereksinimleri
- **reports/SRS.md** - Teknik spesifikasyon
- **reports/model_rapor.md** - Model performansı (eğitim sonrası dolacak)

---

## 🎯 Özet: Ne Yapmalıyım?

```
1. ✅ Hazırlık tamam (virtual env, paketler)
2. 🔴 Model eğit        → python src/train_model.py
3. 🟡 API başlat        → uvicorn api.main:app --reload
4. 🟢 Test et           → pytest tests/ -v
5. 🎉 Kullan!           → http://localhost:8000/docs
```

---

## 🆘 Yardım

**Sorun mu yaşıyorsunuz?**

1. Önce `docs/TODO_TRACKING.md` dosyasına bakın
2. Hata mesajını Google'da aratın
3. `docs/DEVELOPMENT_GUIDE.md` → Troubleshooting bölümü

---

## 🎊 Başarılar!

Projeniz hazır! Artık:
- ✅ Film yorumları için sentiment analizi yapabilirsiniz
- ✅ REST API olarak kullanabilirsiniz
- ✅ Kendi uygulamalarınıza entegre edebilirsiniz

**Keyifli kodlamalar! 🚀**

---

**Son Güncelleme:** 5 Kasım 2025  
**Versiyon:** 1.0.0  
**Proje:** IMDB Sentiment Analizi

