# Business Requirements Document (BRD)
# IMDB Film Sentiment Analizi Projesi

**Versiyon:** 1.0  
**Tarih:** 5 Kasım 2025  
**Proje Sahibi:** AI Yazılım Mühendisi: Emre Yılmaz  
**Durum:** Onaylandı

---

## 1. Executive Summary

Bu doküman, IMDB film yorumları üzerinde otomatik sentiment analizi yapan bir yapay zeka sisteminin iş gereksinimlerini tanımlar. Sistem, kullanıcı yorumlarını analiz ederek pozitif veya negatif sentiment tahmini yapar ve bu hizmeti REST API üzerinden sunar.

### 1.1 Proje Amacı

Film endüstrisi ve dijital platformlar için kullanıcı geri bildirimlerinin otomatik analiz edilmesini sağlayarak:
- Manuel analiz maliyetlerini azaltmak
- Gerçek zamanlı sentiment izleme imkanı sunmak
- Veri odaklı karar verme süreçlerini desteklemek

---

## 2. İş Problemi

### 2.1 Mevcut Durum

Film prodüksiyon şirketleri, dağıtım platformları ve pazarlama ekipleri:
- **Manuel Analiz:** Binlerce kullanıcı yorumunu manuel olarak analiz etmek zorunda
- **Yavaş Geri Bildirim:** Filmlerin piyasaya sürülmesinden sonra sentiment analizi haftalar alıyor
- **Yüksek Maliyet:** Analist ekipleri ve zaman kaynakları yoğun kullanılıyor
- **Sınırlı Kapsam:** Sadece örneklem üzerinde çalışılabiliyor, tüm yorumlar analiz edilemiyor

### 2.2 Fırsatlar

- **Otomasyon:** Yapay zeka ile %90+ doğrulukla otomatik sentiment analizi
- **Gerçek Zamanlı:** API entegrasyonu ile anlık analiz
- **Ölçeklenebilirlik:** Günde binlerce yorum analiz edilebilir
- **Maliyet Tasarrufu:** Manuel iş gücü maliyetinin %70 azaltılması

---

## 3. İş Hedefleri

### 3.1 Birincil Hedefler

1. **Yüksek Doğruluk:** %85+ doğruluk oranı ile sentiment tahmini
2. **Hızlı Response:** <100ms tahmin süresi
3. **Ölçeklenebilirlik:** Günde 10,000+ istek kapasitesi
4. **Kolay Entegrasyon:** REST API ile platform bağımsız kullanım

### 3.2 İkincil Hedefler

1. Model açıklanabilirliği (interpretability)
2. Sürekli öğrenme ve model güncelleme altyapısı
3. Multi-dil desteği (ilk aşamada İngilizce, sonra Türkçe)
4. Batch prediction desteği

---

## 4. Paydaşlar

### 4.1 İçsel Paydaşlar

| Paydaş | Rol | Sorumluluk |
|--------|-----|------------|
| AI Yazılım Mühendisi: Emre Yılmaz | Proje Sahibi | Geliştirme, deployment, bakım |
| Veri Bilimci | Konsültan | Model optimizasyonu, feature engineering |

### 4.2 Dışsal Paydaşlar

| Paydaş | İhtiyaç | Beklenti |
|--------|---------|----------|
| Film Prodüksiyon Şirketleri | Piyasa geri bildirimi | Gerçek zamanlı sentiment raporu |
| Pazarlama Ekipleri | Kampanya etkinliği | Reklam sonrası sentiment değişimi |
| İçerik Platformları | Kullanıcı memnuniyeti | Film önerilerinde kullanım |
| Veri Analistleri | Trend analizi | Historical data ve API erişimi |

---

## 5. Kapsam

### 5.1 Kapsam İçinde

✅ **Fonksiyonellik**
- Film yorumları için binary sentiment analizi (pozitif/negatif)
- REST API servisi (FastAPI)
- Docker containerization
- Model versiyonlama ve metadata yönetimi
- Basic authentication ve rate limiting

✅ **Veri**
- 50,000 IMDB film yorumu
- İngilizce metin

✅ **Teknik**
- TF-IDF vektörizasyon
- Logistic Regression ve Random Forest modelleri
- Automated testing
- Comprehensive documentation

### 5.2 Kapsam Dışında

❌ **Gelecek Fazlar**
- Çok sınıflı sentiment (1-5 yıldız)
- Aspect-based sentiment (oyunculuk, senaryo, efektler ayrı)
- Çoklu dil desteği
- Gerçek zamanlı streaming analizi
- Grafik dashboard

---

## 6. Başarı Kriterleri

### 6.1 Teknik Metrikler

| Metrik | Hedef | Kritik Eşik |
|--------|-------|-------------|
| Accuracy | ≥%88 | %85 |
| F1 Score | ≥%88 | %85 |
| ROC-AUC | ≥%93 | %90 |
| Response Time | <50ms | <100ms |
| API Uptime | %99.5 | %99.0 |

### 6.2 İş Metrikleri

| Metrik | Hedef | Ölçüm Yöntemi |
|--------|-------|--------------|
| Maliyet Azaltma | %70 | Manuel analiz saati vs. API maliyeti |
| Analiz Hızı | 100x | Manuel vs. otomatik analiz süresi |
| Kapsam Artışı | %100 | Örneklem yerine tüm yorumlar |
| Kullanıcı Memnuniyeti | 4.5/5 | Paydaş anketleri |

---

## 7. Fonksiyonel Gereksinimler

### 7.1 Temel Fonksiyonlar

**FR-001: Sentiment Tahmini**
- Sistem, kullanıcı tarafından gönderilen metin için sentiment tahmini yapabilmeli
- Girdi: Film yorumu metni (10-5000 karakter)
- Çıktı: Sentiment (positive/negative) + Güven skoru (0-1)

**FR-002: API Erişimi**
- RESTful API üzerinden erişilebilir olmalı
- JSON formatında request/response
- Swagger/OpenAPI dokümantasyonu otomatik

**FR-003: Model Bilgisi**
- Model versiyonu, tipi ve metrikleri sorgulanabilir olmalı
- Metadata endpoint üzerinden erişilebilir

**FR-004: Sağlık Kontrolü**
- Health check endpoint
- Model yüklü mü kontrolü
- Servis durumu (healthy/unhealthy)

### 7.2 Kalite Gereksinimleri

**QR-001: Doğruluk**
- Minimum %85 accuracy
- Dengeli precision ve recall

**QR-002: Performans**
- <100ms tahmin süresi
- Concurrent request desteği

**QR-003: Güvenilirlik**
- %99+ uptime
- Graceful error handling
- Automatic recovery

---

## 8. Teknik Gereksinimler

### 8.1 Altyapı

- **Sunucu:** Cloud-based (Render, Heroku, AWS)
- **Container:** Docker
- **Database:** Model storage için dosya sistemi yeterli
- **Monitoring:** Logging (future: Prometheus)

### 8.2 Güvenlik

- Input validation (XSS, injection attacks)
- Rate limiting (100 req/min)
- HTTPS zorunlu (production)
- API key authentication (future phase)

---

## 9. Riskler ve Kısıtlar

### 9.1 Riskler

| Risk | Olasılık | Etki | Azaltma Stratejisi |
|------|----------|------|-------------------|
| Model accuracy hedefine ulaşamama | Orta | Yüksek | Hyperparameter tuning, ensemble models |
| Yüksek inference latency | Düşük | Orta | Model optimization, caching |
| Dataset bias | Orta | Yüksek | Balanced training, bias testing |
| API aşırı yüklenmesi | Orta | Orta | Rate limiting, load balancing |

### 9.2 Kısıtlar

- **Veri:** Sadece İngilizce IMDB yorumları
- **Model:** Binary classification (çok sınıflı değil)
- **Kaynak:** Single instance deployment (ilk aşamada)
- **Bütçe:** Open-source teknolojiler kullanımı

---

## 10. ROI Analizi

### 10.1 Maliyet

**Geliştirme:**
- Proje süresi: 5 iş günü
- Geliştirici maliyet: ~5,000 TL

**İşletme (Aylık):**
- Cloud hosting: ~$10-20
- Bakım: 2 saat/ay
- Toplam: ~$50/ay

### 10.2 Fayda

**Maliyet Tasarrufu:**
- Manuel analiz: 1 analist x 40 saat/ay = 8,000 TL/ay
- Otomatik analiz: ~$50/ay = 1,500 TL/ay
- **Net Tasarruf:** 6,500 TL/ay = 78,000 TL/yıl

**Verimlilik Artışı:**
- Manuel: 100 yorum/saat
- Otomatik: 10,000+ yorum/saat
- **100x hız artışı**

**ROI:**
- İlk yıl ROI: (78,000 - 5,000) / 5,000 = **1,460%**
- Break-even: <1 ay

---

## 11. Proje Zaman Planı

| Faz | Süre | Teslimler |
|-----|------|-----------|
| **Faz 1:** Gereksinim Analizi | 0.5 gün | BRD, SRS |
| **Faz 2:** Veri Hazırlama | 1 gün | Data pipeline, EDA |
| **Faz 3:** Model Geliştirme | 2 gün | Trained models, metrics |
| **Faz 4:** API Geliştirme | 1 gün | REST API, tests |
| **Faz 5:** Deployment | 0.5 gün | Docker, cloud deployment |
| **Toplam** | **5 gün** | Çalışan production sistem |

---

## 12. Onaylar

| Rol | İsim | İmza | Tarih |
|-----|------|------|-------|
| Proje Sahibi | AI Yazılım Mühendisi: Emre Yılmaz | ✓ | 5 Kasım 2025 |
| Teknik Lead | AI Yazılım Mühendisi: Emre Yılmaz | ✓ | 5 Kasım 2025 |

---

## 13. Revizyon Geçmişi

| Versiyon | Tarih | Değişiklikler | Yazar |
|----------|-------|---------------|-------|
| 1.0 | 5 Kasım 2025 | İlk versiyon | AI Yazılım Mühendisi: Emre Yılmaz |

---

**Doküman Sonu**


