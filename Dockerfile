# IMDB Sentiment Analizi Docker Image
# Version: 2.0.0 (MongoDB + Redis + JWT Blacklist)

# Python 3.10 slim base image kullan
FROM python:3.10-slim

# Metadata
LABEL maintainer="AI Yazılım Mühendisi: Emre Yılmaz"
LABEL description="IMDB Film Sentiment Analizi API - JWT Auth + Redis Blacklist"
LABEL version="2.0.0"

# Çalışma dizini
WORKDIR /app

# Sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıklarını kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# NLTK data indir
RUN python -c "import nltk; nltk.download('stopwords', quiet=True)"

# Proje dosyalarını kopyala
COPY . .

# .env dosyası varsa kopyala (opsiyonel, docker-compose override eder)
# COPY .env .env

# Environment variables (docker-compose'da override edilebilir)
ENV MONGO_URL=mongodb://mongodb:27017
ENV REDIS_URL=redis://redis:6379
ENV SECRET_KEY=your-secret-key-change-in-production
ENV CORS_ORIGINS=http://localhost:3000,http://localhost:3001

# Port aç
EXPOSE 8000

# Health check (curl kullan, daha güvenilir)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Uvicorn ile servisi başlat
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]


