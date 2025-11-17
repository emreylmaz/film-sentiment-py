# IMDB Sentiment Analizi Docker Image

# Python 3.10 slim base image kullan
FROM python:3.10-slim

# Metadata
LABEL maintainer="Emre Yılmaz"
LABEL description="IMDB Film Sentiment Analizi API"
LABEL version="1.0.0"

# Çalışma dizini
WORKDIR /app

# Sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıklarını kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# NLTK data indir
RUN python -c "import nltk; nltk.download('stopwords', quiet=True)"

# Proje dosyalarını kopyala
COPY . .

# Port aç
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Uvicorn ile servisi başlat
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]


