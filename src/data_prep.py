import re
import nltk
from nltk.corpus import stopwords

# İlk kullanımda gerekli paketleri indir
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")
    _ = stopwords.words("english")

STOPWORDS = set(_)

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<.*?>")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    x = text.lower()
    x = URL_RE.sub(" ", x)
    x = HTML_RE.sub(" ", x)
    x = NON_ALPHA_RE.sub(" ", x)
    x = re.sub(r"\s+", " ", x).strip()
    # Basit stopword temizliği (TF‑IDF tokenizer'ı kelime bölmeyi kendisi yapacak)
    tokens = [t for t in x.split() if t not in STOPWORDS]
    return " ".join(tokens)
