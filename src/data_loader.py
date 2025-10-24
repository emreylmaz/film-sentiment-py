import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_MAP = {"positive": 1, "negative": 0}

def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    # Beklenen kolon adları: review, sentiment
    # Gerekirse otomatik keşif ekleyin
    if "review" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("CSV must contain 'review' and 'sentiment' columns.")
    df = df.dropna(subset=["review", "sentiment"]).copy()
    df["label"] = df["sentiment"].map(LABEL_MAP)
    if df["label"].isna().any():
        # Farklı etiket isimleri varsa burada uyarlayın
        raise ValueError("Unexpected sentiment labels found. Expected 'positive'/'negative'.")
    return df

def train_val_split(df, test_size=0.2, random_state=42, stratify=True):
    y = df["label"]
    X = df["review"]
    if stratify:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    return X_train, X_val, y_train, y_val
