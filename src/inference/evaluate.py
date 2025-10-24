import argparse
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from .preprocess import clean_text
from .data import load_dataset, train_val_split

def main(args):
    df = load_dataset(args.csv_path)
    _, X_val, _, y_val = train_val_split(df, test_size=0.2, random_state=42, stratify=True)

    model = joblib.load(args.model_path)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"Validation Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_val, preds, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_val, preds))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--model_path", default="models/imdb_lr.joblib")
    args = parser.parse_args()
    main(args)
