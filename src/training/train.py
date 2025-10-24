import argparse
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Corrected import paths based on the new structure
from src.data_prep import clean_text
from src.data_loader import load_dataset, train_val_split

def build_pipeline():
    """Builds the TF-IDF and Logistic Regression pipeline."""
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, max_features=50000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=-1))  # Use all available cores
    ])
    return pipe

def train_model(df, test_size=0.2, random_state=42):
    """
    Trains the sentiment analysis model and returns the trained model and evaluation metrics.

    Args:
        df (pd.DataFrame): The input DataFrame with 'review' and 'label' columns.
        test_size (float): The proportion of the dataset to use for validation.
        random_state (int): The random seed for reproducibility.

    Returns:
        tuple: A tuple containing:
            - The trained scikit-learn pipeline.
            - A dictionary of evaluation metrics (accuracy, precision, recall, f1).
    """
    X_train, X_val, y_train, y_val = train_val_split(
        df, test_size=test_size, random_state=random_state, stratify=True
    )

    pipeline = build_pipeline()
    print("Training the model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating the model...")
    preds = pipeline.predict(X_val)
    
    metrics = {
        "accuracy": accuracy_score(y_val, preds),
        "precision": precision_score(y_val, preds, average='macro'),
        "recall": recall_score(y_val, preds, average='macro'),
        "f1": f1_score(y_val, preds, average='macro'),
        "classification_report": classification_report(y_val, preds, digits=4),
        "confusion_matrix": confusion_matrix(y_val, preds)
    }

    return pipeline, metrics

def main(cli_args):
    """Main function to run the training script from the command line."""
    print(f"Loading dataset from {cli_args.csv_path}...")
    df = load_dataset(cli_args.csv_path)
    
    model, metrics = train_model(df)

    print(f"\nValidation Accuracy: {metrics['accuracy']:.4f}")
    print("Classification Report:\n", metrics['classification_report'])
    print("Confusion Matrix:\n", metrics['confusion_matrix'])

    joblib.dump(model, cli_args.model_path)
    print(f"\nSaved model to {cli_args.model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True, help="Path to CSV with columns: review, sentiment")
    parser.add_argument("--model_path", default="models/imdb_lr.joblib", help="Where to save the trained model")
    args = parser.parse_args()
    main(args)
