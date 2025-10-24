import click
import os
import pandas as pd
import joblib
from src.training.train import train_model
from src.data_loader import load_dataset

@click.group()
def cli():
    """Film Sentiment Analysis CLI tool."""
    pass

import json

@cli.command("train")
@click.option("--csv-path", required=True, type=click.Path(exists=True), help="Path to the training CSV file.")
@click.option("--model-path", default="models/sentiment_model.joblib", help="Path to save the trained model.")
@click.option("--report-path", default="reports/training_report.json", help="Path to save the evaluation report.")
@click.option("--test-size", default=0.2, help="Proportion of the dataset to include in the test split.")
@click.option("--random-state", default=42, help="Random seed for reproducibility.")
def train(csv_path, model_path, report_path, test_size, random_state):
    """Trains a model and saves the evaluation metrics to a report file."""
    click.echo(f"Starting training process...")
    click.echo(f"Loading data from: {csv_path}")
    df = load_dataset(csv_path)

    # Ensure directories exist
    for path in [model_path, report_path]:
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            click.echo(f"Created directory: {dir_name}")

    # Train the model
    model, metrics = train_model(df, test_size, random_state)

    # Save the model
    joblib.dump(model, model_path)
    click.echo(f"\nModel trained and saved to {model_path}")

    # --- Create and Save Report ---
    report = {
        "model_info": {
            "model_path": model_path,
            "training_data": csv_path,
        },
        "training_params": {
            "test_size": test_size,
            "random_state": random_state,
        },
        "evaluation_metrics": {
            "accuracy": metrics['accuracy'],
            "precision_macro": metrics['precision'],
            "recall_macro": metrics['recall'],
            "f1_macro": metrics['f1'],
            "confusion_matrix": metrics['confusion_matrix'].tolist(),
            "classification_report": metrics['classification_report']
        }
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)
    click.echo(f"Evaluation report saved to {report_path}")

    # --- Display Metrics to Console ---
    click.echo("\n--- Evaluation Metrics ---")
    click.echo(f"Accuracy: {metrics['accuracy']:.4f}")
    click.echo(f"Precision (Macro): {metrics['precision']:.4f}")
    click.echo(f"Recall (Macro): {metrics['recall']:.4f}")
    click.echo(f"F1-Score (Macro): {metrics['f1']:.4f}")
    click.echo("------------------------\n")

@cli.command("predict")
@click.option("--model-path", default="models/sentiment_model.joblib", type=click.Path(exists=True), help="Path to the trained model.")
@click.option("--input-file", required=True, type=click.Path(exists=True), help="Path to the input file with texts (one per line).")
@click.option("--output-file", required=True, help="Path to save the predictions.")
def predict(model_path, input_file, output_file):
    """Makes predictions on a file with texts."""
    click.echo(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    click.echo(f"Reading texts from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]

    if not texts:
        click.echo("Input file is empty or contains no valid text. Exiting.")
        return

    predictions = model.predict(texts)
    labels = ["positive" if pred == 1 else "negative" for pred in predictions]
    
    output_df = pd.DataFrame({
        'text': texts,
        'predicted_label': labels
    })
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_df.to_csv(output_file, index=False, encoding='utf-8')
    click.echo(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    cli()
