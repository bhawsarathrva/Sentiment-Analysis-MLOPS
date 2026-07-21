# predict model
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import argparse
import pickle
import numpy as np
import pandas as pd
from src.logger import logging
from src.data.data_preprocessing import preprocess_dataframe


def load_vectorizer(vectorizer_path: str):
    """Load the trained CountVectorizer object."""
    try:
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logging.info("Vectorizer loaded successfully from %s", vectorizer_path)
        return vectorizer
    except Exception as e:
        logging.error("Failed to load vectorizer from %s: %s", vectorizer_path, e)
        raise


def load_model(model_path: str):
    """Load the trained machine learning model."""
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logging.info("Model loaded successfully from %s", model_path)
        return model
    except Exception as e:
        logging.error("Failed to load model from %s: %s", model_path, e)
        raise


def predict_text(text: str, model, vectorizer) -> dict:
    """Preprocess a single text string and return the predicted sentiment."""
    try:
        # Create a single-row DataFrame for preprocessing consistency
        temp_df = pd.DataFrame({'text': [text]})
        processed_df = preprocess_dataframe(temp_df, col='text')

        # If preprocessing emptied the text (e.g. empty string or stopwords only)
        cleaned_text = processed_df['text'].iloc[0] if not processed_df.empty else ""

        # Transform using vectorizer
        features = vectorizer.transform([cleaned_text])
        features_matrix = features.toarray()

        # Predict class and probabilities if supported
        prediction = int(model.predict(features_matrix)[0])
        label = "positive" if prediction == 1 else "negative"

        probability = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features_matrix)[0]
            probability = float(probs[prediction])

        result = {
            'input_text': text,
            'cleaned_text': cleaned_text,
            'prediction': prediction,
            'label': label,
            'confidence': probability
        }
        logging.info("Prediction completed for text input.")
        return result
    except Exception as e:
        logging.error("Error occurred during text prediction: %s", e)
        raise


def predict_file(file_path: str, text_col: str, model, vectorizer, output_path: str = None) -> pd.DataFrame:
    """Run predictions on a CSV file containing a column of texts."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at {file_path}")

        df = pd.read_csv(file_path)
        if text_col not in df.columns:
            raise KeyError(f"Column '{text_col}' not found in dataset columns: {list(df.columns)}")

        # Make a copy for processing
        df_processed = df.copy()
        df_processed = preprocess_dataframe(df_processed, col=text_col)

        features = vectorizer.transform(df_processed[text_col])
        predictions = model.predict(features.toarray())
        df_processed['predicted_label'] = predictions
        df_processed['sentiment'] = df_processed['predicted_label'].map({1: 'positive', 0: 'negative'})

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df_processed.to_csv(output_path, index=False)
            logging.info("Prediction results saved to %s", output_path)

        return df_processed
    except Exception as e:
        logging.error("Error occurred during batch file prediction: %s", e)
        raise


def main():
    parser = argparse.ArgumentParser(description="Sentiment Analysis Prediction Script")
    parser.add_argument("--text", type=str, help="Input text string for sentiment prediction")
    parser.add_argument("--file", type=str, help="Path to input CSV file for batch prediction")
    parser.add_argument("--col", type=str, default="text", help="Column name containing text in CSV file")
    parser.add_argument("--output", type=str, help="Output CSV path for saving batch prediction results")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "../../models/model.pkl"),
        help="Path to trained model pickle file"
    )
    parser.add_argument(
        "--vectorizer_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "../../models/vectorizer.pkl"),
        help="Path to trained vectorizer pickle file"
    )

    args = parser.parse_args()

    try:
        model_path = os.path.abspath(args.model_path)
        vectorizer_path = os.path.abspath(args.vectorizer_path)

        model = load_model(model_path)
        vectorizer = load_vectorizer(vectorizer_path)

        if args.file:
            print(f"Running batch prediction on file: {args.file}")
            results_df = predict_file(args.file, args.col, model, vectorizer, args.output)
            print(results_df.head())
        elif args.text:
            result = predict_text(args.text, model, vectorizer)
            print("\n--- Prediction Result ---")
            print(f"Input Text: {result['input_text']}")
            print(f"Cleaned Text: {result['cleaned_text']}")
            print(f"Prediction: {result['prediction']} ({result['label'].upper()})")
            if result['confidence'] is not None:
                print(f"Confidence: {result['confidence']:.4f}")
        else:
            # Default test case when no CLI args are passed
            sample_text = "This product is amazing and works perfectly!"
            print(f"No arguments provided. Running test prediction on sample text: '{sample_text}'")
            result = predict_text(sample_text, model, vectorizer)
            print("\n--- Sample Prediction Result ---")
            print(f"Input Text: {result['input_text']}")
            print(f"Cleaned Text: {result['cleaned_text']}")
            print(f"Prediction: {result['prediction']} ({result['label'].upper()})")
            if result['confidence'] is not None:
                print(f"Confidence: {result['confidence']:.4f}")

    except Exception as e:
        logging.error("Failed to complete prediction: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
