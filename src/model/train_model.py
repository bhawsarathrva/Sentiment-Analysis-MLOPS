import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import yaml
from src.logger import logging


def load_data(file_path: str) -> pd.DataFrame:
    """Load preprocessed features and labels from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except FileNotFoundError:
        logging.error('Data file not found: %s', file_path)
        raise
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train the Logistic Regression model."""
    try:
        # Logistic Regression with L1 regularization using liblinear solver
        clf = LogisticRegression(C=1, solver='liblinear', penalty='l1')
        clf.fit(X_train, y_train)
        logging.info('Model training completed')
        return clf
    except Exception as e:
        logging.error('Error during model training: %s', e)
        raise


def save_model(model, file_path: str) -> None:
    """Save the trained model as a serialized pickle file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info('Model saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model: %s', e)
        raise


def main():
    try:
        # Source train set paths
        train_data_path = './data/processed/train_bow.csv'
        model_output_path = 'models/model.pkl'

        train_data = load_data(train_data_path)
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        logging.info('Starting model training...')
        clf = train_model(X_train, y_train)
        
        save_model(clf, model_output_path)
        logging.info('Model training pipeline successfully finished.')
    except Exception as e:
        logging.error('Failed to complete the model training process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
