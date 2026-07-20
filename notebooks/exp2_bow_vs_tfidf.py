import setuptools
import os
import re
import string
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import xgboost
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, RocCurveDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import scipy.sparse

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# ========================== CONFIGURATION ==========================
CONFIG = {
    "data_path": "notebooks/data.csv",
    "test_size": 0.2,
    "mlflow_tracking_uri": "https://dagshub.com/bhawsarathrva/MLOPS_NLP.mlflow",
    "dagshub_repo_owner": "bhawsarathrva",
    "dagshub_repo_name": "MLOPS_NLP",
    "experiment_name": "Bow vs TfIdf"
}

# ========================== SETUP MLflow & DAGSHUB ==========================
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"], repo_name=CONFIG["dagshub_repo_name"], mlflow=True)
mlflow.set_experiment(CONFIG["experiment_name"])

# ========================== TEXT PREPROCESSING ==========================
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stop_words])

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    return text.lower()

def removing_punctuations(text):
    return re.sub(f"[{re.escape(string.punctuation)}]", ' ', text)

def removing_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def normalize_text(df):
    try:
        df['review'] = df['review'].apply(lower_case)
        df['review'] = df['review'].apply(remove_stop_words)
        df['review'] = df['review'].apply(removing_numbers)
        df['review'] = df['review'].apply(removing_punctuations)
        df['review'] = df['review'].apply(removing_urls)
        df['review'] = df['review'].apply(lemmatization)
        return df
    except Exception as e:
        print(f"Error during text normalization: {e}")
        raise

# ========================== LOAD & PREPROCESS DATA ==========================
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df = normalize_text(df)
        df = df[df['sentiment'].isin(['positive', 'negative'])]
        df['sentiment'] = df['sentiment'].replace({'negative': 0, 'positive': 1}).infer_objects(copy=False)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# ========================== FEATURE ENGINEERING ==========================
VECTORIZERS = {
    'BoW': CountVectorizer(),
    'TF-IDF': TfidfVectorizer()
}

ALGORITHMS = {
    'LogisticRegression': LogisticRegression(),
    'MultinomialNB': MultinomialNB(),
    'XGBoost': XGBClassifier(),
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier()
}

# ========================== TRAIN & EVALUATE MODELS ==========================
def train_and_evaluate(df):
    results = []
    with mlflow.start_run(run_name="All Experiments") as parent_run:
        for algo_name, algorithm in ALGORITHMS.items():
            for vec_name, vectorizer in VECTORIZERS.items():
                with mlflow.start_run(run_name=f"{algo_name} with {vec_name}", nested=True) as child_run:
                    try:
                        # Feature extraction
                        X = vectorizer.fit_transform(df['review'])
                        y = df['sentiment']
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG["test_size"], random_state=42)

                        # Log preprocessing parameters
                        mlflow.log_params({
                            "vectorizer": vec_name,
                            "algorithm": algo_name,
                            "test_size": CONFIG["test_size"]
                        })

                        # Train model
                        model = algorithm
                        model.fit(X_train, y_train)

                        # Log model parameters
                        log_model_params(algo_name, model)

                        # Evaluate model
                        y_pred = model.predict(X_test)
                        metrics = {
                            "accuracy": accuracy_score(y_test, y_pred),
                            "precision": precision_score(y_test, y_pred),
                            "recall": recall_score(y_test, y_pred),
                            "f1_score": f1_score(y_test, y_pred)
                        }
                        mlflow.log_metrics(metrics)

                        # Save results for comparison
                        results.append({
                            "algorithm": algo_name,
                            "vectorizer": vec_name,
                            "accuracy": metrics["accuracy"],
                            "precision": metrics["precision"],
                            "recall": metrics["recall"],
                            "f1_score": metrics["f1_score"]
                        })

                        # Generate and log Confusion Matrix
                        cm = confusion_matrix(y_test, y_pred)
                        fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
                        disp_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['negative', 'positive'])
                        disp_cm.plot(ax=ax_cm, cmap=plt.cm.Blues, values_format='d')
                        ax_cm.set_title(f"Confusion Matrix\n{algo_name} with {vec_name}")
                        mlflow.log_figure(fig_cm, "confusion_matrix.png")
                        plt.close(fig_cm)

                        # Generate and log ROC Curve
                        if hasattr(model, "predict_proba"):
                            try:
                                y_prob = model.predict_proba(X_test)[:, 1]
                                fpr, tpr, _ = roc_curve(y_test, y_prob)
                                roc_auc = auc(fpr, tpr)
                                fig_roc, ax_roc = plt.subplots(figsize=(6, 6))
                                disp_roc = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, name=f"{algo_name} ({vec_name})")
                                disp_roc.plot(ax=ax_roc)
                                ax_roc.set_title(f"ROC Curve\n{algo_name} with {vec_name}")
                                mlflow.log_figure(fig_roc, "roc_curve.png")
                                plt.close(fig_roc)
                            except Exception as roc_err:
                                print(f"Error generating ROC curve for {algo_name} with {vec_name}: {roc_err}")
                        elif hasattr(model, "decision_function"):
                            try:
                                y_score = model.decision_function(X_test)
                                fpr, tpr, _ = roc_curve(y_test, y_score)
                                roc_auc = auc(fpr, tpr)
                                fig_roc, ax_roc = plt.subplots(figsize=(6, 6))
                                disp_roc = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, name=f"{algo_name} ({vec_name})")
                                disp_roc.plot(ax=ax_roc)
                                ax_roc.set_title(f"ROC Curve\n{algo_name} with {vec_name}")
                                mlflow.log_figure(fig_roc, "roc_curve.png")
                                plt.close(fig_roc)
                            except Exception as roc_err:
                                print(f"Error generating ROC curve for {algo_name} with {vec_name}: {roc_err}")

                        # Log model
                        input_example = X_test[:5] if not scipy.sparse.issparse(X_test) else X_test[:5].toarray()
                        mlflow.sklearn.log_model(model, "model", input_example=input_example)

                        # Print results for verification
                        print(f"\nAlgorithm: {algo_name}, Vectorizer: {vec_name}")
                        print(f"Metrics: {metrics}")

                    except Exception as e:
                        print(f"Error in training {algo_name} with {vec_name}: {e}")
                        mlflow.log_param("error", str(e))

        # Generate comparative charts and log to the Parent Run
        if results:
            try:
                res_df = pd.DataFrame(results)
                res_csv_path = "model_comparison_results.csv"
                res_df.to_csv(res_csv_path, index=False)
                mlflow.log_artifact(res_csv_path)
                try:
                    os.remove(res_csv_path)
                except:
                    pass

                # Plot comparative bar chart of Accuracy
                fig_acc, ax_acc = plt.subplots(figsize=(12, 6))
                pivot_acc = res_df.pivot(index='algorithm', columns='vectorizer', values='accuracy')
                pivot_acc.plot(kind='bar', ax=ax_acc, width=0.6)
                ax_acc.set_ylabel('Accuracy')
                ax_acc.set_xlabel('Algorithm')
                ax_acc.set_title('Accuracy Comparison across Models and Vectorizers')
                ax_acc.set_ylim(0, 1.05)
                ax_acc.grid(axis='y', linestyle='--', alpha=0.7)
                ax_acc.legend(title='Vectorizer')
                plt.xticks(rotation=45)
                plt.tight_layout()
                mlflow.log_figure(fig_acc, "accuracy_comparison.png")
                plt.close(fig_acc)

                # Plot comparative bar chart of F1 Score
                fig_f1, ax_f1 = plt.subplots(figsize=(12, 6))
                pivot_f1 = res_df.pivot(index='algorithm', columns='vectorizer', values='f1_score')
                pivot_f1.plot(kind='bar', ax=ax_f1, width=0.6)
                ax_f1.set_ylabel('F1 Score')
                ax_f1.set_xlabel('Algorithm')
                ax_f1.set_title('F1 Score Comparison across Models and Vectorizers')
                ax_f1.set_ylim(0, 1.05)
                ax_f1.grid(axis='y', linestyle='--', alpha=0.7)
                ax_f1.legend(title='Vectorizer')
                plt.xticks(rotation=45)
                plt.tight_layout()
                mlflow.log_figure(fig_f1, "f1_score_comparison.png")
                plt.close(fig_f1)

            except Exception as plot_err:
                print(f"Error logging comparative plots to parent run: {plot_err}")

def log_model_params(algo_name, model):
    """Logs hyperparameters of the trained model to MLflow."""
    params_to_log = {}
    if algo_name == 'LogisticRegression':
        params_to_log["C"] = model.C
    elif algo_name == 'MultinomialNB':
        params_to_log["alpha"] = model.alpha
    elif algo_name == 'XGBoost':
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["learning_rate"] = model.learning_rate
    elif algo_name == 'RandomForest':
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["max_depth"] = model.max_depth
    elif algo_name == 'GradientBoosting':
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["learning_rate"] = model.learning_rate
        params_to_log["max_depth"] = model.max_depth

    mlflow.log_params(params_to_log)

# ========================== EXECUTION ==========================
if __name__ == "__main__":
    df = load_data(CONFIG["data_path"])
    train_and_evaluate(df)
