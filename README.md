# End-to-End MLOps Sentiment Analysis Pipeline

This repository contains a production-grade, end-to-end Machine Learning Operations (MLOps) project for Sentiment Analysis on Twitter/Movie Review data. The project showcases modern MLOps practices including Data Version Control (DVC), experiment tracking via MLflow & DagsHub, continuous integration/continuous deployment (CI/CD) using GitHub Actions, containerization with Docker, Kubernetes deployment via AWS EKS, and monitoring via Prometheus and Grafana.

---

## 🏗️ Project Architecture & Workflow

The system is designed as a modular pipeline where data, models, code, and infrastructure are fully versioned and monitored.

---

## 📁 Directory Structure

The project layout follows standard data science and MLOps structure:

```text
├── .github/
│   └── workflows_stop/       # CI/CD Workflow configuration (ci.yaml)
├── data/
│   ├── raw/                  # Downloaded raw dataset splits
│   ├── interim/              # Preprocessed data files
│   └── processed/            # Feature-engineered Bag-of-Words representations
├── docs/                     # Project documentation
├── flask_app/
│   ├── app.py                # Main Flask web server with Prometheus client
│   ├── templates/            # HTML pages for user UI
│   ├── requirements.txt      # Dependencies for the serving app
│   └── preprocessing_utility.py
├── models/                   # Saved model and vectorizer binary files
├── notebooks/                # Jupyter Notebooks for exploration and experiments
├── reports/                  # Evaluation reports (metrics.json, experiment_info.json)
├── scripts/                  # Utility scripts for CI (e.g. promoting models)
├── src/                      # Source code modules
│   ├── data/                 # Ingestion and cleaning scripts
│   ├── features/             # Bag of Words extraction script
│   ├── model/                # Model building, evaluation, and registration scripts
│   ├── logger/               # Custom logger implementation
│   └── connections/          # AWS S3 utility functions
├── tests/                    # Unit testing suite
├── Dockerfile                # Production Docker container blueprint
├── Makefile                  # Automation commands
├── dvc.yaml                  # DVC pipeline stage definitions
├── params.yaml               # Configurable pipeline parameters
└── setup.py                  # Project package installation setup
```

---

## ⚡ ML Pipeline & Data Version Control (DVC)

We use **DVC** to orchestrate the pipeline stages, version large data artifacts, and ensure reproducible runs.

### Pipeline Stages

1. **Data Ingestion** ([data_ingestion.py](file:///c:/Users/athrv/OneDrive/Desktop/MLOPS/PROJ_MLOPS/src/data/data_ingestion.py)): Downloads raw CSV data from the dataset URL, filters for binary sentiment labels (`positive` / `negative`), encodes labels as `0` and `1`, and splits data.
2. **Data Preprocessing** ([data_preprocessing.py](file:///c:/Users/athrv/OneDrive/Desktop/MLOPS/PROJ_MLOPS/src/data/data_preprocessing.py)): Cleans the review text by converting to lowercase, removing URLs, digits, punctuation, and English stopwords, and applying WordNet Lemmatization.
3. **Feature Engineering** ([feature_engineering.py](file:///c:/Users/athrv/OneDrive/Desktop/MLOPS/PROJ_MLOPS/src/features/feature_engineering.py)): Converts cleaned reviews into Bag-of-Words features using `CountVectorizer` based on `max_features` parameter in [params.yaml](file:///c:/Users/athrv/OneDrive/Desktop/MLOPS/PROJ_MLOPS/params.yaml). Saves `vectorizer.pkl`.
4. **Model Building** ([model_building.py](file:///c:/Users/athrv/OneDrive/Desktop/MLOPS/PROJ_MLOPS/src/model/model_building.py)): Fits a `LogisticRegression` model (L1 regularization) on BoW features and saves `model.pkl`.
5. **Model Evaluation** ([model_evaluation.py](file:///c:/Users/athrv/OneDrive/Desktop/MLOPS/PROJ_MLOPS/src/model/model_evaluation.py)): Computes metrics (Accuracy, Precision, Recall, and ROC-AUC) on the test set, saves metrics as a JSON report, and logs metrics/parameters/artifacts to **MLflow** hosted on **DagsHub**.
6. **Model Registration** ([register_model.py](file:///c:/Users/athrv/OneDrive/Desktop/MLOPS/PROJ_MLOPS/src/model/register_model.py)): Registers the model version within MLflow under the model name `my_model`.

### DVC Usage

To run the DVC pipeline:

```bash
dvc repro
```

To check pipeline status:

```bash
dvc status
```

---

## 🚀 Model Serving & Flask Web Application

A Flask application is provided inside the `flask_app/` folder to serve model predictions interactively.

### Key Features

- **Dynamic Loading**: Loads the latest version of the model from the **MLflow Production Stage** dynamically.
- **Preprocessing Integration**: Normalizes user input text using the same preprocessing steps as the training pipeline.
- **Monitoring Instrumentation**: Integrates `prometheus_client` to export live web app metrics. Exposes the following custom metrics under the `/metrics` endpoint:
  - `app_request_count`: Total requests received by method and endpoint.
  - `app_request_latency_seconds`: Histogram of request processing latency.
  - `model_prediction_count`: Counters for predicted positive and negative reviews.

To run the Flask application locally:

```bash
cd flask_app
pip install -r requirements.txt
python app.py
```

---

## 🐳 Containerization & CI/CD Pipeline

The web application is containerized with a multi-stage Docker build optimized for production.

### Docker commands

- **Build image**:
  ```bash
  docker build -t capstone-app:latest .
  ```
- **Run container locally**:
  ```bash
  docker run -p 8888:5000 -e CAPSTONE_TEST=<your-dagshub-token> capstone-app:latest
  ```

### GitHub Actions CI/CD ([ci.yaml](file:///c:/Users/athrv/OneDrive/Desktop/MLOPS/PROJ_MLOPS/.github/workflows_stop/ci.yaml))

On every code push, GitHub Actions automates:

1. Setting up Python, caching and installing dependencies.
2. Re-running the DVC pipeline (`dvc repro`) with Dagshub MLflow credentials.
3. Executing unit tests for the model and Flask web server.
4. Promoting the newly trained model to production using script rules.
5. Authenticating with AWS and building the Docker container image.
6. Pushing the tagged image to the **AWS Elastic Container Registry (ECR)**.
7. Deploying the application to an **AWS Elastic Kubernetes Service (EKS)** cluster.

---

## ☸️ Production Deployment (AWS EKS)

The app is orchestrated using Kubernetes for high availability and load balancing.

- **Deployment Specification** ([deployment.yaml](file:///c:/Users/athrv/OneDrive/Desktop/MLOPS/PROJ_MLOPS/deployment.yaml)):
  - Defines a deployment with `2` replicas.
  - Sets CPU/Memory requests & limits.
  - Loads the DagsHub token as a Kubernetes secret (`capstone-secret`).
  - Configures a `LoadBalancer` service to expose port 5000 externally.

### EKS Operations

- **Create Cluster**:
  ```bash
  eksctl create cluster --name flask-app-cluster --region us-east-1 --nodegroup-name flask-app-nodes --node-type t3.small --nodes 1 --nodes-min 1 --nodes-max 1 --managed
  ```
- **Configure local `kubectl`**:
  ```bash
  aws eks --region us-east-1 update-kubeconfig --name flask-app-cluster
  ```
- **Deploy to Cluster**:
  ```bash
  kubectl apply -f deployment.yaml
  ```
- **Teardown / Cleanup**:
  ```bash
  eksctl delete cluster --name flask-app-cluster --region us-east-1
  ```

---

## 📊 Live Monitoring (Prometheus & Grafana)

To ensure application health and track predictions, a dedicated monitoring stack is deployed.

1. **Prometheus Server**: Set up on a dedicated EC2 instance to scrape the `/metrics` endpoint of the LoadBalancer IP exposed by AWS EKS.
   - Configure target inside `/etc/prometheus/prometheus.yml`:
     ```yaml
     scrape_configs:
       - job_name: "flask-app"
         static_configs:
           - targets: ["<EKS-LOADBALANCER-EXTERNAL-IP>:5000"]
     ```
2. **Grafana Dashboard**: Set up on another EC2 instance, using the Prometheus instance as a datasource (`http://<prometheus-ip>:9090`). Custom dashboard visualizations are built to track request throughput, latency quantiles, and predictions over time.
