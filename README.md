# 🐱🐶 MLOps Assignment 2 --- End-to-End Cats vs Dogs Classification

> **Course:** MLOps --- S1-25_AIMLCZG523\
> **Assignment:** 2 \| **Total Marks:** 50\
> **Objective:** Build an end-to-end MLOps pipeline for model
> development, artifact creation, packaging, containerization, CI/CD
> deployment, and monitoring using open-source tools.

## 1. Project Overview

This project implements an end-to-end MLOps workflow for a **binary Cats
vs Dogs image-classification service** for a pet-adoption platform.

The implementation covers the complete lifecycle:

``` text
Dataset
   ↓
Data Validation & Pre-processing
   ↓
Model Training
   ↓
Experiment Tracking
   ↓
Model Evaluation & Artifacts
   ↓
FastAPI Inference Service
   ↓
Docker Image
   ↓
Automated Tests
   ↓
GitHub Actions CI
   ↓
Container Registry
   ↓
Continuous Deployment
   ↓
Smoke Tests
   ↓
Monitoring & Logs
   ↓
Post-Deployment Model Evaluation
```

The assignment defines five modules, each worth 10 marks:

  -----------------------------------------------------------------------
  Module                  Objective               Evidence
  ----------------------- ----------------------- -----------------------
  **M1**                  Model Development &     Dataset validation,
                          Experiment Tracking     preprocessing, CNN,
                                                  training, MLflow, DVC,
                                                  model/plot artifacts

  **M2**                  Model Packaging &       FastAPI `/health`,
                          Containerization        `/predict`,
                                                  dependencies, Docker

  **M3**                  CI --- Build, Test &    pytest, GitHub Actions,
                          Image Creation          Docker build, registry
                                                  publishing

  **M4**                  CD & Deployment         Docker Compose
                                                  deployment, automated
                                                  deployment,
                                                  health/prediction smoke
                                                  tests

  **M5**                  Monitoring, Logs &      request/latency
                          Final Submission        metrics, logs,
                                                  post-deployment
                                                  evaluation and reports
  -----------------------------------------------------------------------

The assignment requires a 224×224 RGB preprocessing pipeline,
train/validation/test splitting, and training augmentation.
fileciteturn0file0L5-L11

------------------------------------------------------------------------

# 2. Technology Stack

  Area                       Technology
  -------------------------- --------------------------------------------
  Language                   Python 3.11
  ML                         PyTorch / Torchvision
  Data / evaluation          Pandas / scikit-learn
  Experiment tracking        MLflow
  Data/pipeline versioning   DVC
  API                        FastAPI + Uvicorn
  Image handling             Pillow
  Validation                 Pydantic
  Testing                    pytest
  Containerization           Docker
  Deployment                 Docker Compose
  CI/CD                      GitHub Actions
  Registry                   Docker Hub / configured container registry
  Logging                    Python logging
  Metrics                    In-application request/latency metrics

The selected tools satisfy the assignment's open-source MLOps
requirements and its permitted CI/CD, registry, and deployment choices.
fileciteturn0file0L26-L38 fileciteturn0file0L46-L66

------------------------------------------------------------------------

# 3. Architecture

## 3.1 End-to-End MLOps Architecture

``` text
                         ┌──────────────────────┐
                         │   Kaggle Dataset     │
                         │    Cats vs Dogs      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ DVC / Data Versioning│
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Validation &         │
                         │ Pre-processing       │
                         │ 224×224 RGB          │
                         │ Train Augmentation   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Baseline CNN         │
                         │ Training             │
                         └──────────┬───────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
            ┌─────────────────┐          ┌─────────────────┐
            │ MLflow          │          │ Model Artifacts │
            │ Params/Metrics/ │          │ best_model.pt  │
            │ Artifacts       │          │ last_model.pt  │
            └─────────────────┘          └────────┬────────┘
                                                   │
                                                   ▼
                                          ┌─────────────────┐
                                          │ FastAPI         │
                                          │ /health         │
                                          │ /predict        │
                                          │ /metrics        │
                                          └────────┬────────┘
                                                   │
                                                   ▼
                                          ┌─────────────────┐
                                          │ Docker Image    │
                                          └────────┬────────┘
                                                   │
                                                   ▼
                                          ┌─────────────────┐
                                          │ GitHub Actions  │
                                          │ CI              │
                                          │ pytest → build  │
                                          └────────┬────────┘
                                                   │
                                                   ▼
                                          ┌─────────────────┐
                                          │ Container       │
                                          │ Registry        │
                                          └────────┬────────┘
                                                   │
                                                   ▼
                                          ┌─────────────────┐
                                          │ CD              │
                                          │ Docker Compose  │
                                          └────────┬────────┘
                                                   │
                                                   ▼
                                          ┌─────────────────┐
                                          │ Smoke Tests     │
                                          │ health +        │
                                          │ prediction      │
                                          └────────┬────────┘
                                                   │
                                                   ▼
                                          ┌─────────────────┐
                                          │ Monitoring      │
                                          │ logs + metrics  │
                                          └────────┬────────┘
                                                   │
                                                   ▼
                                          ┌─────────────────┐
                                          │ Post-deployment │
                                          │ evaluation      │
                                          └─────────────────┘
```

------------------------------------------------------------------------

# 4. Repository Structure

The repository separates ML development, inference, testing, deployment
and monitoring responsibilities.

``` text
mlops-2/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
│
├── api/
│   ├── main.py
│   ├── predictor.py
│   └── schemas.py
│
├── artifacts/
│   ├── models/
│   │   ├── best_model.pt
│   │   └── last_model.pt
│   ├── plots/
│   │   ├── loss_curve.png
│   │   ├── accuracy_curve.png
│   │   └── confusion_matrix.png
│   ├── logs/
│   │   └── api.log
│   └── reports/
│       ├── predictions.csv
│       └── evaluation_report.txt
│
├── configs/
├── data/
│
├── deployment/
│   ├── docker-compose.yml
│   ├── deploy.ps1
│   ├── deploy.sh
│   ├── smoke_test.py
│   └── prometheus.yml
├── grafana/dashboards
│			└── mlops-dashboard.json 
├── grafana/provisioning/datasources
│			└── datasource.yml
├── grafana/provisioning/dashboards
│			└── dashboard.yml
├── scripts/
├── src/
│   ├── dataset/
│   ├── models/
│   └── monitoring/
│
├── tests/
├── train.py
├── requirements.txt
├── Dockerfile
├── Makefile
├── dvc.yaml
└── README.md
```

> Keep this structure synchronized with the actual repository.
> Documentation should never claim a file exists when it does not.

------------------------------------------------------------------------

# 5. Environment Setup

## Prerequisites

Install:

-   Python 3.11
-   Git
-   Docker
-   DVC
-   GitHub account/repository access
-   Docker registry access
-   GNU Make is optional

### Windows

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Git Bash

``` bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify:

``` bash
python --version
python -m dvc --version
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import mlflow; print('MLflow:', mlflow.__version__)"
python -c "import fastapi; print('FastAPI:', fastapi.__version__)"
python -c "import pytest; print('pytest:', pytest.__version__)"
```

------------------------------------------------------------------------

# 6. M1 --- Model Development & Experiment Tracking

## Objective

Build, evaluate and track a reproducible baseline CNN while versioning
the dataset and source code.

The assignment requires Git for source versioning, DVC or Git-LFS for
dataset versioning, a baseline model, serialized model artifacts, and
experiment tracking with parameters, metrics and artifacts.
fileciteturn0file0L13-L25

## 6.1 Dataset

Source:

``` text
Cats and Dogs binary classification dataset from Kaggle
```

Required preprocessing:

``` text
Input image
   ↓
RGB conversion
   ↓
224 × 224 resize
   ↓
Train / Validation / Test split
   ↓
Training augmentation
```

The assignment recommends a split such as:

``` text
80% Train
10% Validation
10% Test
```

The exact counts should be determined from the dataset and configuration
rather than hard-coded into the README. fileciteturn0file0L8-L11

## 6.2 Data Versioning

DVC is used to version data/pipeline state.

Useful commands:

``` bash
python -m dvc status
python -m dvc dag
```

If the project has a configured DVC remote:

``` bash
python -m dvc pull
```

The Git repository should contain DVC metadata rather than unnecessarily
committing large raw datasets.

## 6.3 Dataset Validation

Run:

``` bash
python -m src.dataset.validator
```

Expected validation should establish:

``` text
Number of classes
Class distribution
Image availability
Image format/readability
Dataset split
```

## 6.4 Baseline CNN

The baseline model uses a CNN architecture suitable for binary image
classification.

Example conceptual flow:

``` text
224×224 RGB Image
      ↓
Conv2D
      ↓
ReLU
      ↓
MaxPool
      ↓
Conv2D
      ↓
ReLU
      ↓
MaxPool
      ↓
Conv2D
      ↓
ReLU
      ↓
Pooling
      ↓
Flatten / classifier
      ↓
2 class logits
```

Model implementation:

``` text
src/models/cnn.py
```

A model smoke test should verify that the network accepts the configured
input shape and returns two class outputs.

## 6.5 Training

Run:

``` bash
python train.py
```

or, if implemented:

``` bash
make m1-train
```

The training lifecycle should be:

``` text
Load configuration
      ↓
Load/versioned dataset
      ↓
Create datasets/loaders
      ↓
Create CNN
      ↓
Create loss function
      ↓
Create optimizer
      ↓
Start MLflow run
      ↓
Train
      ↓
Validate
      ↓
Save best checkpoint
      ↓
Save final checkpoint
      ↓
Evaluate test set
      ↓
Generate plots/reports
      ↓
Log artifacts to MLflow
```

For CPU development, a reduced configuration may be used to validate the
pipeline quickly. The **final experiment must use the documented final
configuration** and record image size, epochs, batch size, seed and
dataset version in MLflow.

## 6.6 Model Artifacts

Expected artifacts:

``` text
artifacts/models/
├── best_model.pt
└── last_model.pt
```

`best_model.pt` should represent the selected best validation
checkpoint.

`last_model.pt` should represent the final training checkpoint.

## 6.7 Evaluation

Generate appropriate metrics such as:

``` text
Loss
Accuracy
Precision
Recall
F1 Score
Confusion Matrix
```

Do not hard-code final performance numbers in this README. The final
measured values belong to the actual experiment/evaluation output.

## 6.8 Training Artifacts

``` text
artifacts/plots/
├── loss_curve.png
├── accuracy_curve.png
└── confusion_matrix.png
```

These provide evaluator evidence for:

-   Training convergence
-   Validation behavior
-   Classification errors
-   Final model performance

## 6.9 MLflow

Start the local UI:

``` bash
python -m mlflow ui
```

Open:

``` text
http://127.0.0.1:5000
```

Each run should contain:

### Parameters

``` text
image_size
batch_size
epochs
learning_rate
optimizer
random_seed
dataset/version information
model configuration
```

### Metrics

``` text
train_loss
train_accuracy
val_loss
val_accuracy
test_loss
test_accuracy
precision
recall
f1_score
```

### Artifacts

``` text
trained model
loss curve
accuracy curve
confusion matrix
evaluation report
```

This satisfies the assignment requirement to log runs, parameters,
metrics and artifacts. fileciteturn0file0L23-L25

------------------------------------------------------------------------

# 7. M2 --- Model Packaging & Containerization

## Objective

Expose the trained model through a REST API and package the complete
inference service into a reproducible Docker image.

The assignment requires FastAPI/Flask, a health endpoint, a prediction
endpoint, pinned dependencies, a Dockerfile and local prediction
verification. fileciteturn0file0L26-L38

## 7.1 API Components

``` text
api/
├── main.py
├── predictor.py
└── schemas.py
```

Responsibilities:

### `predictor.py`

``` text
Load model
   ↓
Validate/transform image
   ↓
Apply inference preprocessing
   ↓
Run model inference
   ↓
Calculate probabilities
   ↓
Return prediction
```

### `schemas.py`

Defines API request/response models where applicable.

### `main.py`

Defines the FastAPI application, endpoints and monitoring middleware.

## 7.2 Start API

``` bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Swagger UI:

``` text
http://127.0.0.1:8000/docs
```

Expected endpoints:

``` text
GET  /
GET  /health
GET  /metrics
POST /predict
```

## 7.3 Health Check

``` bash
curl.exe http://127.0.0.1:8000/health
```

Example response:

``` json
{
  "status": "healthy",
  "model_loaded": true
}
```

The exact response must match the implementation.

## 7.4 Prediction

Open:

``` text
http://127.0.0.1:8000/docs
```

Execute:

``` text
POST /predict
```

Example response structure:

``` json
{
  "label": "Cat",
  "class_id": 0,
  "confidence": 0.98,
  "probabilities": {
    "Cat": 0.98,
    "Dog": 0.02
  }
}
```

Exact values depend on the trained model.

## 7.5 Dependencies

Dependencies are declared in:

``` text
requirements.txt
```

Install:

``` bash
python -m pip install -r requirements.txt
```

Key runtime/ML dependencies should be pinned to versions tested by the
project.

## 7.6 Docker

Build:

``` bash
docker build -t cats-vs-dogs-api:latest .
```

Run:

``` bash
docker run --name cats-vs-dogs-api -p 8000:8000 cats-vs-dogs-api:latest
```

Verify:

``` bash
curl.exe http://127.0.0.1:8000/health
```

Then verify `/predict` with a test image.

------------------------------------------------------------------------

# 8. M3 --- CI Pipeline

## Objective

Automatically test the code, build the Docker image and publish the
image to a registry.

The assignment requires automated preprocessing/inference tests, a CI
system triggered on push/merge-request activity, dependency
installation, tests, Docker image creation and registry publishing.
fileciteturn0file0L39-L50

## 8.1 Unit Tests

Run:

``` bash
python -m pytest -v
```

At minimum:

``` text
Data preprocessing test
Model utility / inference test
```

Example:

``` bash
python -m pytest tests/test_preprocessing.py -v
python -m pytest tests/test_inference.py -v
```

All tests:

``` bash
python -m pytest -v
```

## 8.2 GitHub Actions

Workflow files:

``` text
.github/workflows/
├── ci.yml
└── cd.yml
```

The CI pipeline should follow:

``` text
Push / Pull Request
       ↓
Checkout
       ↓
Set up Python
       ↓
Install dependencies
       ↓
Run pytest
       ↓
Build Docker image
       ↓
Publish image
```

The assignment explicitly requires checkout, dependency installation,
unit tests and Docker image building in CI.
fileciteturn0file0L46-L50

## 8.3 Registry Publishing

A registry image should use a reproducible naming/tagging convention.

Preferred:

``` text
<registry>/<namespace>/cats-vs-dogs-api:<git-sha>
```

A human-readable release tag can also be used:

``` text
v1.0.0
```

Avoid relying only on `latest` for deployment because immutable tags
make rollback and audit easier.

### Credentials

Registry credentials must never be committed to Git.

Configure repository secrets under:

``` text
GitHub
→ Repository
→ Settings
→ Secrets and variables
→ Actions
```

Typical secrets:

``` text
DOCKERHUB_USERNAME
DOCKERHUB_TOKEN
```

Use the exact names configured by the workflow.

------------------------------------------------------------------------

# 9. M4 --- Continuous Deployment & Deployment

## Objective

Deploy the published image to the selected environment and automatically
verify the running service.

The selected deployment target for this project is:

``` text
Docker Compose
```

The assignment permits Docker Compose, a local Kubernetes cluster, or a
VM. fileciteturn0file0L51-L58

## 9.1 Deployment Files

``` text
deployment/
├── docker-compose.yml
├── deploy.ps1
├── deploy.sh
├── smoke_test.py
└── prometheus.yml
```

## 9.2 Deployment

``` bash
docker compose -f deployment/docker-compose.yml up -d
```

Check:

``` bash
docker compose -f deployment/docker-compose.yml ps
```

Logs:

``` bash
docker compose -f deployment/docker-compose.yml logs
```

Follow logs:

``` bash
docker compose -f deployment/docker-compose.yml logs -f
```

## 9.3 CD Flow

``` text
Main branch change
       ↓
Build / publish image
       ↓
Pull new image
       ↓
Deploy / update service
       ↓
Wait for service readiness
       ↓
Health check
       ↓
Prediction smoke test
       ↓
     ┌─ PASS → Deployment successful
     │
     └─ FAIL → Pipeline fails
```

The assignment requires deployment/update on main-branch changes and a
smoke test that fails the pipeline when health/prediction verification
fails. fileciteturn0file0L59-L66

## 9.4 Smoke Test

Run:

``` bash
python deployment/smoke_test.py
```

The smoke test should verify:

``` text
Service reachable
    ↓
/health succeeds
    ↓
/predict succeeds
    ↓
Response schema is valid
```

Expected:

``` text
SMOKE TEST PASSED
```

------------------------------------------------------------------------

# 10. M5 --- Monitoring, Logs & Post-Deployment Performance

## Objective

Observe the deployed inference service and evaluate the model using
post-deployment requests and true labels.

The assignment specifically requires request/response logging excluding
sensitive data, request count/latency tracking, and post-deployment
evaluation using real or simulated requests with true labels.
fileciteturn0file0L67-L80

## 10.1 Logging

Logging implementation:

``` text
src/monitoring/logger.py
```

Example log location:

``` text
artifacts/logs/api.log
```

Operational logs may include:

``` text
HTTP method
Request path
HTTP status
Latency
Success/failure
Prediction event
```

Do not log:

-   Raw uploaded images
-   Credentials
-   Tokens
-   Other sensitive request information

## 10.2 Request Metrics

Metrics implementation:

``` text
src/monitoring/metrics.py
```

Useful metrics:

``` text
request_count
success_count
failure_count
total_latency_ms
average_latency_ms
```

Endpoint:

``` text
GET /metrics
```

Test:

``` bash
curl.exe http://127.0.0.1:8000/metrics
```

If the implementation exposes a response header such as:

``` text
X-Latency-ms
```

it can also be used as simple request-level latency evidence.

## 10.3 Post-Deployment Performance

Performance tracker:

``` text
src/monitoring/performance_tracker.py
```

The evaluation dataset should contain:

``` text
Input image
True label
Predicted label
```

Possible metrics:

``` text
Accuracy
Precision
Recall
F1 Score
Classification Report
Confusion Matrix
```

Generated reports:

``` text
artifacts/reports/
├── predictions.csv
└── evaluation_report.txt
```

The README intentionally does not hard-code final model performance. The
measured values from the actual deployed model are the authoritative
results.

------------------------------------------------------------------------

# 11. CI/CD Quality Gates

The pipeline should stop promotion when a required gate fails.

``` text
Code Change
    │
    ├── Unit tests FAIL ────────► STOP
    │
    ├── Docker build FAIL ──────► STOP
    │
    ├── Registry publish FAIL ──► STOP
    │
    ├── Deployment FAIL ────────► STOP
    │
    └── Smoke test FAIL ────────► STOP
```

This establishes a basic production-style promotion model:

``` text
Test
 ↓
Package
 ↓
Publish
 ↓
Deploy
 ↓
Validate
```

------------------------------------------------------------------------

# 12. Reproducibility

Reproducibility is addressed through:

-   Git source-code versioning
-   DVC data/pipeline versioning
-   Pinned dependencies
-   Configuration management
-   MLflow experiment tracking
-   Serialized model artifacts
-   Docker packaging
-   Automated tests
-   CI/CD workflows
-   Deployment configuration
-   Smoke testing
-   Recorded evaluation outputs

The target is:

``` text
Versioned Code
      +
Versioned Data
      +
Pinned Environment
      +
Tracked Experiment
      +
Versioned Model
      +
Versioned Container
      =
Reproducible Inference Service
```

------------------------------------------------------------------------

# 13. Security & Production-Oriented Practices

Even though this is an academic assignment, the implementation follows
useful production principles:

### Secrets

-   Never commit credentials.
-   Store registry credentials as GitHub Actions secrets.
-   Never print secrets in CI logs.

### API

-   Validate uploaded images.
-   Restrict supported formats where appropriate.
-   Return meaningful HTTP errors.
-   Avoid exposing internal stack traces.

### Logging

-   Do not log sensitive image/request content.
-   Log operational metadata instead.

### Containers

-   Pin important dependencies.
-   Keep the runtime image minimal where practical.
-   Use health checks where appropriate.
-   Prefer immutable image tags for deployment.

### ML

-   Version data.
-   Track experiments.
-   Version model artifacts.
-   Record the model/data/configuration used for the deployed run.

------------------------------------------------------------------------

# 14. Evaluator Quick Start

This is the recommended shortest path for an evaluator.

## Step 1 --- Install

``` bash
python -m venv .venv
# activate .venv
python -m pip install -r requirements.txt
```

## Step 2 --- M1 Dataset

``` bash
python -m src.dataset.validator
```

## Step 3 --- M1 Model

``` bash
python -m src.models.cnn
```

Verify two class outputs.

## Step 4 --- M1 Training

``` bash
python train.py
```

Verify:

``` text
artifacts/models/best_model.pt
artifacts/models/last_model.pt
```

## Step 5 --- MLflow

``` bash
python -m mlflow ui
```

Open:

``` text
http://127.0.0.1:5000
```

Verify parameters, metrics and artifacts.

## Step 6 --- DVC

``` bash
python -m dvc status
python -m dvc dag
```

## Step 7 --- API

``` bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open:

``` text
http://127.0.0.1:8000/docs
```

## Step 8 --- Health

``` bash
curl.exe http://127.0.0.1:8000/health
```

## Step 9 --- Prediction

Use:

``` text
POST /predict
```

with a test image.

## Step 10 --- Tests

``` bash
python -m pytest -v
```

## Step 11 --- Docker

``` bash
docker build -t cats-vs-dogs-api:test .
docker run --rm -p 8000:8000 cats-vs-dogs-api:test
```

## Step 12 --- Deployment

``` bash
docker compose -f deployment/docker-compose.yml up -d
```

## Step 13 --- Smoke Test

``` bash
python deployment/smoke_test.py
```

## Step 14 --- Monitoring

``` bash
curl.exe http://127.0.0.1:8000/metrics
```

Inspect:

``` text
artifacts/logs/api.log
artifacts/reports/predictions.csv
artifacts/reports/evaluation_report.txt
```

## Step 15 --- CI/CD

Open:

``` text
GitHub → Actions
```

Verify:

``` text
CI:
checkout
→ dependencies
→ tests
→ Docker build
→ registry publish

CD:
image pull
→ deployment
→ health
→ prediction smoke test
```

------------------------------------------------------------------------

# 15. Makefile Automation

If GNU Make is available:

``` bash
make help
```

Suggested module commands:

``` bash
# M1
make m1-validate
make m1-model
make m1-train
make m1-mlflow
make m1-dvc

# M2
make m2-api
make m2-health
make m2-predict
make m2-docker-build
make m2-docker-run

# M3
make m3-test
make m3-image
make m3-publish

# M4
make m4-deploy
make m4-status
make m4-logs
make m4-smoke

# M5
make m5-metrics
make m5-logs
make m5-performance
make m5-report
```

Complete workflow:

``` bash
make all
```

If Make is unavailable on Windows, use the direct Python/Docker commands
documented above.

------------------------------------------------------------------------

# 16. Verification Checklist

## M1 --- 10 Marks

-   [ ] Git source versioning available
-   [ ] Dataset versioning configured with DVC/Git-LFS
-   [ ] Dataset validation completed
-   [ ] 224×224 RGB preprocessing implemented
-   [ ] Train/validation/test split implemented
-   [ ] Training augmentation implemented
-   [ ] Baseline CNN implemented
-   [ ] Model training completed
-   [ ] Best model checkpoint saved
-   [ ] Final model checkpoint saved
-   [ ] Evaluation metrics generated
-   [ ] Loss/accuracy plots generated
-   [ ] Confusion matrix generated
-   [ ] MLflow run available
-   [ ] Parameters logged
-   [ ] Metrics logged
-   [ ] Artifacts logged

## M2 --- 10 Marks

-   [ ] FastAPI service starts
-   [ ] Health endpoint works
-   [ ] Prediction endpoint works
-   [ ] Class probabilities/label returned
-   [ ] `requirements.txt` available
-   [ ] Key dependencies pinned
-   [ ] Dockerfile available
-   [ ] Docker image builds
-   [ ] Container starts
-   [ ] Container health check works
-   [ ] Container prediction works

## M3 --- 10 Marks

-   [ ] Preprocessing unit test
-   [ ] Inference/model utility unit test
-   [ ] pytest passes
-   [ ] GitHub Actions CI exists
-   [ ] CI installs dependencies
-   [ ] CI runs tests
-   [ ] CI builds Docker image
-   [ ] CI publishes image
-   [ ] Registry credentials stored as secrets

## M4 --- 10 Marks

-   [ ] Docker Compose configuration
-   [ ] Deployment starts
-   [ ] Service becomes reachable
-   [ ] Health check passes
-   [ ] Prediction works
-   [ ] Smoke-test script exists
-   [ ] CD workflow exists
-   [ ] Main-branch deployment flow configured
-   [ ] New image is deployed
-   [ ] Smoke-test failure fails the pipeline

## M5 --- 10 Marks

-   [ ] Request/response operational logging
-   [ ] Sensitive data excluded from logs
-   [ ] Request count
-   [ ] Success/failure count
-   [ ] Request latency
-   [ ] Metrics endpoint or equivalent
-   [ ] Post-deployment request batch
-   [ ] True labels collected
-   [ ] Accuracy
-   [ ] Precision
-   [ ] Recall
-   [ ] F1 score
-   [ ] Predictions CSV
-   [ ] Evaluation report

------------------------------------------------------------------------

# 17. Evidence for the Evaluator

## M1

Show:

1.  Dataset validation output
2.  Preprocessing configuration
3.  CNN/model smoke test
4.  Training output
5.  MLflow experiment
6.  Model checkpoints
7.  Loss curve
8.  Accuracy curve
9.  Confusion matrix
10. DVC status/pipeline

## M2

Show:

1.  FastAPI Swagger
2.  `/health`
3.  `/predict`
4.  Dockerfile
5.  Docker build
6.  Running container

## M3

Show:

1.  `pytest -v`
2.  `.github/workflows/ci.yml`
3.  Successful GitHub Actions run
4.  Docker image build
5.  Registry image

## M4

Show:

1.  `docker-compose.yml`
2.  Running deployment
3.  Deployment logs
4.  `/health`
5.  `/predict`
6.  Smoke-test result
7.  Successful CD workflow

## M5

Show:

1.  `/metrics`
2.  API logs
3.  Post-deployment predictions
4.  True labels
5.  Evaluation report
6.  Final measured metrics

------------------------------------------------------------------------

# 18. Five-Minute Demonstration Plan

The assignment requires a screen recording of **less than 5 minutes**
showing the complete workflow. fileciteturn0file0L76-L80

Recommended sequence:

``` text
00:00 – 00:30   Repository + architecture
00:30 – 01:00   M1: MLflow + model artifacts
01:00 – 01:30   M2: FastAPI /health + /predict
01:30 – 02:00   Docker image/container
02:00 – 02:45   M3: GitHub Actions CI
02:45 – 03:30   Registry + CD deployment
03:30 – 04:00   Smoke test
04:00 – 04:30   Logs + /metrics
04:30 – 05:00   Post-deployment report + final result
```

The strongest evidence is a **single continuous story from code change
to deployed prediction**, rather than a collection of disconnected
screenshots.

------------------------------------------------------------------------

# 19. Troubleshooting

## Uvicorn command not found

Use:

``` bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

## MLflow command not found

Use:

``` bash
python -m mlflow ui
```

## DVC command not found

Use:

``` bash
python -m dvc --version
python -m dvc status
```

## Port 8000 already in use

Windows:

``` powershell
netstat -ano | findstr :8000
```

Then stop the process or use another port.

## Pytest import mismatch

Avoid duplicate test module names in different directories.

For example, do not create two unrelated modules both named:

``` text
smoke_test.py
```

Use unique names such as:

``` text
deployment/smoke_test.py
scripts/api_smoke_test.py
```

Clear Python caches if required:

``` powershell
Get-ChildItem -Path . -Recurse -Directory -Filter __pycache__ |
Remove-Item -Recurse -Force
```

Then:

``` bash
python -m pytest -v
```

## CPU training is slow

Use a reduced configuration only for pipeline verification.

For the final submission, use the documented final configuration and
record the final experiment in MLflow.

------------------------------------------------------------------------

# 20. Final MLOps Lifecycle

``` text
                    DATA
                     │
                     ▼
              Version with DVC
                     │
                     ▼
              Pre-processing
              224×224 RGB
                     │
                     ▼
             Train / Val / Test
                     │
                     ▼
               Baseline CNN
                     │
                     ▼
            MLflow Experiment
                     │
                     ▼
              Model Artifact
                     │
                     ▼
               FastAPI API
              /health /predict
                     │
                     ▼
                 Docker
                     │
                     ▼
              Unit Testing
                     │
                     ▼
               GitHub CI
                     │
                     ▼
             Container Registry
                     │
                     ▼
               GitHub CD
                     │
                     ▼
             Docker Compose
                     │
                     ▼
              Smoke Testing
                     │
                     ▼
              Logs + Metrics
                     │
                     ▼
         Post-Deployment Evaluation
                     │
                     ▼
                  REPORT
```

------------------------------------------------------------------------

# 21. Conclusion

This project demonstrates the complete MLOps lifecycle for a **Cats vs
Dogs binary image-classification service**:

**Version → Train → Track → Evaluate → Package → Test → Containerize →
Build → Publish → Deploy → Smoke Test → Monitor → Evaluate**

The implementation directly addresses all five assignment modules:

-   **M1:** Model Development & Experiment Tracking
-   **M2:** Model Packaging & Containerization
-   **M3:** CI --- Build, Test & Image Creation
-   **M4:** CD & Deployment
-   **M5:** Monitoring, Logs & Final Submission

The repository is structured so an evaluator can move from the
assignment requirement to the corresponding code, artifact, workflow or
runtime evidence without needing to understand every implementation
detail.

------------------------------------------------------------------------

## Assignment Reference

The supplied assignment requires an open-source end-to-end MLOps
pipeline for Cats vs Dogs classification, including 224×224 RGB
preprocessing, dataset splitting/augmentation, model development,
experiment tracking, REST inference, Docker, automated testing, CI,
registry publishing, CD/deployment, smoke testing and monitoring.

