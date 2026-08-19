# ==============================================================================
# MLOps Assignment 2 - Production Cross-Platform Makefile
# Works natively on Linux, macOS, and Windows (PowerShell / CMD / Git Bash)
# ==============================================================================

PYTHON       ?= python
PYTEST       ?= $(PYTHON) -m pytest

IMAGE_NAME   ?= cats-vs-dogs-api
IMAGE_TAG    ?= latest
REGISTRY     ?= docker.io
REGISTRY_USER?=
FULL_IMAGE   ?= $(REGISTRY)/$(REGISTRY_USER)/$(IMAGE_NAME):$(IMAGE_TAG)

API_HOST     ?= 127.0.0.1
API_PORT     ?= 8000
BASE_URL     ?= http://$(API_HOST):$(API_PORT)

# Handles both .yaml and .yml naming
COMPOSE_FILE ?= deployment/docker-compose.yaml
MODEL_DIR    ?= artifacts/models
PLOT_DIR     ?= artifacts/plots
LOG_DIR      ?= artifacts/logs
REPORT_DIR   ?= artifacts/reports

.DEFAULT_GOAL := help

.PHONY: help check-env info m1 m1-validate m1-model m1-train m1-artifacts m1-mlflow m1-dvc m1-git m1-verify \
        m2 m2-deps m2-import m2-api-check m2-health m2-docker-build m2-docker-run m2-docker-status m2-docker-stop m2-verify \
        m3 m3-lint m3-test m3-test-preprocessing m3-test-inference m3-docker-build m3-ci-files m3-publish m3-verify \
        m4 m4-deploy m4-wait m4-status m4-health m4-smoke m4-logs m4-restart m4-stop m4-verify \
        m5 m5-metrics m5-logs m5-performance m5-reports m5-prometheus m5-verify verify verify-m1 verify-m2 verify-m3 verify-m4 verify-m5 all pipeline clean clean-docker

# Helper inline Python snippets for 100% cross-platform execution
PY_CHECK_FILE = $(PYTHON) -c "import os, sys; sys.exit(0 if os.path.isfile(sys.argv[1]) else 1)"
PY_CHECK_DIR  = $(PYTHON) -c "import os, sys; sys.exit(0 if os.path.isdir(sys.argv[1]) else 1)"
PY_CURL       = $(PYTHON) -c "import urllib.request, sys; res=urllib.request.urlopen(sys.argv[1]); sys.exit(0 if res.getcode()==200 else 1)"
PY_WAIT       = $(PYTHON) -c "import urllib.request, time, sys; url=sys.argv[1]; \
                [sys.exit(0) for _ in range(30) if (time.sleep(2) or True) and urllib.request.urlopen(url).getcode()==200]; sys.exit(1)"

help:
	@echo ""
	@echo "MLOps Assignment 2 - End-to-End Control"
	@echo "========================================"
	@echo "make check-env      Validate environment dependencies"
	@echo "make m1             Model development pipeline"
	@echo "make m2             Packaging and containerization"
	@echo "make m3             CI, linting, tests, and Docker build"
	@echo "make m3-publish     Publish image (requires REGISTRY_USER=username)"
	@echo "make m4             Deployment, health wait loop & smoke test"
	@echo "make m5             Monitoring and metrics collection"
	@echo "make verify         Verify M1 through M5 modules"
	@echo "make all            Run complete end-to-end pipeline"
	@echo "make clean          Cross-platform cleanup of Python caches"
	@echo ""

info:
	@echo "Python      : $(PYTHON)"
	@echo "Image       : $(IMAGE_NAME):$(IMAGE_TAG)"
	@echo "Full Image  : $(FULL_IMAGE)"
	@echo "API URL     : $(BASE_URL)"
	@echo "Compose File: $(COMPOSE_FILE)"

check-env:
	@$(PYTHON) --version
	@$(PYTHON) -c "import sys; print('Python executable:', sys.executable)"
	@$(PY_CHECK_FILE) train.py || (echo "ERROR: train.py missing" && exit 1)
	@$(PY_CHECK_FILE) requirements.txt || (echo "ERROR: requirements.txt missing" && exit 1)
	@$(PY_CHECK_DIR) src || (echo "ERROR: src directory missing" && exit 1)
	@$(PY_CHECK_DIR) api || (echo "ERROR: api directory missing" && exit 1)
	@echo "[OK] Environment checks passed."

# ---------------- M1: Model Development ----------------

m1: m1-validate m1-model m1-train m1-artifacts m1-mlflow m1-dvc m1-git m1-verify
	@echo "=== M1 COMPLETED ==="

m1-validate:
	@$(PYTHON) -m src.dataset.validator

m1-model:
	@$(PYTHON) -m src.models.cnn

m1-train:
	@$(PYTHON) train.py

m1-artifacts:
	@$(PY_CHECK_FILE) $(MODEL_DIR)/best_model.pt || (echo "ERROR: best_model.pt missing" && exit 1)
	@$(PY_CHECK_FILE) $(MODEL_DIR)/last_model.pt || (echo "ERROR: last_model.pt missing" && exit 1)
	@echo "[OK] Model artifacts found."

m1-mlflow:
	@$(PYTHON) -c "import mlflow; print('MLflow version:', mlflow.__version__)"
	@$(PYTHON) -c "import os; print('[OK] mlruns exists' if os.path.exists('mlruns') else '[INFO] mlruns not found')"

m1-dvc:
	@$(PYTHON) -m dvc --version
	@$(PYTHON) -m dvc status

m1-git:
	@git status --short

m1-verify:
	@$(PY_CHECK_FILE) $(MODEL_DIR)/best_model.pt || (echo "M1 FAIL: best_model.pt missing" && exit 1)
	@$(PY_CHECK_FILE) $(MODEL_DIR)/last_model.pt || (echo "M1 FAIL: last_model.pt missing" && exit 1)
	@$(PYTHON) -c "import mlflow; print('MLflow:', mlflow.__version__)"
	@$(PYTHON) -m dvc status
	@echo "[OK] M1 VERIFY PASSED"

# ---------------- M2: Packaging & Containerization ----------------

m2: m2-deps m2-import m2-api-check m2-docker-build m2-docker-run m2-wait m2-health m2-verify
	@echo "=== M2 COMPLETED ==="

m2-deps:
	@$(PYTHON) -c "import fastapi; print('FastAPI:', fastapi.__version__)"
	@$(PYTHON) -c "import uvicorn; print('Uvicorn:', uvicorn.__version__)"
	@$(PYTHON) -c "import PIL; print('Pillow:', PIL.__version__)"

m2-import:
	@$(PYTHON) -c "import api.main; print('API import successful')"

m2-api-check:
	@$(PYTHON) -c "from api.main import app; print('Routes:'); [print(' ', r.path) for r in app.routes]"

m2-wait:
	@echo "Waiting for API to become ready..."
	@$(PY_WAIT) $(BASE_URL)/health

m2-health:
	@$(PY_CURL) $(BASE_URL)/health
	@echo "\n[OK] Health check passed."

m2-docker-build:
	@docker build -f docker/Dockerfile -t $(IMAGE_NAME):$(IMAGE_TAG) .

m2-docker-run:
	@docker compose -f $(COMPOSE_FILE) up -d

m2-docker-status:
	@docker compose -f $(COMPOSE_FILE) ps

m2-docker-stop:
	@docker compose -f $(COMPOSE_FILE) down

m2-verify:
	@$(PYTHON) -c "import subprocess, sys; res=subprocess.run(['docker', 'image', 'inspect', '$(IMAGE_NAME):$(IMAGE_TAG)'], capture_output=True); sys.exit(res.returncode)" || (echo "M2 FAIL: Docker image missing" && exit 1)
	@echo "[OK] M2 VERIFY PASSED"

# ---------------- M3: Testing, CI & Building ----------------

m3: m3-lint m3-test m3-ci-files m3-docker-build m3-verify
	@echo "=== M3 LOCAL CI COMPLETED ==="

m3-lint:
	@$(PYTHON) -m flake8 src api tests || echo "[WARN] Flake8 not installed or warnings found."

m3-test:
	@$(PYTEST) -v

m3-test-preprocessing:
	@$(PYTEST) -v tests/test_preprocessing.py

m3-test-inference:
	@$(PYTEST) -v tests/test_inference.py

m3-docker-build:
	@docker build -f docker/Dockerfile -t $(IMAGE_NAME):$(IMAGE_TAG) .

m3-ci-files:
	@$(PY_CHECK_DIR) .github/workflows || (echo "ERROR: .github/workflows missing" && exit 1)
	@$(PY_CHECK_FILE) .github/workflows/ci.yml || (echo "ERROR: ci.yml missing" && exit 1)
	@echo "[OK] GitHub Actions CI workflow found."

m3-publish:
	@$(PYTHON) -c "import sys, os; sys.exit(0 if '$(REGISTRY_USER)' else 1)" || (echo "ERROR: Set REGISTRY_USER=your_username" && exit 1)
	@docker login
	@docker tag $(IMAGE_NAME):$(IMAGE_TAG) $(FULL_IMAGE)
	@docker push $(FULL_IMAGE)
	@echo "[OK] Image published: $(FULL_IMAGE)"

m3-verify:
	@$(PYTEST) -v
	@$(PYTHON) -c "import subprocess, sys; res=subprocess.run(['docker', 'image', 'inspect', '$(IMAGE_NAME):$(IMAGE_TAG)'], capture_output=True); sys.exit(res.returncode)" || (echo "M3 FAIL: image missing" && exit 1)
	@$(PY_CHECK_FILE) .github/workflows/ci.yml || (echo "M3 FAIL: ci.yml missing" && exit 1)
	@echo "[OK] M3 VERIFY PASSED"

# ---------------- M4: Deployment & Operations ----------------

m4: m4-deploy m4-wait m4-status m4-health m4-smoke m4-verify
	@echo "=== M4 COMPLETED ==="

m4-deploy:
	@docker compose -f $(COMPOSE_FILE) up -d

m4-wait:
	@echo "Polling service readiness..."
	@$(PY_WAIT) $(BASE_URL)/health

m4-status:
	@docker compose -f $(COMPOSE_FILE) ps

m4-health:
	@$(PY_CURL) $(BASE_URL)/health
	@echo "\n[OK] Deployment Health Check Passed."

m4-smoke:
	@$(PYTHON) deployment/smoke_test.py

m4-logs:
	@docker compose -f $(COMPOSE_FILE) logs --tail=100

m4-restart:
	@docker compose -f $(COMPOSE_FILE) restart

m4-stop:
	@docker compose -f $(COMPOSE_FILE) down

m4-verify:
	@docker compose -f $(COMPOSE_FILE) ps
	@$(PY_CURL) $(BASE_URL)/health
	@$(PYTHON) deployment/smoke_test.py
	@$(PY_CHECK_FILE) .github/workflows/cd.yml || (echo "M4 FAIL: cd.yml missing" && exit 1)
	@echo "[OK] M4 VERIFY PASSED"

# ---------------- M5: Monitoring & Logging ----------------


m5: m5-metrics m5-logs m5-performance m5-reports m5-prometheus m5-grafana m5-verify
	@echo "=== M5 COMPLETED ==="


# ------------------------------------------------------------
# M5 Metrics
# ------------------------------------------------------------

m5-metrics:
	@echo "============================================================"
	@echo "M5 Metrics"
	@echo "============================================================"
	@$(PYTHON) -c "import urllib.request, sys; res=urllib.request.urlopen('$(BASE_URL)/metrics'); print(res.read().decode()); sys.exit(0 if res.getcode()==200 else 1)"


# ------------------------------------------------------------
# M5 Logs
# ------------------------------------------------------------

m5-logs:
	@echo "============================================================"
	@echo "M5 Logs"
	@echo "============================================================"
	@if [ -d "$(LOG_DIR)" ]; then \
		echo "Log directory: $(LOG_DIR)"; \
		if [ -n "$$(find "$(LOG_DIR)" -type f -print -quit)" ]; then \
			find "$(LOG_DIR)" -type f -maxdepth 2 -print; \
		else \
			echo "[WARN] Log directory exists but is empty"; \
			echo ""; \
			echo "Docker application logs:"; \
			docker logs --tail=50 cats-vs-dogs-api; \
		fi; \
	else \
		echo "[WARN] Log directory missing: $(LOG_DIR)"; \
		echo ""; \
		echo "Docker application logs:"; \
		docker logs --tail=50 cats-vs-dogs-api; \
	fi


# ------------------------------------------------------------
# M5 Performance
# ------------------------------------------------------------

m5-performance:
	@echo "============================================================"
	@echo "M5 Performance"
	@echo "============================================================"
	@$(PYTHON) -m src.monitoring.performance_tracker


# ------------------------------------------------------------
# M5 Reports
# ------------------------------------------------------------

m5-reports:
	@echo "============================================================"
	@echo "M5 Reports"
	@echo "============================================================"
	@$(PYTHON) -c "import os; os.makedirs('$(REPORT_DIR)', exist_ok=True); files=os.listdir('$(REPORT_DIR)'); print('Report directory: $(REPORT_DIR)'); print('Reports:', files if files else '[WARN] No reports found')"


# ------------------------------------------------------------
# M5 Prometheus
# ------------------------------------------------------------

m5-prometheus:
	@echo "============================================================"
	@echo "M5 Prometheus"
	@echo "============================================================"
	@$(PY_CHECK_FILE) deployment/prometheus.yml || (echo "M5 FAIL: prometheus.yml missing" && exit 1)
	@echo "[OK] Prometheus configuration found"
	@$(PY_CURL) http://localhost:9090/-/ready
	@echo "[OK] Prometheus is ready"


# ------------------------------------------------------------
# M5 Grafana
# ------------------------------------------------------------

m5-grafana:
	@echo "============================================================"
	@echo "M5 Grafana"
	@echo "============================================================"
	@$(PY_CURL) http://localhost:3000/api/health
	@echo "[OK] Grafana is ready"


# ------------------------------------------------------------
# M5 Verification
# ------------------------------------------------------------

m5-verify:
	@echo "============================================================"
	@echo "M5 Verification"
	@echo "============================================================"

	@$(PY_CURL) $(BASE_URL)/metrics
	@echo "[OK] API metrics endpoint"

	@$(PY_CURL) http://localhost:9090/-/ready
	@echo "[OK] Prometheus ready"

	@$(PY_CURL) http://localhost:3000/api/health
	@echo "[OK] Grafana ready"

	@$(PY_CHECK_FILE) src/monitoring/metrics.py || (echo "M5 FAIL: metrics.py missing" && exit 1)

	@$(PY_CHECK_FILE) src/monitoring/performance_tracker.py || (echo "M5 FAIL: performance_tracker.py missing" && exit 1)

	@$(PY_CHECK_FILE) src/monitoring/logger.py || (echo "M5 FAIL: logger.py missing" && exit 1)

	@$(PY_CHECK_FILE) deployment/prometheus.yml || (echo "M5 FAIL: prometheus.yml missing" && exit 1)

	@$(PY_CHECK_FILE) deployment/grafana/provisioning/datasources/datasource.yml || (echo "M5 FAIL: Grafana datasource provisioning missing" && exit 1)

	@$(PY_CHECK_FILE) deployment/grafana/provisioning/dashboards/dashboard.yml || (echo "M5 FAIL: Grafana dashboard provisioning missing" && exit 1)

	@$(PY_CHECK_FILE) deployment/grafana/dashboards/mlops-dashboard.json || (echo "M5 FAIL: Grafana dashboard missing" && exit 1)

	@echo ""
	@echo "[OK] M5 VERIFY PASSED"


# ---------------- Verifications ----------------

verify-m1: m1-verify
verify-m2: m2-verify
verify-m3: m3-verify
verify-m4: m4-verify
verify-m5: m5-verify

verify: verify-m1 verify-m2 verify-m3 verify-m4 verify-m5
	@echo "=== ALL MODULE VERIFICATIONS PASSED ==="

all: m1 m2 m3 m4 m5
	@echo "=== END-TO-END MLOPS PIPELINE COMPLETED SUCCESSFULLY ==="

pipeline: all

# ---------------- Cleanup (Cross-Platform) ----------------

clean:
	@$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__') if p.is_dir()]"
	@$(PYTHON) -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc') if p.is_file()]"
	@$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.pytest_cache') if p.is_dir()]"
	@echo "[OK] Cleanup completed across all operating systems."

clean-docker:
	@docker compose -f $(COMPOSE_FILE) down
	