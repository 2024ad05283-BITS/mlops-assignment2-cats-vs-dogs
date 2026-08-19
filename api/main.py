"""
main.py

FastAPI inference service.

Endpoints
---------
GET  /
GET  /health
GET  /metrics
GET  /metrics/prometheus
POST /predict

Run
---
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
"""

from io import BytesIO
import logging
import time

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
)

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from PIL import Image

from starlette.middleware.base import BaseHTTPMiddleware

from api.predictor import Predictor

from api.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
)


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger("api")


# ============================================================
# Request Metrics
# ============================================================

REQUEST_COUNT = 0
SUCCESS_COUNT = 0
FAILURE_COUNT = 0
TOTAL_LATENCY = 0.0


# ============================================================
# Prometheus Metrics
# IMPORTANT: These MUST be defined before they are used.
# ============================================================

API_REQUESTS = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["method", "path", "status"],
)

API_REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds",
    ["method", "path"],
)

MODEL_READY_METRIC = Gauge(
    "model_ready",
    "Whether the inference model is loaded",
)

MODEL_PREDICTIONS = Counter(
    "model_predictions_total",
    "Total number of model predictions",
    ["label"],
)


# ============================================================
# Monitoring Middleware
# ============================================================

class MonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request counting,
    Prometheus metrics and latency logging.
    """

    async def dispatch(
        self,
        request: Request,
        call_next,
    ):
        global REQUEST_COUNT
        global SUCCESS_COUNT
        global FAILURE_COUNT
        global TOTAL_LATENCY

        REQUEST_COUNT += 1

        start = time.perf_counter()

        response = None

        try:

            response = await call_next(request)

            if response.status_code < 400:

                SUCCESS_COUNT += 1

            else:

                FAILURE_COUNT += 1

            API_REQUESTS.labels(
                method=request.method,
                path=request.url.path,
                status=str(response.status_code),
            ).inc()

        except Exception:

            FAILURE_COUNT += 1

            API_REQUESTS.labels(
                method=request.method,
                path=request.url.path,
                status="500",
            ).inc()

            raise

        finally:

            latency = (
                time.perf_counter() - start
            )

            TOTAL_LATENCY += latency

            API_REQUEST_LATENCY.labels(
                method=request.method,
                path=request.url.path,
            ).observe(latency)

            logger.info(
                "Request=%s | "
                "Path=%s | "
                "Status=%s | "
                "Latency=%.2f ms | "
                "Count=%d",
                request.method,
                request.url.path,
                (
                    response.status_code
                    if response is not None
                    else 500
                ),
                latency * 1000,
                REQUEST_COUNT,
            )

        if response is not None:

            response.headers[
                "X-Latency-ms"
            ] = f"{latency * 1000:.2f}"

        return response


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Cats vs Dogs Classifier",
    version="1.0.0",
    description="Production-ready FastAPI inference service",
)


# ============================================================
# Register Middleware
# ============================================================

app.add_middleware(
    MonitoringMiddleware
)


# ============================================================
# Load Model Once
# ============================================================

try:

    logger.info(
        "Loading inference model..."
    )

    predictor = Predictor()

    MODEL_READY = True

    MODEL_READY_METRIC.set(1)

    MODEL_ERROR = None

    logger.info(
        "Inference model loaded successfully."
    )

except Exception as e:

    predictor = None

    MODEL_READY = False

    MODEL_READY_METRIC.set(0)

    MODEL_ERROR = str(e)

    logger.exception(
        "Failed to load inference model."
    )


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Cats vs Dogs Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "prometheus": "/metrics/prometheus",
    }


# ============================================================
# Health Check
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():

    if MODEL_READY:

        return HealthResponse(
            status="healthy",
            model_loaded=True,
        )

    raise HTTPException(
        status_code=500,
        detail=(
            f"Model not loaded: "
            f"{MODEL_ERROR}"
        ),
    )


# ============================================================
# Application Metrics
# ============================================================

@app.get("/metrics")
def metrics():

    average_latency_ms = 0.0

    if REQUEST_COUNT > 0:

        average_latency_ms = (
            TOTAL_LATENCY
            / REQUEST_COUNT
            * 1000
        )

    return {

        "request_count":
            REQUEST_COUNT,

        "success_count":
            SUCCESS_COUNT,

        "failure_count":
            FAILURE_COUNT,

        "total_latency_ms":
            round(
                TOTAL_LATENCY * 1000,
                2,
            ),

        "average_latency_ms":
            round(
                average_latency_ms,
                2,
            ),

        "model_ready":
            MODEL_READY,

        "status":
            "running",
    }


# ============================================================
# Prometheus Metrics Endpoint
# ============================================================

@app.get("/metrics/prometheus")
def prometheus_metrics():

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ============================================================
# Prediction Endpoint
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {
            "model": ErrorResponse,
        },
        500: {
            "model": ErrorResponse,
        },
    },
)
async def predict(
    file: UploadFile = File(...),
):

    if not MODEL_READY:

        raise HTTPException(
            status_code=500,
            detail=(
                "Model not loaded: "
                f"{MODEL_ERROR}"
            ),
        )

    try:

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        contents = await file.read()

        image = Image.open(
            BytesIO(contents)
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        result = predictor.predict(
            image
        )

        # ----------------------------------------------------
        # Prometheus prediction metric
        # ----------------------------------------------------

        MODEL_PREDICTIONS.labels(
            label=result["label"]
        ).inc()

        return PredictionResponse(
            **result
        )

    except Exception as e:

        logger.exception(
            "Prediction failed."
        )

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )


# ============================================================
# Local Run
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )