"""FastAPI service exposing the Phase 3 risk and claim models.

Endpoints
---------
GET  /health           liveness + readiness, reports which models are loaded
GET  /metadata         served model versions, targets, thresholds
POST /predict/risk     Model A — visit risk (Low / Medium / High)
POST /predict/claim    Model B — claim outcome (Paid / Pending / Rejected)
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from time import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from . import __version__
from .config import get_settings
from .logging_config import log_prediction
from .model_service import service
from .schemas import (
    ClaimPredictionRequest,
    HealthResponse,
    ModelMetadata,
    PredictionResponse,
    RiskPredictionRequest,
)

_START_TIME = time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model artifacts once, at process start.
    service.load()
    yield
    # nothing to tear down — artifacts are GC'd with the process.


app = FastAPI(
    title="Hospital ML Prediction API",
    description=(
        "Real-time inference for visit-risk and claim-outcome classification. "
        "Send raw business inputs; feature engineering is performed server-side."
    ),
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness + readiness probe (used by ECS / load balancer health checks)."""
    settings = get_settings()
    return HealthResponse(
        status="ok" if service.ready else "degraded",
        service=settings.SERVICE_NAME,
        model_version=settings.MODEL_VERSION,
        models_loaded={
            "risk_model": service.risk is not None,
            "claim_model": service.claim is not None,
        },
        uptime_seconds=round(time() - _START_TIME, 2),
    )


@app.get("/metadata", response_model=dict[str, ModelMetadata], tags=["ops"])
def metadata() -> dict:
    """Served-model metadata for dashboards and governance."""
    if not service.ready:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return service.metadata()


@app.post("/predict/risk", response_model=PredictionResponse, tags=["predict"])
def predict_risk(payload: RiskPredictionRequest) -> PredictionResponse:
    """Model A — predict operational/clinical risk for a single visit."""
    if not service.ready:
        raise HTTPException(status_code=503, detail="Models not loaded")
    request_id = str(uuid.uuid4())
    try:
        result = service.predict_risk(payload.model_dump(mode="json"))
    except ValueError as exc:  # bad category etc.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _finalize(request_id, result)


@app.post("/predict/claim", response_model=PredictionResponse, tags=["predict"])
def predict_claim(payload: ClaimPredictionRequest) -> PredictionResponse:
    """Model B — predict claim outcome before submission."""
    if not service.ready:
        raise HTTPException(status_code=503, detail="Models not loaded")
    request_id = str(uuid.uuid4())
    try:
        result = service.predict_claim(payload.model_dump(mode="json"))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _finalize(request_id, result)


def _finalize(request_id: str, result: dict) -> PredictionResponse:
    result["request_id"] = request_id
    # Audit log: timestamp, model version, feature hash, prediction, latency.
    log_prediction(
        {
            "event": "prediction",
            "request_id": request_id,
            "model_name": result["model_name"],
            "model_version": result["model_version"],
            "prediction": result["prediction"],
            "probabilities": result["probabilities"],
            "feature_hash": result["feature_hash"],
            "threshold_applied": result["threshold_applied"],
            "timestamp": result["timestamp"],
            "latency_ms": result["latency_ms"],
        }
    )
    return PredictionResponse(**result)


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    # Never leak stack traces to clients; log server-side context.
    log_prediction({"event": "error", "path": str(request.url), "error": str(exc)})
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
