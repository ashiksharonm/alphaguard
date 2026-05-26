"""
api/routers/health.py — Health and readiness endpoints.
"""
from __future__ import annotations
import time
from fastapi import APIRouter, Request
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from api.schemas import HealthResponse, ModelInfoResponse

router = APIRouter()
_start_time = time.time()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health(request: Request):
    """Liveness + readiness probe. Used by Jenkins smoke test."""
    predictor = request.app.state.predictor
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None,
        model_version=predictor.version if predictor else "not loaded",
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@router.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info(request: Request):
    """Return model metadata — version, AUC, threshold, top features."""
    meta = request.app.state.metadata
    return ModelInfoResponse(
        version=meta["version"],
        cv_auc=meta["cv_auc"],
        cv_ap=meta["cv_ap"],
        train_recall=meta["train_recall"],
        train_precision=meta["train_precision"],
        threshold=meta["threshold"],
        n_features_raw=meta["n_features_raw"],
        n_features_selected=meta["n_features_sel"],
        models=meta["models"],
        top_features=meta["top_features"],
    )


@router.get("/metrics", tags=["Observability"])
async def metrics():
    """Prometheus metrics scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
