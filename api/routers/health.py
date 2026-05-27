"""
api/routers/health.py — Health and readiness endpoints.
"""
from __future__ import annotations
import time
from fastapi import APIRouter, HTTPException, Request
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from api.schemas import HealthResponse, ModelInfoResponse

router = APIRouter()
_start_time = time.time()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health(request: Request):
    """Liveness + readiness probe. Used by Jenkins smoke test."""
    predictor = getattr(request.app.state, "predictor", None)
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None,
        model_version=predictor.version if predictor else "not loaded",
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@router.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info(request: Request):
    """Return model metadata — version, AUC, threshold, top features."""
    meta = getattr(request.app.state, "metadata", {})
    if not meta:
        raise HTTPException(status_code=503, detail="Model not loaded — run python ml/train.py first")
    return ModelInfoResponse(
        version=meta.get("version", "unknown"),
        cv_auc=meta.get("cv_auc", 0.0),
        cv_ap=meta.get("cv_ap", 0.0),
        train_recall=meta.get("train_recall", 0.0),
        train_precision=meta.get("train_precision", 0.0),
        threshold=meta.get("threshold", 0.0),
        n_features_raw=meta.get("n_features_raw", 0),
        n_features_selected=meta.get("n_features_sel", 0),
        models=meta.get("models", []),
        top_features=meta.get("top_features", []),
    )


@router.get("/metrics", tags=["Observability"])
async def metrics():
    """Prometheus metrics scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
