"""
api/middleware.py — Prometheus metrics middleware.
Exposes request count, latency histogram, and error counter.
"""
from __future__ import annotations
import time
from prometheus_client import Counter, Histogram, Gauge, REGISTRY
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# ── Metrics ───────────────────────────────────────────────────────────────────
REQUESTS_TOTAL = Counter(
    "alphaguard_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "alphaguard_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
PREDICTIONS_TOTAL = Counter(
    "alphaguard_predictions_total",
    "Total coil predictions made",
)
DEFECTS_FLAGGED = Counter(
    "alphaguard_defects_flagged_total",
    "Total defects flagged",
)
MODEL_AUC = Gauge(
    "alphaguard_model_auc",
    "Model cross-validated AUC-ROC",
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Record request metrics for every HTTP call."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        endpoint = request.url.path
        REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=endpoint,
            status=response.status_code,
        ).inc()
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=endpoint,
        ).observe(duration)

        return response
