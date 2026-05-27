"""
api/main.py — AlphaGuard FastAPI Application
"""
from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.middleware import PrometheusMiddleware, MODEL_AUC
from api.routers import health, predict


ARTIFACTS_DIR = Path("ml/artifacts")
MODEL_PATH    = ARTIFACTS_DIR / "model.pkl"
META_PATH     = ARTIFACTS_DIR / "metadata.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    # Always pre-initialize state so endpoints never throw AttributeError
    app.state.predictor   = None
    app.state.metadata    = {}
    app.state.start_time  = time.time()

    print("Loading AlphaGuard model...")
    try:
        from ml.predict import AlphaGuardPredictor
        app.state.predictor = AlphaGuardPredictor.load(MODEL_PATH)
        with open(META_PATH) as f:
            app.state.metadata = json.load(f)
        MODEL_AUC.set(app.state.metadata.get("cv_auc", 0))
        print(f"Model loaded: v{app.state.predictor.version} | AUC={app.state.metadata.get('cv_auc')}")
    except FileNotFoundError:
        print("WARNING: No model artifact found. Run: python ml/train.py")
    yield
    print("Shutting down.")


app = FastAPI(
    title="AlphaGuard — Alpha Defect Detection API",
    description=(
        "Production ML API for detecting Alpha defects in Tata Steel Hot Rolling Mill coils.\n\n"
        "**Key properties:**\n"
        "- Recall = 100% (zero false negatives)\n"
        "- 6-model ensemble (RF, ET, XGB, LGB, GB, LR)\n"
        "- Real-time inference (<100ms per batch)\n"
        "- Prometheus metrics at `/metrics`"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware
app.add_middleware(PrometheusMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router)
app.include_router(predict.router)

# Serve frontend
FRONTEND = Path("frontend")
if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_dashboard():
        return FileResponse(str(FRONTEND / "index.html"))
