# Multi-stage Dockerfile for AlphaGuard
# Stage 1 (trainer): installs deps + trains the model
# Stage 2 (runtime): lean image with only what's needed to serve

# ── Stage 1: Trainer ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS trainer

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 && rm -rf /var/lib/apt/lists/*

# Install Python deps (cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source + data (data is bind-mounted or COPY'd in CI)
COPY ml/ ml/
COPY setup.py .
RUN pip install -e . --no-deps

# Train — produces ml/artifacts/model.pkl + metadata.json
# DATA_DIR can be overridden at build time: --build-arg DATA_DIR=path/to/data
ARG DATA_DIR=data
COPY ${DATA_DIR}/ data/
RUN python ml/train.py --data-dir data/ --output ml/artifacts/


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy trained artifacts from trainer stage (no raw data, no training code)
COPY --from=trainer /app/ml/artifacts/ ml/artifacts/

# Copy application source
COPY ml/__init__.py ml/__init__.py
COPY ml/features.py ml/features.py
COPY ml/predict.py  ml/predict.py
COPY api/           api/
COPY frontend/      frontend/
COPY setup.py .
RUN pip install -e . --no-deps

EXPOSE 8000

# Health check (used by Docker + Jenkins smoke test)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--access-log"]
