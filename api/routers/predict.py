"""
api/routers/predict.py — Prediction endpoints.
"""
from __future__ import annotations
import io
import time
from typing import List

import pandas as pd
from fastapi import APIRouter, HTTPException, Request, UploadFile, File

from api.schemas import PredictRequest, PredictResponse, CoilPrediction
from api.middleware import PREDICTIONS_TOTAL, DEFECTS_FLAGGED

router = APIRouter()


@router.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(body: PredictRequest, request: Request):
    """
    Predict Alpha defect for one or more coils.

    Send process parameters for each coil; receive risk score,
    risk level, and binary prediction (1=Defect, 0=Clean).
    """
    predictor = request.app.state.predictor
    metadata  = request.app.state.metadata
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    df = pd.DataFrame([c.model_dump() for c in body.coils])
    df = df.rename(columns={"coil_id": "CoilID"})

    result = predictor.predict(df)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Update Prometheus counters
    PREDICTIONS_TOTAL.inc(len(result))
    DEFECTS_FLAGGED.inc(int(result["prediction"].sum()))

    predictions = [
        CoilPrediction(
            coil_id=int(row.coil_id) if row.coil_id is not None else None,
            prediction=int(row.prediction),
            risk_score=float(row.risk_score),
            risk_level=row.risk_level,
        )
        for row in result.itertuples()
    ]

    return PredictResponse(
        predictions=predictions,
        n_defects=int(result["prediction"].sum()),
        n_clean=int((result["prediction"] == 0).sum()),
        model_version=predictor.version,
        threshold=predictor.threshold,
        inference_ms=round(elapsed_ms, 2),
    )


@router.post("/predict/batch", response_model=PredictResponse, tags=["Prediction"])
async def predict_batch(request: Request, file: UploadFile = File(...)):
    """
    Predict from an uploaded CSV file.
    CSV must have columns: CoilID (optional), X1..X49.
    """
    predictor = request.app.state.predictor
    metadata  = request.app.state.metadata
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")

    start  = time.perf_counter()
    result = predictor.predict(df)
    elapsed_ms = (time.perf_counter() - start) * 1000

    PREDICTIONS_TOTAL.inc(len(result))
    DEFECTS_FLAGGED.inc(int(result["prediction"].sum()))

    predictions = [
        CoilPrediction(
            coil_id=int(row.coil_id) if row.coil_id is not None else None,
            prediction=int(row.prediction),
            risk_score=float(row.risk_score),
            risk_level=row.risk_level,
        )
        for row in result.itertuples()
    ]

    return PredictResponse(
        predictions=predictions,
        n_defects=int(result["prediction"].sum()),
        n_clean=int((result["prediction"] == 0).sum()),
        model_version=predictor.version,
        threshold=predictor.threshold,
        inference_ms=round(elapsed_ms, 2),
    )
