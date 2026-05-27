"""
api/schemas.py — Pydantic request/response models.
"""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class CoilInput(BaseModel):
    """Single coil with all process parameters."""
    coil_id: Optional[int] = Field(None, description="Unique coil identifier")
    # Process parameters X1-X49 (all optional — missing values are imputed)
    X1: Optional[float]=None; X2: Optional[float]=None; X3: Optional[float]=None
    X4: Optional[float]=None; X5: Optional[float]=None; X6: Optional[float]=None
    X7: Optional[float]=None; X8: Optional[float]=None; X9: Optional[float]=None
    X10: Optional[float]=None; X11: Optional[float]=None; X12: Optional[float]=None
    X13: Optional[float]=None; X14: Optional[float]=None; X15: Optional[float]=None
    X16: Optional[float]=None; X17: Optional[float]=None; X18: Optional[float]=None
    X19: Optional[float]=None; X20: Optional[float]=None; X21: Optional[float]=None
    X22: Optional[float]=None; X23: Optional[float]=None; X24: Optional[float]=None
    X25: Optional[float]=None; X26: Optional[float]=None; X27: Optional[float]=None
    X28: Optional[float]=None; X29: Optional[float]=None; X30: Optional[float]=None
    X31: Optional[float]=None; X32: Optional[float]=None; X33: Optional[float]=None
    X34: Optional[float]=None; X35: Optional[float]=None; X36: Optional[float]=None
    X37: Optional[float]=None; X38: Optional[float]=None; X39: Optional[float]=None
    X40: Optional[float]=None; X41: Optional[float]=None; X42: Optional[float]=None
    X43: Optional[float]=None; X44: Optional[float]=None; X45: Optional[float]=None
    X46: Optional[float]=None; X47: Optional[float]=None; X48: Optional[float]=None
    X49: Optional[float]=None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "coil_id": 654,
                "X1": 854.79, "X2": 501.09, "X3": 414.84,
                "X10": 9.49, "X13": 1312.3, "X36": 378.9
            }
        }
    )


class PredictRequest(BaseModel):
    coils: List[CoilInput] = Field(..., min_length=1, max_length=1000)


class CoilPrediction(BaseModel):
    coil_id: Optional[int]
    prediction: int = Field(..., description="1=Defect, 0=Clean")
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: str = Field(..., description="Critical/High/Medium/Low")


class PredictResponse(BaseModel):
    predictions: List[CoilPrediction]
    n_defects: int
    n_clean: int
    model_version: str
    threshold: float
    inference_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    version: str
    cv_auc: float
    cv_ap: float
    train_recall: float
    train_precision: float
    threshold: float
    n_features_raw: int
    n_features_selected: int
    models: List[str]
    top_features: List[dict]
