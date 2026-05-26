"""
ml/predict.py — AlphaGuard Inference Module
Loads a trained artifact and generates predictions.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from ml.features import engineer_features


class AlphaGuardPredictor:
    """Load a trained model artifact and run inference."""

    def __init__(self, artifact: dict) -> None:
        self._imputer  = artifact["imputer"]
        self._imp2     = artifact["imp2"]
        self._qt       = artifact["qt"]
        self._models   = artifact["models"]
        self._feat_idx = artifact["feat_idx"]
        self._threshold = artifact["threshold"]
        self._base_cols = artifact["base_cols"]
        self._version   = artifact.get("version", "unknown")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AlphaGuardPredictor":
        """Load from a .pkl file."""
        with open(path, "rb") as f:
            artifact = pickle.load(f)
        return cls(artifact)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run inference on a DataFrame of raw process parameters.

        Args:
            df: DataFrame with columns [CoilID?, X1..X49]. CoilID is optional.

        Returns:
            DataFrame with columns: coil_id, prediction, risk_score, risk_level.
        """
        coil_ids = df["CoilID"].values if "CoilID" in df.columns else np.arange(len(df))
        X = df[[c for c in self._base_cols if c in df.columns]]

        # Add missing columns as NaN
        for col in self._base_cols:
            if col not in X.columns:
                X[col] = np.nan
        X = X[self._base_cols]

        # Preprocess
        X_imp = pd.DataFrame(self._imputer.transform(X), columns=self._base_cols)
        X_fe  = engineer_features(X_imp)
        Xarr  = self._imp2.transform(X_fe)
        Xqt   = self._qt.transform(Xarr)

        Xsel   = Xarr[:, self._feat_idx]
        Xqt_sel = Xqt[:, self._feat_idx]

        # Ensemble score
        probs = np.mean([
            m.predict_proba(Xqt_sel if name == "lr" else Xsel)[:, 1]
            for name, m in self._models.items()
        ], axis=0)

        preds = (probs >= self._threshold).astype(int)

        return pd.DataFrame({
            "coil_id":    coil_ids,
            "prediction": preds,
            "risk_score": np.round(probs, 4),
            "risk_level":  [self._risk_level(p) for p in probs],
        })

    @staticmethod
    def _risk_level(prob: float) -> str:
        if prob > 0.65: return "Critical"
        if prob > 0.40: return "High"
        if prob > 0.20: return "Medium"
        return "Low"

    @property
    def version(self) -> str:
        return self._version

    @property
    def threshold(self) -> float:
        return self._threshold
