"""
ml/features.py
Feature engineering module for AlphaGuard Alpha Defect Detection.
All transformations are pure functions — stateless and reproducible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List


STAGE_RANGES = [(1, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
TOP_SIGNAL_FEATURES = ["X35", "X13", "X36", "X34", "X10", "X30", "X31", "X32", "X29", "X37"]
TOP_RATIOS = [
    ("X13", "X35"), ("X13", "X36"), ("X10", "X13"),
    ("X34", "X35"), ("X30", "X31"), ("X31", "X32"), ("X10", "X30"),
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand raw process parameters into a rich feature set.

    Adds:
        - Global statistics (mean, std, max, min, range, CV, IQR, skew, kurtosis)
        - Stage-wise aggregates per rolling stage (X1-X9, X10-X19, ...)
        - Inter-stage deltas (process transition signals)
        - Squared and log transforms of top-signal features
        - Pairwise ratios among top-signal features

    Args:
        df: DataFrame with raw Xn features (no CoilID/Y columns).

    Returns:
        DataFrame with original + engineered features.
    """
    df = df.copy()
    vals = df.values.astype(float)

    # ── Global statistics ─────────────────────────────────────────────────────
    df["g_mean"]  = np.nanmean(vals, axis=1)
    df["g_std"]   = np.nanstd(vals, axis=1)
    df["g_max"]   = np.nanmax(vals, axis=1)
    df["g_min"]   = np.nanmin(vals, axis=1)
    df["g_range"] = df["g_max"] - df["g_min"]
    df["g_cv"]    = df["g_std"] / (np.abs(df["g_mean"]) + 1e-9)
    df["g_iqr"]   = np.percentile(vals, 75, axis=1) - np.percentile(vals, 25, axis=1)
    df["g_skew"]  = pd.DataFrame(vals).skew(axis=1).values
    df["g_kurt"]  = pd.DataFrame(vals).kurt(axis=1).values

    # ── Stage-wise aggregates ─────────────────────────────────────────────────
    for stage_num, (lo, hi) in enumerate(STAGE_RANGES, start=1):
        cols = [f"X{c}" for c in range(lo, hi) if f"X{c}" in df.columns]
        if cols:
            s = df[cols]
            df[f"s{stage_num}_mean"]  = s.mean(axis=1)
            df[f"s{stage_num}_std"]   = s.std(axis=1)
            df[f"s{stage_num}_max"]   = s.max(axis=1)
            df[f"s{stage_num}_min"]   = s.min(axis=1)
            df[f"s{stage_num}_range"] = df[f"s{stage_num}_max"] - df[f"s{stage_num}_min"]

    # ── Inter-stage deltas ────────────────────────────────────────────────────
    for i in range(1, 5):
        a, b = f"s{i}_mean", f"s{i+1}_mean"
        if a in df.columns and b in df.columns:
            df[f"d_{i}_{i+1}"] = df[a] - df[b]

    # ── Top-signal transforms ─────────────────────────────────────────────────
    for feat in TOP_SIGNAL_FEATURES:
        if feat in df.columns:
            df[f"{feat}_sq"]  = df[feat] ** 2
            df[f"{feat}_log"] = np.log1p(np.abs(df[feat]))

    # ── Pairwise ratios ───────────────────────────────────────────────────────
    for a, b in TOP_RATIOS:
        if a in df.columns and b in df.columns:
            df[f"r_{a}_{b}"] = df[a] / (np.abs(df[b]) + 1e-9)

    return df


def get_feature_names(base_cols: List[str]) -> List[str]:
    """Return the expected output column names after feature engineering."""
    dummy = pd.DataFrame(0.0, index=[0], columns=base_cols)
    return list(engineer_features(dummy).columns)
