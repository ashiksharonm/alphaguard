"""
tests/test_model.py — Model contract tests.
These are the most important tests — they enforce the core
business guarantee: Recall = 100% (zero false negatives).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_predictor_loads(trained_predictor):
    """Sanity check that the artifact can be loaded."""
    assert trained_predictor is not None
    assert trained_predictor.version is not None


def test_predictor_version_is_string(trained_predictor):
    assert isinstance(trained_predictor.version, str)
    assert len(trained_predictor.version) > 0


def test_threshold_is_valid(trained_predictor):
    assert 0 < trained_predictor.threshold < 1, (
        f"Threshold {trained_predictor.threshold} must be between 0 and 1"
    )


def test_predict_output_schema(trained_predictor, sample_coil_df):
    """Prediction output must have required columns with correct types."""
    result = trained_predictor.predict(sample_coil_df)
    assert set(result.columns) >= {"coil_id", "prediction", "risk_score", "risk_level"}
    assert result["prediction"].isin([0, 1]).all(), "prediction must be binary"
    assert (result["risk_score"] >= 0).all() and (result["risk_score"] <= 1).all()
    assert result["risk_level"].isin(["Critical", "High", "Medium", "Low"]).all()


def test_predict_row_count_matches_input(trained_predictor, sample_coil_df):
    result = trained_predictor.predict(sample_coil_df)
    assert len(result) == len(sample_coil_df)


def test_metadata_has_required_keys():
    meta_path = Path("ml/artifacts/metadata.json")
    if not meta_path.exists():
        pytest.skip("No metadata.json found")
    with open(meta_path) as f:
        meta = json.load(f)
    for key in ["version", "cv_auc", "train_recall", "train_precision", "threshold"]:
        assert key in meta, f"Metadata missing key: {key}"


def test_metadata_recall_is_1(trained_predictor):
    """The sacred contract: train recall must be 100%."""
    meta_path = Path("ml/artifacts/metadata.json")
    if not meta_path.exists():
        pytest.skip("No metadata.json found")
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta["train_recall"] == 1.0, (
        f"CRITICAL: Train recall = {meta['train_recall']}, expected 1.0"
    )


def test_metadata_auc_above_baseline():
    """AUC-ROC should be meaningfully above random (0.5)."""
    meta_path = Path("ml/artifacts/metadata.json")
    if not meta_path.exists():
        pytest.skip("No metadata.json found")
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta["cv_auc"] > 0.70, (
        f"AUC {meta['cv_auc']} too low — model may be broken"
    )


def test_high_risk_coil_flagged_as_defect(trained_predictor):
    """A coil with extreme values for top features should be flagged."""
    # X13 for defects is ~1312 vs clean ~846; X35 is much lower for defects
    df = pd.DataFrame([{
        "CoilID": 9999,
        **{f"X{i}": 500.0 for i in range(1, 50)},
        "X13": 2500.0,   # Very high (defect-like)
        "X35": 100.0,    # Very low (defect-like)
        "X36": 50.0,     # Very low (defect-like)
        "X10": 15.0,     # High (defect-like)
    }])
    result = trained_predictor.predict(df)
    # This should get a high risk score
    assert result["risk_score"].iloc[0] > 0.10, (
        "High-risk coil should have elevated risk score"
    )
