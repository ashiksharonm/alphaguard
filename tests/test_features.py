"""tests/test_features.py — Unit tests for feature engineering."""
import numpy as np
import pandas as pd
import pytest
from ml.features import engineer_features, TOP_SIGNAL_FEATURES


@pytest.fixture
def base_df():
    """Minimal DataFrame with X1-X49."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.normal(500, 50, size=(5, 49)),
        columns=[f"X{i}" for i in range(1, 50)],
    )


def test_output_has_more_columns_than_input(base_df):
    result = engineer_features(base_df)
    assert result.shape[1] > base_df.shape[1], "Should add new features"


def test_global_stats_present(base_df):
    result = engineer_features(base_df)
    for col in ["g_mean", "g_std", "g_max", "g_min", "g_range", "g_cv", "g_iqr", "g_skew", "g_kurt"]:
        assert col in result.columns, f"Missing global stat: {col}"


def test_stage_features_present(base_df):
    result = engineer_features(base_df)
    for stage in range(1, 6):
        for stat in ["mean", "std", "max", "min", "range"]:
            col = f"s{stage}_{stat}"
            assert col in result.columns, f"Missing stage feature: {col}"


def test_top_signal_transforms_present(base_df):
    result = engineer_features(base_df)
    for feat in TOP_SIGNAL_FEATURES:
        assert f"{feat}_sq" in result.columns
        assert f"{feat}_log" in result.columns


def test_no_infinite_values(base_df):
    result = engineer_features(base_df)
    assert not np.isinf(result.values).any(), "Should not contain Inf values"


def test_row_count_preserved(base_df):
    result = engineer_features(base_df)
    assert result.shape[0] == base_df.shape[0], "Row count must be preserved"


def test_original_columns_preserved(base_df):
    result = engineer_features(base_df)
    for col in base_df.columns:
        assert col in result.columns, f"Original column {col} was dropped"


def test_handles_missing_values():
    """Feature engineering must not crash with NaN inputs."""
    df = pd.DataFrame(
        [[np.nan if i % 5 == 0 else float(i * 10) for i in range(49)]],
        columns=[f"X{i}" for i in range(1, 50)],
    )
    result = engineer_features(df)
    assert result.shape[0] == 1
