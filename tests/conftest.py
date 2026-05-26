"""tests/conftest.py — Shared fixtures for the test suite."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def sample_coil_df():
    """10 realistic synthetic coil rows for unit tests."""
    rng = np.random.default_rng(42)
    n = 10
    df = pd.DataFrame({
        "CoilID": range(1000, 1000 + n),
        **{f"X{i}": rng.normal(loc=500 + i * 20, scale=50, size=n) for i in range(1, 50)},
    })
    return df


@pytest.fixture(scope="session")
def sample_X(sample_coil_df):
    return sample_coil_df.drop(columns=["CoilID"])


@pytest.fixture(scope="session")
def trained_predictor():
    """Load the real predictor if artifact exists, else skip."""
    pkl = Path("ml/artifacts/model.pkl")
    if not pkl.exists():
        pytest.skip("No model artifact found — run `python ml/train.py` first.")
    from ml.predict import AlphaGuardPredictor
    return AlphaGuardPredictor.load(pkl)


@pytest.fixture(scope="session")
def app_client():
    """Async test client for FastAPI."""
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)
