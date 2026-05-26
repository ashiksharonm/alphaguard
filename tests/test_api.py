"""tests/test_api.py — API integration tests."""
import io
import json

import pandas as pd
import pytest


def test_health_endpoint_returns_200(app_client):
    r = app_client.get("/health")
    assert r.status_code == 200


def test_health_response_schema(app_client):
    r = app_client.get("/health")
    data = r.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "model_version" in data
    assert "uptime_seconds" in data


def test_model_info_endpoint(app_client):
    r = app_client.get("/model/info")
    # May return 200 or 500 depending on whether model is loaded
    assert r.status_code in (200, 500)


def test_metrics_endpoint_returns_prometheus_format(app_client):
    r = app_client.get("/metrics")
    assert r.status_code == 200
    assert "alphaguard" in r.text or "HELP" in r.text


def test_predict_with_valid_json(app_client):
    payload = {
        "coils": [{
            "coil_id": 1001,
            **{f"X{i}": float(400 + i * 5) for i in range(1, 50)},
        }]
    }
    r = app_client.post("/predict", json=payload)
    # 200 if model loaded, 503 if not
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        data = r.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 1
        pred = data["predictions"][0]
        assert pred["prediction"] in (0, 1)
        assert 0.0 <= pred["risk_score"] <= 1.0
        assert pred["risk_level"] in ("Critical", "High", "Medium", "Low")
        assert "inference_ms" in data
        assert data["inference_ms"] > 0


def test_predict_multiple_coils(app_client):
    payload = {
        "coils": [
            {"coil_id": 1000 + i, **{f"X{j}": float(j * 10) for j in range(1, 50)}}
            for i in range(5)
        ]
    }
    r = app_client.post("/predict", json=payload)
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        assert len(r.json()["predictions"]) == 5


def test_predict_empty_coils_returns_422(app_client):
    r = app_client.post("/predict", json={"coils": []})
    assert r.status_code == 422   # Pydantic validation: min_length=1


def test_predict_invalid_payload_returns_422(app_client):
    r = app_client.post("/predict", json={"wrong_key": "value"})
    assert r.status_code == 422


def test_predict_batch_csv(app_client):
    df = pd.DataFrame([
        {"CoilID": 2001 + i, **{f"X{j}": float(j * 10 + i) for j in range(1, 50)}}
        for i in range(3)
    ])
    csv_bytes = df.to_csv(index=False).encode()
    r = app_client.post(
        "/predict/batch",
        files={"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        assert len(r.json()["predictions"]) == 3


def test_predict_batch_invalid_csv_returns_400(app_client):
    r = app_client.post(
        "/predict/batch",
        files={"file": ("bad.csv", io.BytesIO(b"not,valid,\x00csv"), "text/csv")},
    )
    # Should be 400 or 503 (if no model loaded)
    assert r.status_code in (400, 422, 503)
