"""API integration tests using FastAPI's TestClient (in-process, no server)."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    # `with` triggers the lifespan handler, which loads the model artifacts.
    with TestClient(app) as c:
        yield c


VALID_RISK = {
    "age": 67,
    "chronic_flag": 1,
    "length_of_stay_hours": 72.5,
    "visit_month": 11,
    "visit_quarter": 4,
    "is_weekend": 0,
    "days_since_registration": 420,
    "visit_frequency": 5,
    "avg_los_per_patient": 40.2,
    "outlier_los": 0,
    "dept_avg_billed": 18500.0,
    "visit_type": "ICU",
    "department": "Cardiology",
}

VALID_CLAIM = {
    "billed_amount": 42000.0,
    "provider_rejection_rate": 0.18,
    "dept_avg_billed": 18500.0,
    "age": 67,
    "chronic_flag": 1,
    "billing_lag": 5,
    "visit_month": 11,
    "visit_quarter": 4,
    "outlier_billed": 0,
    "length_of_stay_hours": 72.5,
    "visit_frequency": 5,
    "insurance_provider": "MediCareX",
    "visit_type": "ICU",
    "department": "Cardiology",
    "risk_score": "High",
}


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] == {"risk_model": True, "claim_model": True}


def test_metadata(client):
    r = client.get("/metadata")
    assert r.status_code == 200
    body = r.json()
    assert body["risk_model"]["target_classes"] == ["High", "Low", "Medium"]
    assert body["claim_model"]["target_classes"] == ["Paid", "Pending", "Rejected"]
    assert body["risk_model"]["n_features"] == 17
    assert body["claim_model"]["n_features"] == 20


def test_predict_risk_valid(client):
    r = client.post("/predict/risk", json=VALID_RISK)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in ["High", "Low", "Medium"]
    assert set(body["probabilities"]) == {"High", "Low", "Medium"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-3
    assert len(body["feature_hash"]) == 64
    assert body["model_name"] == "risk_model_v2"
    assert "request_id" in body


def test_predict_claim_valid(client):
    r = client.post("/predict/claim", json=VALID_CLAIM)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in ["Paid", "Pending", "Rejected"]
    assert set(body["probabilities"]) == {"Paid", "Pending", "Rejected"}
    assert len(body["feature_hash"]) == 64
    assert body["model_name"] == "claim_model_v2"


def test_feature_hash_is_deterministic(client):
    a = client.post("/predict/risk", json=VALID_RISK).json()
    b = client.post("/predict/risk", json=VALID_RISK).json()
    assert a["feature_hash"] == b["feature_hash"]
    assert a["request_id"] != b["request_id"]  # ids are unique per request


def test_reject_unknown_category(client):
    bad = dict(VALID_RISK, department="Oncology")
    r = client.post("/predict/risk", json=bad)
    assert r.status_code == 422


def test_reject_out_of_range(client):
    bad = dict(VALID_RISK, age=999)
    r = client.post("/predict/risk", json=bad)
    assert r.status_code == 422


def test_reject_inconsistent_quarter(client):
    bad = dict(VALID_RISK, visit_month=1, visit_quarter=4)
    r = client.post("/predict/risk", json=bad)
    assert r.status_code == 422


def test_reject_extra_field(client):
    bad = dict(VALID_RISK, hospital_id="H123")
    r = client.post("/predict/risk", json=bad)
    assert r.status_code == 422
