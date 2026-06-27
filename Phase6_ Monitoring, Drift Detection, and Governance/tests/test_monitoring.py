"""Unit tests for the Phase 6 monitoring toolkit."""

import numpy as np
import pandas as pd
import pytest

from monitoring.data_validation import validate_batch
from monitoring.drift_detection import numeric_drift, categorical_drift, compute_prediction_drift
from monitoring.audit_log import append_record, read_log, summarize_log


# ------------------------------------------------------------- validation
def _good_claim_batch(n=50):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "billed_amount": rng.uniform(1000, 50000, n),
        "provider_rejection_rate": rng.uniform(0, 1, n),
        "dept_avg_billed": rng.uniform(5000, 30000, n),
        "age": rng.integers(1, 90, n),
        "chronic_flag": rng.integers(0, 2, n),
        "billing_lag": rng.integers(0, 60, n),
        "visit_month": rng.integers(1, 13, n),
        "visit_quarter": rng.integers(1, 5, n),
        "outlier_billed": rng.integers(0, 2, n),
        "length_of_stay_hours": rng.uniform(1, 200, n),
        "visit_frequency": rng.integers(1, 10, n),
        "insurance_provider": rng.choice(["CareOne", "HealthPlus", "MediCareX", "SecureLife"], n),
        "visit_type": rng.choice(["ER", "ICU", "OPD"], n),
        "department": rng.choice(["Cardiology", "ER", "General", "ICU", "Neurology", "Orthopedics"], n),
        "risk_score": rng.choice(["High", "Low", "Medium"], n),
    })


def test_validation_passes_clean_batch():
    report = validate_batch(_good_claim_batch(), model_key="claim_model")
    assert report.passed, report.summary()


def test_validation_flags_out_of_range():
    df = _good_claim_batch()
    df.loc[0, "age"] = 999
    df.loc[1, "provider_rejection_rate"] = 5.0
    report = validate_batch(df, model_key="claim_model")
    assert not report.passed
    checks = {(i.column, i.check) for i in report.issues}
    assert ("age", "range") in checks
    assert ("provider_rejection_rate", "range") in checks


def test_validation_flags_unseen_category():
    df = _good_claim_batch()
    df.loc[0, "department"] = "Oncology"
    report = validate_batch(df, model_key="claim_model")
    assert any(i.check == "unseen_category" and i.column == "department" for i in report.issues)


def test_validation_flags_missing_value():
    df = _good_claim_batch()
    df.loc[0, "billed_amount"] = np.nan
    report = validate_batch(df, model_key="claim_model")
    assert any(i.check == "missing" and i.column == "billed_amount" for i in report.issues)


def test_validation_flags_missing_column():
    df = _good_claim_batch().drop(columns=["age"])
    report = validate_batch(df, model_key="claim_model")
    assert any(i.check == "missing_column" and i.column == "age" for i in report.issues)


# ----------------------------------------------------------------- drift
def test_no_drift_same_distribution():
    rng = np.random.default_rng(1)
    ref = pd.Series(rng.normal(0, 1, 5000))
    cur = pd.Series(rng.normal(0, 1, 5000))
    res = numeric_drift("x", ref, cur)
    assert res.psi < 0.1
    assert not res.drifted


def test_detects_numeric_shift():
    rng = np.random.default_rng(2)
    ref = pd.Series(rng.normal(0, 1, 5000))
    cur = pd.Series(rng.normal(3, 1, 5000))  # large mean shift
    res = numeric_drift("x", ref, cur)
    assert res.psi >= 0.25
    assert res.drifted


def test_detects_categorical_shift():
    ref = pd.Series(["A"] * 800 + ["B"] * 200)
    cur = pd.Series(["A"] * 200 + ["B"] * 800)
    res = categorical_drift("c", ref, cur)
    assert res.psi >= 0.25
    assert res.drifted


def test_prediction_drift_stable_when_same():
    ref = pd.Series(["Paid"] * 600 + ["Pending"] * 200 + ["Rejected"] * 200)
    cur = ref.copy()
    res = compute_prediction_drift(ref, cur)
    assert res.psi < 0.1
    assert not res.drifted


# ------------------------------------------------------------- audit log
def test_audit_append_and_summarize(tmp_path):
    log = tmp_path / "audit.log"
    append_record(log, model_name="risk_model_v2", model_version="v2.0.0",
                  prediction="High", probabilities={"High": 0.7, "Low": 0.2, "Medium": 0.1},
                  features={"age": 67}, extra={"latency_ms": 10.0})
    append_record(log, model_name="risk_model_v2", model_version="v2.0.0",
                  prediction="Low", probabilities={"High": 0.1, "Low": 0.8, "Medium": 0.1},
                  features={"age": 30}, extra={"latency_ms": 20.0})
    df = read_log(log)
    assert len(df) == 2
    summary = summarize_log(log)
    assert summary["total_predictions"] == 2
    assert summary["duplicate_request_ids"] == 0
    assert summary["latency_ms"]["max"] == 20.0


def test_audit_feature_hash_is_stable(tmp_path):
    log = tmp_path / "a.log"
    r1 = append_record(log, model_name="m", model_version="v1", prediction="X",
                       probabilities={"X": 1.0}, features={"a": 1, "b": 2})
    r2 = append_record(log, model_name="m", model_version="v1", prediction="X",
                       probabilities={"X": 1.0}, features={"b": 2, "a": 1})
    assert r1["feature_hash"] == r2["feature_hash"]  # order-independent
    assert r1["request_id"] != r2["request_id"]
