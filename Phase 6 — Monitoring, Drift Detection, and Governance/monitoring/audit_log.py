"""Audit-log utilities: append structured prediction records and summarize them.

The log is JSON-lines and append-only — the same format the Phase 5 API emits to
stdout/CloudWatch. This module lets batch/offline jobs write to the same trail
and lets governance reviews summarize it (volume, model versions, latency,
class mix) for a time window.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


def feature_hash(model_name: str, features: Dict[str, Any]) -> str:
    """SHA-256 of the engineered feature vector (matches Phase 5 hashing)."""
    payload = json.dumps(
        {"model": model_name, "features": features},
        sort_keys=True, separators=(",", ":"), default=float,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def append_record(
    log_path: Path,
    *,
    model_name: str,
    model_version: str,
    prediction: str,
    probabilities: Dict[str, float],
    features: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Append one audit record and return it."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "event": "prediction",
        "request_id": request_id or str(uuid.uuid4()),
        "model_name": model_name,
        "model_version": model_version,
        "prediction": prediction,
        "probabilities": probabilities,
        "feature_hash": feature_hash(model_name, features) if features is not None else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        record.update(extra)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")
    return record


def read_log(log_path: Path) -> pd.DataFrame:
    """Load a JSON-lines audit log into a DataFrame (prediction events only)."""
    log_path = Path(log_path)
    if not log_path.exists():
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    with open(log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("event") == "prediction":
                rows.append(obj)
    df = pd.DataFrame(rows)
    if not df.empty and "timestamp" in df:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return df


def summarize_log(log_path: Path) -> Dict[str, Any]:
    """Governance summary of the audit trail."""
    df = read_log(log_path)
    if df.empty:
        return {"total_predictions": 0}

    summary: Dict[str, Any] = {
        "total_predictions": int(len(df)),
        "time_range": {
            "start": str(df["timestamp"].min()) if "timestamp" in df else None,
            "end": str(df["timestamp"].max()) if "timestamp" in df else None,
        },
        "by_model_version": {
            f"{m}:{v}": int(c)
            for (m, v), c in df.groupby(["model_name", "model_version"]).size().items()
        }
        if {"model_name", "model_version"}.issubset(df.columns)
        else {},
        "prediction_distribution": {},
    }
    if {"model_name", "prediction"}.issubset(df.columns):
        for model, grp in df.groupby("model_name"):
            summary["prediction_distribution"][model] = {
                str(k): int(v) for k, v in grp["prediction"].value_counts().items()
            }
    if "latency_ms" in df:
        lat = pd.to_numeric(df["latency_ms"], errors="coerce").dropna()
        if not lat.empty:
            summary["latency_ms"] = {
                "mean": round(float(lat.mean()), 3),
                "p50": round(float(lat.quantile(0.5)), 3),
                "p95": round(float(lat.quantile(0.95)), 3),
                "max": round(float(lat.max()), 3),
            }
    # Integrity check: duplicate request_ids should never happen.
    if "request_id" in df:
        dupes = int(df["request_id"].duplicated().sum())
        summary["duplicate_request_ids"] = dupes
    return summary
