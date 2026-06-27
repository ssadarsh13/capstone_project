"""Paths, validation bounds, and drift thresholds for the monitoring toolkit."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

# Repo layout: this file lives in
#   <repo>/Phase 6 — .../monitoring/config.py
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = Path(
    os.getenv("MONITOR_DATA_PATH", REPO_ROOT / "Phase2_EDA" / "model_table.csv")
)
MODELS_DIR = Path(
    os.getenv("MONITOR_MODELS_DIR", REPO_ROOT / "Phase3_Modeling" / "models")
)
SCHEMA_PATH = Path(
    os.getenv("MONITOR_SCHEMA_PATH", REPO_ROOT / "Phase3_Modeling" / "feature_schema.json")
)

# Numeric validation bounds — mirror the Phase 5 API contract (schemas.py).
NUMERIC_BOUNDS: Dict[str, tuple[float, float]] = {
    "age": (0, 120),
    "chronic_flag": (0, 1),
    "length_of_stay_hours": (0, 2000),
    "visit_month": (1, 12),
    "visit_quarter": (1, 4),
    "is_weekend": (0, 1),
    "days_since_registration": (0, 20000),
    "visit_frequency": (1, 1000),
    "avg_los_per_patient": (0, 2000),
    "outlier_los": (0, 1),
    "outlier_billed": (0, 1),
    "dept_avg_billed": (0, 10_000_000),
    "billed_amount": (0, 10_000_000),
    "provider_rejection_rate": (0, 1),
    "billing_lag": (0, 365),
}

# Drift thresholds (Population Stability Index convention).
PSI_NO_DRIFT = 0.10        # < 0.10  : stable
PSI_MODERATE_DRIFT = 0.25  # 0.10–0.25: moderate; >= 0.25: significant
KS_PVALUE_ALPHA = 0.05     # KS / chi-square significance level

# Fraction of data used as the reference (training-era) window.
REFERENCE_FRACTION = 0.80


@lru_cache
def load_schema() -> Dict[str, Any]:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def psi_band(psi: float) -> str:
    if psi < PSI_NO_DRIFT:
        return "stable"
    if psi < PSI_MODERATE_DRIFT:
        return "moderate"
    return "significant"
