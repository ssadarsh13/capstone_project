"""Request and response schemas with strict validation.

Consumers send *raw business inputs* only. All derived/interaction features and
label encodings are reproduced server-side (see model_service.py) so callers can
never introduce train/serve skew.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict

from pydantic import BaseModel, Field, field_validator


# --- Controlled vocabularies (must match feature_schema.json categorical_encodings) ---
class VisitType(str, Enum):
    ER = "ER"
    ICU = "ICU"
    OPD = "OPD"


class Department(str, Enum):
    Cardiology = "Cardiology"
    ER = "ER"
    General = "General"
    ICU = "ICU"
    Neurology = "Neurology"
    Orthopedics = "Orthopedics"


class InsuranceProvider(str, Enum):
    CareOne = "CareOne"
    HealthPlus = "HealthPlus"
    MediCareX = "MediCareX"
    SecureLife = "SecureLife"


class RiskScore(str, Enum):
    High = "High"
    Low = "Low"
    Medium = "Medium"


# --- Model A: Visit Risk ---
class RiskPredictionRequest(BaseModel):
    """Raw inputs available at visit-admission time."""

    model_config = {"extra": "forbid"}

    age: int = Field(..., ge=0, le=120, examples=[67])
    chronic_flag: int = Field(..., ge=0, le=1, examples=[1])
    length_of_stay_hours: float = Field(..., ge=0, le=2000, examples=[72.5])
    visit_month: int = Field(..., ge=1, le=12, examples=[11])
    visit_quarter: int = Field(..., ge=1, le=4, examples=[4])
    is_weekend: int = Field(..., ge=0, le=1, examples=[0])
    days_since_registration: int = Field(..., ge=0, le=20000, examples=[420])
    visit_frequency: int = Field(..., ge=1, le=1000, examples=[5])
    avg_los_per_patient: float = Field(..., ge=0, le=2000, examples=[40.2])
    outlier_los: int = Field(..., ge=0, le=1, examples=[0])
    dept_avg_billed: float = Field(..., ge=0, examples=[18500.0])
    visit_type: VisitType = Field(..., examples=[VisitType.ICU])
    department: Department = Field(..., examples=[Department.Cardiology])

    @field_validator("visit_quarter")
    @classmethod
    def _quarter_matches_month(cls, v: int, info) -> int:
        month = info.data.get("visit_month")
        if month is not None and (month - 1) // 3 + 1 != v:
            raise ValueError(
                f"visit_quarter={v} is inconsistent with visit_month={month}"
            )
        return v


# --- Model B: Claim Outcome ---
class ClaimPredictionRequest(BaseModel):
    """Raw inputs available at claim-submission time."""

    model_config = {"extra": "forbid"}

    billed_amount: float = Field(..., gt=0, examples=[42000.0])
    provider_rejection_rate: float = Field(..., ge=0, le=1, examples=[0.18])
    dept_avg_billed: float = Field(..., ge=0, examples=[18500.0])
    age: int = Field(..., ge=0, le=120, examples=[67])
    chronic_flag: int = Field(..., ge=0, le=1, examples=[1])
    billing_lag: int = Field(..., ge=0, le=365, examples=[5])
    visit_month: int = Field(..., ge=1, le=12, examples=[11])
    visit_quarter: int = Field(..., ge=1, le=4, examples=[4])
    outlier_billed: int = Field(..., ge=0, le=1, examples=[0])
    length_of_stay_hours: float = Field(..., ge=0, le=2000, examples=[72.5])
    visit_frequency: int = Field(..., ge=1, le=1000, examples=[5])
    insurance_provider: InsuranceProvider = Field(..., examples=[InsuranceProvider.MediCareX])
    visit_type: VisitType = Field(..., examples=[VisitType.ICU])
    department: Department = Field(..., examples=[Department.Cardiology])
    risk_score: RiskScore = Field(..., examples=[RiskScore.High])

    @field_validator("visit_quarter")
    @classmethod
    def _quarter_matches_month(cls, v: int, info) -> int:
        month = info.data.get("visit_month")
        if month is not None and (month - 1) // 3 + 1 != v:
            raise ValueError(
                f"visit_quarter={v} is inconsistent with visit_month={month}"
            )
        return v


# --- Shared response envelope ---
class PredictionResponse(BaseModel):
    request_id: str = Field(..., description="Unique id for this prediction (audit trail).")
    model_name: str = Field(..., examples=["risk_model_v2"])
    model_version: str = Field(..., examples=["v2.0.0"])
    prediction: str = Field(..., description="Predicted class label.")
    probabilities: Dict[str, float] = Field(..., description="Per-class probabilities.")
    threshold_applied: float = Field(
        ..., description="Decision threshold used for the business-critical minority class."
    )
    feature_hash: str = Field(..., description="SHA-256 of the engineered feature vector.")
    timestamp: str = Field(..., description="UTC ISO-8601 prediction time.")
    latency_ms: float = Field(..., description="Server-side inference latency in milliseconds.")


# --- Health / metadata ---
class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    service: str
    model_version: str
    models_loaded: Dict[str, bool]
    uptime_seconds: float


class ModelMetadata(BaseModel):
    model_name: str
    target: str
    target_classes: list[str]
    n_features: int
    threshold: float
    threshold_class: str
