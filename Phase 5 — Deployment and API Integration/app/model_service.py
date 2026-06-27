"""Model loading, server-side feature engineering, and prediction.

Feature engineering here is a faithful reproduction of Phase 3
(`02_risk_model.ipynb` / `03_claim_model.ipynb`). Categorical encoding uses the
index into the alphabetically-sorted class list stored in feature_schema.json,
which is exactly what scikit-learn's LabelEncoder produced at training time —
so no encoder object needs to be unpickled for the inputs.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from .config import get_settings


class ModelBundle:
    """A trained pipeline plus its schema, threshold, and encoders."""

    def __init__(
        self,
        name: str,
        model: Any,
        label_encoder: Any,
        schema: Dict[str, Any],
        threshold: float,
        threshold_idx: int,
    ) -> None:
        self.name = name
        self.model = model
        self.label_encoder = label_encoder
        self.schema = schema
        self.threshold = threshold
        self.threshold_idx = threshold_idx
        self.all_features: List[str] = schema["all_features"]
        self.classes: List[str] = list(label_encoder.classes_)
        self.threshold_class: str = self.classes[threshold_idx]
        self.encodings: Dict[str, List[str]] = schema["categorical_encodings"]

    def encode_categorical(self, column: str, value: str) -> int:
        """Reproduce LabelEncoder: index into the sorted class list."""
        classes = self.encodings[column]
        if value not in classes:
            raise ValueError(
                f"Unknown category '{value}' for '{column}'. Allowed: {classes}"
            )
        return classes.index(value)


class PredictionService:
    """Loads both model bundles and serves predictions."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._schema: Dict[str, Any] = {}
        self.risk: ModelBundle | None = None
        self.claim: ModelBundle | None = None

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        s = self._settings
        with open(s.schema_path, "r", encoding="utf-8") as fh:
            self._schema = json.load(fh)

        risk_thr = joblib.load(s.risk_threshold_path)
        self.risk = ModelBundle(
            name=self._schema["risk_model"]["model"],
            model=joblib.load(s.risk_model_path),
            label_encoder=joblib.load(s.risk_label_encoder_path),
            schema=self._schema["risk_model"],
            threshold=float(risk_thr["threshold"]),
            threshold_idx=int(risk_thr["hi_idx"]),
        )

        claim_thr = joblib.load(s.claim_threshold_path)
        self.claim = ModelBundle(
            name=self._schema["claim_model"]["model"],
            model=joblib.load(s.claim_model_path),
            label_encoder=joblib.load(s.claim_label_encoder_path),
            schema=self._schema["claim_model"],
            threshold=float(claim_thr["threshold"]),
            threshold_idx=int(claim_thr["rej_idx"]),
        )

    @property
    def ready(self) -> bool:
        return self.risk is not None and self.claim is not None

    # -------------------------------------------------- feature engineering
    @staticmethod
    def _engineer_risk(p: Dict[str, Any], bundle: ModelBundle) -> Dict[str, float]:
        feats: Dict[str, float] = {
            "age": p["age"],
            "chronic_flag": p["chronic_flag"],
            "length_of_stay_hours": p["length_of_stay_hours"],
            "visit_month": p["visit_month"],
            "visit_quarter": p["visit_quarter"],
            "is_weekend": p["is_weekend"],
            "days_since_registration": p["days_since_registration"],
            "visit_frequency": p["visit_frequency"],
            "avg_los_per_patient": p["avg_los_per_patient"],
            "outlier_los": p["outlier_los"],
            "dept_avg_billed": p["dept_avg_billed"],
            # interaction features (mirror Phase 3)
            "los_x_chronic": p["length_of_stay_hours"] * p["chronic_flag"],
            "age_x_frequency": p["age"] * p["visit_frequency"],
            "dept_los_ratio": round(
                p["length_of_stay_hours"] / (p["dept_avg_billed"] + 1), 4
            ),
            "age_x_chronic": p["age"] * p["chronic_flag"],
            # encoded categoricals
            "visit_type_enc": bundle.encode_categorical("visit_type", p["visit_type"]),
            "department_enc": bundle.encode_categorical("department", p["department"]),
        }
        return feats

    @staticmethod
    def _engineer_claim(p: Dict[str, Any], bundle: ModelBundle) -> Dict[str, float]:
        los = p["length_of_stay_hours"]
        billed = p["billed_amount"]
        feats: Dict[str, float] = {
            "billed_amount": billed,
            "billed_per_hour": round(billed / los, 2) if los > 0 else 0.0,
            "provider_rejection_rate": p["provider_rejection_rate"],
            "dept_avg_billed": p["dept_avg_billed"],
            "age": p["age"],
            "chronic_flag": p["chronic_flag"],
            "billing_lag": p["billing_lag"],
            "visit_month": p["visit_month"],
            "visit_quarter": p["visit_quarter"],
            "outlier_billed": p["outlier_billed"],
            "length_of_stay_hours": los,
            "visit_frequency": p["visit_frequency"],
            # interaction features (mirror Phase 3)
            "bill_vs_dept_avg": round(billed / (p["dept_avg_billed"] + 1), 4),
            "bill_x_provider_rate": round(billed * p["provider_rejection_rate"], 2),
            "lag_x_amount": round(p["billing_lag"] * billed, 2),
            "chronic_x_amount": round(p["chronic_flag"] * billed, 2),
            # encoded categoricals
            "insurance_provider_enc": bundle.encode_categorical(
                "insurance_provider", p["insurance_provider"]
            ),
            "visit_type_enc": bundle.encode_categorical("visit_type", p["visit_type"]),
            "department_enc": bundle.encode_categorical("department", p["department"]),
            "risk_score_enc": bundle.encode_categorical("risk_score", p["risk_score"]),
        }
        return feats

    # -------------------------------------------------------------- predict
    def _predict(
        self, bundle: ModelBundle, feats: Dict[str, float]
    ) -> Tuple[str, Dict[str, float], str, float]:
        # Build a single-row frame with columns in the exact training order,
        # so the pipeline aligns features by name (no positional drift).
        row = {name: float(feats[name]) for name in bundle.all_features}
        frame = pd.DataFrame([row], columns=bundle.all_features)

        proba = bundle.model.predict_proba(frame)[0]
        # classes_ are encoded ints [0..k]; align with label_encoder.classes_.
        probabilities = {
            bundle.classes[i]: round(float(proba[i]), 6) for i in range(len(proba))
        }

        # Threshold tuning for the business-critical minority class (Phase 3 logic).
        if proba[bundle.threshold_idx] >= bundle.threshold:
            pred_idx = bundle.threshold_idx
        else:
            pred_idx = int(np.argmax(proba))
        prediction = bundle.classes[pred_idx]

        feature_hash = self._hash_features(bundle.name, feats)
        return prediction, probabilities, feature_hash, bundle.threshold

    @staticmethod
    def _hash_features(model_name: str, feats: Dict[str, float]) -> str:
        payload = json.dumps(
            {"model": model_name, "features": feats},
            sort_keys=True,
            separators=(",", ":"),
            default=float,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def predict_risk(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._run(self.risk, self._engineer_risk, payload)

    def predict_claim(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._run(self.claim, self._engineer_claim, payload)

    def _run(self, bundle: ModelBundle, engineer, payload: Dict[str, Any]) -> Dict[str, Any]:
        if bundle is None:
            raise RuntimeError("Model not loaded")
        start = perf_counter()
        feats = engineer(payload, bundle)
        prediction, probabilities, feature_hash, threshold = self._predict(bundle, feats)
        latency_ms = round((perf_counter() - start) * 1000, 3)
        return {
            "model_name": bundle.name,
            "model_version": self._settings.MODEL_VERSION,
            "prediction": prediction,
            "probabilities": probabilities,
            "threshold_applied": threshold,
            "feature_hash": feature_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_ms": latency_ms,
        }

    # ------------------------------------------------------------- metadata
    def metadata(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for key, bundle in (("risk_model", self.risk), ("claim_model", self.claim)):
            if bundle is None:
                continue
            out[key] = {
                "model_name": bundle.name,
                "target": bundle.schema["target"],
                "target_classes": bundle.classes,
                "n_features": len(bundle.all_features),
                "threshold": bundle.threshold,
                "threshold_class": bundle.threshold_class,
            }
        return out


# Module-level singleton, populated on app startup.
service = PredictionService()
