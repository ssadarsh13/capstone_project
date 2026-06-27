"""Runtime configuration, sourced from environment variables with safe defaults."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


class Settings:
    """Application settings. Override any field via the matching env var."""

    def __init__(self) -> None:
        # Directory holding the .joblib artifacts and feature_schema.json.
        # Defaults to the `models/` dir shipped alongside the app in the image.
        default_models = Path(__file__).resolve().parent.parent / "models"
        self.MODELS_DIR: Path = Path(os.getenv("MODELS_DIR", str(default_models)))

        # Semantic version of the served model bundle. Surfaced in every
        # response and prediction log line for audit/governance.
        self.MODEL_VERSION: str = os.getenv("MODEL_VERSION", "v2.0.0")

        # Where prediction audit logs are written (JSON lines).
        default_log = Path(__file__).resolve().parent.parent / "logs" / "predictions.log"
        self.PREDICTION_LOG_PATH: Path = Path(
            os.getenv("PREDICTION_LOG_PATH", str(default_log))
        )

        # Service metadata.
        self.SERVICE_NAME: str = os.getenv("SERVICE_NAME", "hospital-ml-api")
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    @property
    def risk_model_path(self) -> Path:
        return self.MODELS_DIR / "risk_model.joblib"

    @property
    def risk_label_encoder_path(self) -> Path:
        return self.MODELS_DIR / "risk_label_encoder.joblib"

    @property
    def risk_threshold_path(self) -> Path:
        return self.MODELS_DIR / "risk_threshold.joblib"

    @property
    def claim_model_path(self) -> Path:
        return self.MODELS_DIR / "claim_model.joblib"

    @property
    def claim_label_encoder_path(self) -> Path:
        return self.MODELS_DIR / "claim_label_encoder.joblib"

    @property
    def claim_threshold_path(self) -> Path:
        return self.MODELS_DIR / "claim_threshold.joblib"

    @property
    def schema_path(self) -> Path:
        return self.MODELS_DIR / "feature_schema.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
