"""Structured prediction audit logging.

Every prediction emits one JSON object, both to a rotating file and to stdout
(so it is captured by Docker / CloudWatch Logs on ECS Fargate). The record is
audit-ready: timestamp, model name + version, the SHA-256 hash of the engineered
feature vector, the prediction, and latency.
"""

from __future__ import annotations

import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict

from .config import get_settings


def _build_logger() -> logging.Logger:
    settings = get_settings()
    logger = logging.getLogger("prediction_audit")
    logger.setLevel(settings.LOG_LEVEL)
    logger.propagate = False

    if logger.handlers:  # already configured (e.g. reload)
        return logger

    # stdout — captured by the container runtime / CloudWatch.
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream)

    # rotating file — local durable copy for replay / investigation.
    try:
        log_path: Path = settings.PREDICTION_LOG_PATH
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(file_handler)
    except OSError:
        # A read-only filesystem (some container setups) must not crash the
        # service — stdout logging still satisfies the audit requirement.
        logger.warning(json.dumps({"event": "file_logging_disabled"}))

    return logger


_logger = _build_logger()


def log_prediction(record: Dict[str, Any]) -> None:
    """Emit one audit record as a single JSON line."""
    _logger.info(json.dumps(record, separators=(",", ":"), default=str))
