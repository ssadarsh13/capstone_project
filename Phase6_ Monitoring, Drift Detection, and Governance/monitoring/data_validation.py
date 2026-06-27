"""Incoming-data validation: missing values, numeric ranges, unseen categories.

Designed to screen a batch (DataFrame) of incoming records *before* they reach
the model, and to be reusable as a pre-inference gate. Returns a structured
report; never raises on bad data — the caller decides what to quarantine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .config import NUMERIC_BOUNDS, load_schema


@dataclass
class ValidationIssue:
    column: str
    check: str          # "missing" | "range" | "unseen_category" | "missing_column"
    count: int
    detail: str
    examples: List[Any] = field(default_factory=list)


@dataclass
class ValidationReport:
    n_rows: int
    issues: List[ValidationIssue]

    @property
    def passed(self) -> bool:
        return len(self.issues) == 0

    @property
    def n_flagged_rows(self) -> int:
        return int(sum(i.count for i in self.issues))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "passed": self.passed,
            "n_issues": len(self.issues),
            "issues": [
                {
                    "column": i.column,
                    "check": i.check,
                    "count": i.count,
                    "detail": i.detail,
                    "examples": i.examples,
                }
                for i in self.issues
            ],
        }

    def summary(self) -> str:
        if self.passed:
            return f"PASS — {self.n_rows} rows, no validation issues."
        lines = [f"FAIL — {self.n_rows} rows, {len(self.issues)} issue(s):"]
        for i in self.issues:
            lines.append(f"  [{i.check}] {i.column}: {i.detail} (count={i.count})")
        return "\n".join(lines)


def _allowed_categories(model_key: str) -> Dict[str, List[str]]:
    schema = load_schema()
    return schema[model_key]["categorical_encodings"]


def validate_batch(df: pd.DataFrame, model_key: str = "claim_model") -> ValidationReport:
    """Validate a batch of raw input records for the given model.

    `model_key` selects which schema's required columns and category vocab to
    enforce ("risk_model" or "claim_model"). The claim model is the superset,
    so it is the safe default for a combined feed.
    """
    schema = load_schema()[model_key]
    issues: List[ValidationIssue] = []

    required_numeric = [c for c in schema["numeric_features"] if c in NUMERIC_BOUNDS]
    required_categorical = schema["categorical_features"]
    allowed = _allowed_categories(model_key)

    # --- 1. Missing columns entirely ---
    expected_cols = set(required_numeric) | set(required_categorical)
    present = set(df.columns)
    for col in sorted(expected_cols - present):
        issues.append(
            ValidationIssue(
                column=col, check="missing_column", count=len(df),
                detail="expected column is absent from the batch",
            )
        )

    # --- 2. Missing values ---
    for col in sorted(expected_cols & present):
        n_missing = int(df[col].isna().sum())
        if n_missing:
            issues.append(
                ValidationIssue(
                    column=col, check="missing", count=n_missing,
                    detail=f"{n_missing} null value(s)",
                )
            )

    # --- 3. Numeric range violations ---
    for col in required_numeric:
        if col not in present:
            continue
        low, high = NUMERIC_BOUNDS[col]
        series = pd.to_numeric(df[col], errors="coerce")
        mask = series.notna() & ((series < low) | (series > high))
        n_bad = int(mask.sum())
        if n_bad:
            examples = series[mask].head(3).tolist()
            issues.append(
                ValidationIssue(
                    column=col, check="range", count=n_bad,
                    detail=f"{n_bad} value(s) outside [{low}, {high}]",
                    examples=examples,
                )
            )

    # --- 4. Unseen categories ---
    for col in required_categorical:
        if col not in present:
            continue
        valid = set(allowed[col])
        non_null = df[col].dropna().astype(str)
        bad_mask = ~non_null.isin(valid)
        n_bad = int(bad_mask.sum())
        if n_bad:
            unseen = sorted(non_null[bad_mask].unique().tolist())
            issues.append(
                ValidationIssue(
                    column=col, check="unseen_category", count=n_bad,
                    detail=f"{n_bad} record(s) with categories not in training vocab",
                    examples=unseen[:5],
                )
            )

    return ValidationReport(n_rows=len(df), issues=issues)
