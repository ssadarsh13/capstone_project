"""Generate the Phase 6 drift detection report.

Splits the historical data into a reference window (earliest 80%, the
training era) and a current window (latest 20%, recent operations), then:
  1. validates the current window,
  2. measures feature drift (PSI + KS/chi-square),
  3. scores both windows with the Phase 3 models and measures prediction drift,
  4. writes reports/drift_report.md with charts.

Run:  python generate_drift_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monitoring.config import (
    DATA_PATH, MODELS_DIR, REFERENCE_FRACTION, load_schema,
    PSI_NO_DRIFT, PSI_MODERATE_DRIFT,
)
from monitoring.data_validation import validate_batch
from monitoring.drift_detection import (
    compute_feature_drift, compute_prediction_drift, DriftResult,
)

HERE = Path(__file__).resolve().parent
REPORTS = HERE / "reports"
REPORTS.mkdir(exist_ok=True)


# --------------------------------------------------------------- feature build
def engineer_risk(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    d = df.copy()
    d["los_x_chronic"] = d["length_of_stay_hours"] * d["chronic_flag"]
    d["age_x_frequency"] = d["age"] * d["visit_frequency"]
    d["dept_los_ratio"] = (d["length_of_stay_hours"] / (d["dept_avg_billed"] + 1)).round(4)
    d["age_x_chronic"] = d["age"] * d["chronic_flag"]
    enc = schema["categorical_encodings"]
    for col in schema["categorical_features"]:
        d[col + "_enc"] = d[col].astype(str).map(lambda v, c=col: enc[c].index(v) if v in enc[c] else -1)
    return d[schema["all_features"]]


def engineer_claim(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    d = df.copy()
    d["billed_per_hour"] = np.where(
        d["length_of_stay_hours"] > 0,
        (d["billed_amount"] / d["length_of_stay_hours"]).round(2), 0.0,
    )
    d["bill_vs_dept_avg"] = (d["billed_amount"] / (d["dept_avg_billed"] + 1)).round(4)
    d["bill_x_provider_rate"] = (d["billed_amount"] * d["provider_rejection_rate"]).round(2)
    d["lag_x_amount"] = (d["billing_lag"] * d["billed_amount"]).round(2)
    d["chronic_x_amount"] = (d["chronic_flag"] * d["billed_amount"]).round(2)
    enc = schema["categorical_encodings"]
    for col in schema["categorical_features"]:
        d[col + "_enc"] = d[col].astype(str).map(lambda v, c=col: enc[c].index(v) if v in enc[c] else -1)
    return d[schema["all_features"]]


def predict_labels(model, label_encoder, threshold_obj, X: pd.DataFrame) -> pd.Series:
    proba = model.predict_proba(X)
    classes = list(label_encoder.classes_)
    idx_key = "hi_idx" if "hi_idx" in threshold_obj else "rej_idx"
    t_idx = int(threshold_obj[idx_key])
    t_val = float(threshold_obj["threshold"])
    preds = np.argmax(proba, axis=1)
    preds[proba[:, t_idx] >= t_val] = t_idx
    return pd.Series([classes[i] for i in preds], index=X.index)


# ------------------------------------------------------------------- charts
def psi_bar_chart(results, title: str, path: Path) -> None:
    results = [r for r in results][:15]
    names = [r.feature for r in results]
    psis = [r.psi for r in results]
    colors = [
        "#4CAF50" if p < PSI_NO_DRIFT else "#FF9800" if p < PSI_MODERATE_DRIFT else "#F44336"
        for p in psis
    ]
    fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * len(names))))
    ax.barh(names[::-1], psis[::-1], color=colors[::-1])
    ax.axvline(PSI_NO_DRIFT, color="#FF9800", ls="--", lw=0.8)
    ax.axvline(PSI_MODERATE_DRIFT, color="#F44336", ls="--", lw=0.8)
    ax.set_xlabel("PSI")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------- report
def fmt_drift_table(results) -> str:
    rows = ["| Feature | Type | PSI | Band | Test | p-value | Drifted |",
            "|---------|------|-----|------|------|---------|---------|"]
    for r in results:
        flag = "⚠️ yes" if r.drifted else "no"
        rows.append(
            f"| `{r.feature}` | {r.kind} | {r.psi:.4f} | {r.psi_band} | "
            f"{r.stat_test} | {r.p_value:.4g} | {flag} |"
        )
    return "\n".join(rows)


def main() -> None:
    schema = load_schema()
    df = pd.read_csv(DATA_PATH, parse_dates=["visit_date"])
    df = df.sort_values("visit_date").reset_index(drop=True)

    split = int(len(df) * REFERENCE_FRACTION)
    ref, cur = df.iloc[:split].copy(), df.iloc[split:].copy()
    ref_period = f"{ref['visit_date'].min().date()} → {ref['visit_date'].max().date()}"
    cur_period = f"{cur['visit_date'].min().date()} → {cur['visit_date'].max().date()}"

    # 1. Validation of the current window
    val_report = validate_batch(cur, model_key="claim_model")

    # 2. Feature drift — raw business inputs used by the models
    numeric_feats = sorted({
        "age", "length_of_stay_hours", "days_since_registration", "visit_frequency",
        "avg_los_per_patient", "dept_avg_billed", "billed_amount", "billing_lag",
        "provider_rejection_rate", "chronic_flag",
    })
    categorical_feats = ["visit_type", "department", "insurance_provider", "risk_score"]
    feat_results = compute_feature_drift(ref, cur, numeric_feats, categorical_feats)

    # 3. Prediction drift via the actual models
    risk_schema, claim_schema = schema["risk_model"], schema["claim_model"]
    risk_model = joblib.load(MODELS_DIR / "risk_model.joblib")
    risk_le = joblib.load(MODELS_DIR / "risk_label_encoder.joblib")
    risk_thr = joblib.load(MODELS_DIR / "risk_threshold.joblib")
    claim_model = joblib.load(MODELS_DIR / "claim_model.joblib")
    claim_le = joblib.load(MODELS_DIR / "claim_label_encoder.joblib")
    claim_thr = joblib.load(MODELS_DIR / "claim_threshold.joblib")

    risk_ref = predict_labels(risk_model, risk_le, risk_thr, engineer_risk(ref, risk_schema))
    risk_cur = predict_labels(risk_model, risk_le, risk_thr, engineer_risk(cur, risk_schema))
    claim_ref = predict_labels(claim_model, claim_le, claim_thr, engineer_claim(ref, claim_schema))
    claim_cur = predict_labels(claim_model, claim_le, claim_thr, engineer_claim(cur, claim_schema))

    risk_pred_drift = compute_prediction_drift(risk_ref, risk_cur, "risk_prediction")
    claim_pred_drift = compute_prediction_drift(claim_ref, claim_cur, "claim_prediction")

    # Charts
    psi_bar_chart(feat_results, "Feature Drift — PSI (reference vs current)",
                  REPORTS / "feature_drift_psi.png")
    _prediction_dist_chart(risk_ref, risk_cur, claim_ref, claim_cur,
                           REPORTS / "prediction_drift.png")

    # 4. Write report
    _write_report(
        ref_period, cur_period, len(ref), len(cur), val_report,
        feat_results, risk_pred_drift, claim_pred_drift,
        risk_ref, risk_cur, claim_ref, claim_cur,
    )
    # Machine-readable companion
    (REPORTS / "drift_metrics.json").write_text(json.dumps({
        "reference_period": ref_period, "current_period": cur_period,
        "validation": val_report.to_dict(),
        "feature_drift": [r.to_dict() for r in feat_results],
        "prediction_drift": {
            "risk": risk_pred_drift.to_dict(), "claim": claim_pred_drift.to_dict(),
        },
    }, indent=2))
    print("Drift report written to", REPORTS / "drift_report.md")


def _prediction_dist_chart(risk_ref, risk_cur, claim_ref, claim_cur, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, ref, cur, title, order in [
        (axes[0], risk_ref, risk_cur, "Risk prediction distribution", ["Low", "Medium", "High"]),
        (axes[1], claim_ref, claim_cur, "Claim prediction distribution", ["Paid", "Pending", "Rejected"]),
    ]:
        rp = ref.value_counts(normalize=True).reindex(order).fillna(0)
        cp = cur.value_counts(normalize=True).reindex(order).fillna(0)
        x = np.arange(len(order))
        ax.bar(x - 0.2, rp.values, 0.4, label="reference", color="#90A4AE")
        ax.bar(x + 0.2, cp.values, 0.4, label="current", color="#1E88E5")
        ax.set_xticks(x); ax.set_xticklabels(order)
        ax.set_ylim(0, 1); ax.set_title(title); ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _dist_line(ref: pd.Series, cur: pd.Series, order) -> str:
    rp = ref.value_counts(normalize=True).reindex(order).fillna(0)
    cp = cur.value_counts(normalize=True).reindex(order).fillna(0)
    return " | ".join(f"{o}: {rp[o]*100:.1f}%→{cp[o]*100:.1f}%" for o in order)


def _write_report(ref_period, cur_period, n_ref, n_cur, val_report,
                  feat_results, risk_pred_drift, claim_pred_drift,
                  risk_ref, risk_cur, claim_ref, claim_cur) -> None:
    n_drift = sum(1 for r in feat_results if r.drifted)
    n_sig = sum(1 for r in feat_results if r.psi_band == "significant")

    md = f"""# Drift Detection Report

_Generated by `generate_drift_report.py`. Reference = earliest {int(REFERENCE_FRACTION*100)}% of
history (training era); current = most recent {100-int(REFERENCE_FRACTION*100)}% (recent operations)._

| Window | Period | Records |
|--------|--------|---------|
| Reference | {ref_period} | {n_ref:,} |
| Current | {cur_period} | {n_cur:,} |

## 1. Executive Summary

- **Data validation (current window):** {"✅ PASS" if val_report.passed else "❌ FAIL"} — {val_report.n_rows:,} rows, {len(val_report.issues)} issue(s).
- **Feature drift:** {n_drift} of {len(feat_results)} monitored features flagged ({n_sig} with *significant* PSI ≥ {PSI_MODERATE_DRIFT}).
- **Risk-prediction drift:** PSI {risk_pred_drift.psi:.4f} ({risk_pred_drift.psi_band}), chi² p={risk_pred_drift.p_value:.4g} → {"⚠️ drift" if risk_pred_drift.drifted else "stable"}.
- **Claim-prediction drift:** PSI {claim_pred_drift.psi:.4f} ({claim_pred_drift.psi_band}), chi² p={claim_pred_drift.p_value:.4g} → {"⚠️ drift" if claim_pred_drift.drifted else "stable"}.

> **PSI bands:** &lt;{PSI_NO_DRIFT} stable · {PSI_NO_DRIFT}–{PSI_MODERATE_DRIFT} moderate · ≥{PSI_MODERATE_DRIFT} significant.

## 2. Data Validation — Current Window

{val_report.summary()}

## 3. Feature Drift

![Feature drift PSI](feature_drift_psi.png)

{fmt_drift_table(feat_results)}

## 4. Prediction Drift

![Prediction drift](prediction_drift.png)

**Risk model** (reference→current share): {_dist_line(risk_ref, risk_cur, ["Low","Medium","High"])}
- PSI {risk_pred_drift.psi:.4f} ({risk_pred_drift.psi_band}) · chi² p={risk_pred_drift.p_value:.4g}

**Claim model** (reference→current share): {_dist_line(claim_ref, claim_cur, ["Paid","Pending","Rejected"])}
- PSI {claim_pred_drift.psi:.4f} ({claim_pred_drift.psi_band}) · chi² p={claim_pred_drift.p_value:.4g}

## 5. Recommended Actions

| Condition | Action |
|-----------|--------|
| Any feature PSI ≥ {PSI_MODERATE_DRIFT} | Investigate upstream source; confirm it is a real population shift, not a pipeline bug. |
| Prediction PSI ≥ {PSI_MODERATE_DRIFT} | Pull recent ground-truth labels; recompute live precision/recall against Phase 4 targets. |
| Validation FAIL | Quarantine offending records; do **not** auto-serve until the feed is corrected. |
| Sustained drift across 2+ cycles | Trigger the retraining workflow (see GOVERNANCE.md §Retraining). |

_See [GOVERNANCE.md](../GOVERNANCE.md) for thresholds, ownership, and the retraining policy._
"""
    (REPORTS / "drift_report.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
