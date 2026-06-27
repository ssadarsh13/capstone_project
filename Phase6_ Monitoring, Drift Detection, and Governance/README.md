# Phase 6 — Monitoring, Drift Detection, and Governance

Keeps the Phase 3 models (served via Phase 5) reliable and compliant over time by
validating incoming data, tracking feature & prediction drift, and maintaining an
audit trail.

## Layout

```
Phase 6 — Monitoring, Drift Detection, and Governance/
├── monitoring/
│   ├── config.py            # paths, numeric bounds, PSI thresholds
│   ├── data_validation.py   # missing / range / unseen-category checks
│   ├── drift_detection.py   # PSI + KS + chi-square; feature & prediction drift
│   └── audit_log.py         # append + summarize JSON-lines audit trail
├── generate_drift_report.py # builds reports/drift_report.md (+ charts, json)
├── run_monitoring.py        # CLI: validate | drift | audit
├── reports/                 # generated drift report, metrics, figures
├── tests/test_monitoring.py # 11 unit tests
├── GOVERNANCE.md            # governance & compliance document
└── requirements.txt
```

## Deliverables

| Deliverable | Location |
|-------------|----------|
| **Monitoring scripts** | `monitoring/` + `run_monitoring.py` |
| **Drift detection report** | [`reports/drift_report.md`](reports/drift_report.md) (+ `drift_metrics.json`, PNGs) |
| **Governance & compliance document** | [`GOVERNANCE.md`](GOVERNANCE.md) |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Validate an incoming batch before it reaches the model
python run_monitoring.py validate batch.csv --model claim_model [--json]

# Regenerate the drift detection report (reference = earliest 80%, current = latest 20%)
python run_monitoring.py drift

# Summarize a prediction audit log (volume, versions, latency, integrity)
python run_monitoring.py audit /var/log/hospital-ml/predictions.log

# Tests
python -m pytest tests/ -q
```

## What it checks

**Data validation** — missing columns, null values, numeric-range violations, and
categories outside the training vocabulary. Bounds mirror the Phase 5 API contract.

**Drift detection** — Population Stability Index (PSI) with a KS test for numeric
features and a chi-square test for categoricals, plus drift in each model's output
class distribution. PSI bands: `<0.10` stable · `0.10–0.25` moderate · `≥0.25`
significant.

**Audit log** — append-only JSON lines with `request_id`, `model_version`, and a
SHA-256 `feature_hash` (PHI-free traceability), matching the Phase 5 API format.

See [GOVERNANCE.md](GOVERNANCE.md) for thresholds, ownership, retraining policy,
limitations, and compliance.
