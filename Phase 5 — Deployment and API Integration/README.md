# Phase 5 — Hospital ML Prediction API

Real-time FastAPI service that serves the Phase 3 models:

| Model | Endpoint | Target | Classes |
|-------|----------|--------|---------|
| **A — Visit Risk** | `POST /predict/risk` | `risk_score` | Low / Medium / High |
| **B — Claim Outcome** | `POST /predict/claim` | `claim_status` | Paid / Pending / Rejected |

Consumers send **raw business inputs only**. All interaction features and label
encodings are reproduced server-side (mirroring `Phase3_Modeling`), so callers
cannot introduce train/serve skew. The tuned decision thresholds from Phase 3
(High-Risk and Rejected minority classes) are applied automatically.

---

## Project layout

```
Phase 5 — Deployment and API Integration/
├── app/
│   ├── main.py            # FastAPI app + endpoints
│   ├── schemas.py         # Pydantic request/response validation
│   ├── model_service.py   # model loading, feature engineering, inference
│   ├── logging_config.py  # JSON-lines prediction audit log
│   └── config.py          # env-driven settings
├── models/                # .joblib artifacts + feature_schema.json (self-contained)
├── tests/test_api.py      # 9 integration tests
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── README.md              # this file (API + samples)
└── DEPLOYMENT.md          # AWS ECS Fargate guide + ops runbook
```

---

## Run locally

```bash
# from this folder
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Interactive docs (Swagger UI): <http://localhost:8000/docs>

### Run the tests

```bash
python -m pytest tests/ -v
```

---

## Run with Docker

```bash
docker build -t hospital-ml-api:v2.0.0 .
docker run -d -p 8000:8000 --name hospital-ml hospital-ml-api:v2.0.0
curl http://localhost:8000/health
```

---

## Endpoints

### `GET /health`
Liveness + readiness. Used by the ECS/ALB health check.

```json
{
  "status": "ok",
  "service": "hospital-ml-api",
  "model_version": "v2.0.0",
  "models_loaded": { "risk_model": true, "claim_model": true },
  "uptime_seconds": 14.09
}
```

### `GET /metadata`
Served-model metadata for dashboards and governance.

```json
{
  "risk_model":  { "model_name": "risk_model_v2",  "target": "risk_score",   "target_classes": ["High","Low","Medium"],   "n_features": 17, "threshold": 0.2519, "threshold_class": "High" },
  "claim_model": { "model_name": "claim_model_v2", "target": "claim_status", "target_classes": ["Paid","Pending","Rejected"], "n_features": 20, "threshold": 0.1840, "threshold_class": "Rejected" }
}
```

---

## Sample: Visit Risk

### Request — `POST /predict/risk`

```json
{
  "age": 67,
  "chronic_flag": 1,
  "length_of_stay_hours": 72.5,
  "visit_month": 11,
  "visit_quarter": 4,
  "is_weekend": 0,
  "days_since_registration": 420,
  "visit_frequency": 5,
  "avg_los_per_patient": 40.2,
  "outlier_los": 0,
  "dept_avg_billed": 18500.0,
  "visit_type": "ICU",
  "department": "Cardiology"
}
```

```bash
curl -X POST http://localhost:8000/predict/risk \
  -H "Content-Type: application/json" \
  -d @risk_request.json
```

### Response `200 OK`

```json
{
  "request_id": "f47fec9d-b07b-441d-bc5e-3c262d73ba24",
  "model_name": "risk_model_v2",
  "model_version": "v2.0.0",
  "prediction": "Medium",
  "probabilities": { "High": 0.224264, "Low": 0.318716, "Medium": 0.45702 },
  "threshold_applied": 0.25186468113414767,
  "feature_hash": "08ea630534f8162d5179c9bbbf1ea634c2ba98832b5882a72eb305250046af4d",
  "timestamp": "2026-06-27T09:14:39.396974+00:00",
  "latency_ms": 70.062
}
```

---

## Sample: Claim Outcome

### Request — `POST /predict/claim`

```json
{
  "billed_amount": 42000.0,
  "provider_rejection_rate": 0.18,
  "dept_avg_billed": 18500.0,
  "age": 67,
  "chronic_flag": 1,
  "billing_lag": 5,
  "visit_month": 11,
  "visit_quarter": 4,
  "outlier_billed": 0,
  "length_of_stay_hours": 72.5,
  "visit_frequency": 5,
  "insurance_provider": "MediCareX",
  "visit_type": "ICU",
  "department": "Cardiology",
  "risk_score": "High"
}
```

### Response `200 OK`

```json
{
  "request_id": "8f449ef1-0926-4259-a5be-2b99b91275ef",
  "model_name": "claim_model_v2",
  "model_version": "v2.0.0",
  "prediction": "Pending",
  "probabilities": { "Paid": 0.244296, "Pending": 0.692361, "Rejected": 0.063343 },
  "threshold_applied": 0.18395062530599463,
  "feature_hash": "bdf45c29c4999d3ab1dd489722cf8f375aff9fc5378f15bf1b7a073b051a402d",
  "timestamp": "2026-06-27T09:11:40.056382+00:00",
  "latency_ms": 6.491
}
```

---

## Input reference

### Risk model (`/predict/risk`)

| Field | Type | Constraint |
|-------|------|-----------|
| `age` | int | 0–120 |
| `chronic_flag` | int | 0 or 1 |
| `length_of_stay_hours` | float | 0–2000 |
| `visit_month` | int | 1–12 |
| `visit_quarter` | int | 1–4 (must match month) |
| `is_weekend` | int | 0 or 1 |
| `days_since_registration` | int | 0–20000 |
| `visit_frequency` | int | 1–1000 |
| `avg_los_per_patient` | float | 0–2000 |
| `outlier_los` | int | 0 or 1 |
| `dept_avg_billed` | float | ≥ 0 |
| `visit_type` | enum | `ER`, `ICU`, `OPD` |
| `department` | enum | `Cardiology`, `ER`, `General`, `ICU`, `Neurology`, `Orthopedics` |

### Claim model (`/predict/claim`)

| Field | Type | Constraint |
|-------|------|-----------|
| `billed_amount` | float | > 0 |
| `provider_rejection_rate` | float | 0–1 |
| `dept_avg_billed` | float | ≥ 0 |
| `age` | int | 0–120 |
| `chronic_flag` | int | 0 or 1 |
| `billing_lag` | int | 0–365 |
| `visit_month` | int | 1–12 |
| `visit_quarter` | int | 1–4 (must match month) |
| `outlier_billed` | int | 0 or 1 |
| `length_of_stay_hours` | float | 0–2000 |
| `visit_frequency` | int | 1–1000 |
| `insurance_provider` | enum | `CareOne`, `HealthPlus`, `MediCareX`, `SecureLife` |
| `visit_type` | enum | `ER`, `ICU`, `OPD` |
| `department` | enum | (as above) |
| `risk_score` | enum | `High`, `Low`, `Medium` |

> `billed_per_hour` and all interaction features are derived server-side — do **not** send them.

---

## Error responses

| Status | Meaning | Example trigger |
|--------|---------|-----------------|
| `422` | Validation error | unknown category, out-of-range value, extra field, month/quarter mismatch |
| `503` | Models not loaded | request during cold start / failed artifact load |
| `500` | Internal error | unexpected server fault (no stack trace leaked) |

Example `422`:

```json
{
  "detail": [{
    "type": "enum",
    "loc": ["body", "department"],
    "msg": "Input should be 'Cardiology', 'ER', 'General', 'ICU', 'Neurology' or 'Orthopedics'",
    "input": "Oncology"
  }]
}
```

---

## Prediction audit log

Every prediction emits one JSON line to **stdout** (captured by CloudWatch on
ECS) and to a rotating file (`PREDICTION_LOG_PATH`):

```json
{"event":"prediction","request_id":"f47fec9d-...","model_name":"risk_model_v2","model_version":"v2.0.0","prediction":"Medium","probabilities":{"High":0.224264,"Low":0.318716,"Medium":0.45702},"feature_hash":"08ea6305...","threshold_applied":0.2519,"timestamp":"2026-06-27T09:14:39.396974+00:00","latency_ms":70.062}
```

The `feature_hash` (SHA-256 of the engineered feature vector) lets you prove
which exact inputs produced a prediction without storing PHI in the log.

---

## Configuration (environment variables)

| Variable | Default | Purpose |
|----------|---------|---------|
| `MODELS_DIR` | `./models` | location of `.joblib` + schema |
| `MODEL_VERSION` | `v2.0.0` | version stamped on responses & logs |
| `PREDICTION_LOG_PATH` | `./logs/predictions.log` | durable audit log file |
| `SERVICE_NAME` | `hospital-ml-api` | reported in `/health` |
| `LOG_LEVEL` | `INFO` | logging verbosity |

See [DEPLOYMENT.md](DEPLOYMENT.md) for the AWS ECS Fargate guide and operations runbook.
