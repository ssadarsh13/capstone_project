# Deliverables — Location Guide

This file maps every expected deliverable to its exact location in the repository.
Paths are relative to the project root (`Capstone Project/`).

> **Folder names contain spaces** (e.g. `Phase5_ Deployment and API Integration`).
> When using a terminal, wrap paths in quotes:
> `cd "Phase5_ Deployment and API Integration"`.

## Folder map

| Phase | Folder |
|-------|--------|
| Phase 1 | *(project root)* |
| Phase 2 | `Phase2_EDA/` |
| Phase 3 | `Phase3_Modeling/` |
| Phase 4 | `Phase4_Evaluation/` |
| Phase 5 | `Phase5_ Deployment and API Integration/` |
| Phase 6 | `Phase6_ Monitoring, Drift Detection, and Governance/` |
| Final Phase | `Final Phase_ Executive Business Presentation/` |

---

## Phase 1 — SQL Analytics Layer (Business Intelligence Foundation)

| Expected deliverable | Location |
|----------------------|----------|
| `Phase1_SQL.ipynb` | `Phase1_SQL.ipynb` *(project root)* |

Source database: `HospitalManagementDB.sqlite` (root). Raw exports: `patients.csv`, `visits.csv`, `billing.csv` (root).

---

## Phase 2 — Exploratory Data Analysis and Data Quality

| Expected deliverable | Location |
|----------------------|----------|
| EDA notebook (`01_eda.ipynb`) | `Phase2_EDA/01_eda.ipynb` |
| Feature engineering script (`build_features.py`) | `Phase2_EDA/build_features.py` |
| Modeling dataset (csv or parquet) | `Phase2_EDA/model_table.csv` and `Phase2_EDA/model_table.parquet` |
| Data quality report (docx or markdown) | `Phase2_EDA/data_quality_report.md` |

Supporting EDA figures (`.png`) are also in `Phase2_EDA/`.

---

## Phase 3 — Model Development (Classification Systems)

| Expected deliverable | Location |
|----------------------|----------|
| Risk model notebook (`02_risk_model.ipynb`) | `Phase3_Modeling/02_risk_model.ipynb` |
| Claim model notebook (`03_claim_model.ipynb`) | `Phase3_Modeling/03_claim_model.ipynb` |
| Saved model artifacts (`.joblib`) | `Phase3_Modeling/models/` |
| Feature schema file (`feature_schema.json`) | `Phase3_Modeling/feature_schema.json` |

Artifacts in `Phase3_Modeling/models/`: `risk_model.joblib`, `claim_model.joblib`,
`risk_label_encoder.joblib`, `claim_label_encoder.joblib`,
`risk_threshold.joblib`, `claim_threshold.joblib`.

---

## Phase 4 — Model Evaluation and Explainability

| Expected deliverable | Location |
|----------------------|----------|
| Risk model evaluation report | `Phase4_Evaluation/04_risk_evaluation.ipynb` |
| Claim model evaluation report | `Phase4_Evaluation/05_claim_evaluation.ipynb` |
| Model card document | `Phase4_Evaluation/model_card.ipynb` |
| Explainability summary | Feature-importance / SHAP sections **inside** `04_risk_evaluation.ipynb` and `05_claim_evaluation.ipynb` (see "Feature Importance & Explainability") |

Evaluation figures (confusion matrices, fairness, feature importance `.png`) are in `Phase4_Evaluation/`.

---

## Phase 5 — Deployment and API Integration

| Expected deliverable | Location |
|----------------------|----------|
| API source code | `Phase5_ Deployment and API Integration/app/` |
| Deployment guide | `Phase5_ Deployment and API Integration/DEPLOYMENT.md` (and `DEPLOYMENT.docx`) |
| Sample request and response documentation | `Phase5_ Deployment and API Integration/README.md` + `Phase5_ Deployment and API Integration/samples/` (`risk_request.json`, `claim_request.json`) |

Also included: `Dockerfile`, `requirements.txt`, `tests/`, and bundled model artifacts in `models/`.

---

## Phase 6 — Monitoring, Drift Detection, and Governance

| Expected deliverable | Location |
|----------------------|----------|
| Monitoring scripts | `Phase6_ Monitoring, Drift Detection, and Governance/monitoring/` + `generate_drift_report.py` + `run_monitoring.py` |
| Drift detection report | `Phase6_ Monitoring, Drift Detection, and Governance/reports/drift_report.md` (+ `drift_metrics.json`, PSI charts) |
| Governance and compliance document | `Phase6_ Monitoring, Drift Detection, and Governance/GOVERNANCE.md` |

Tests: `Phase6_ Monitoring, Drift Detection, and Governance/tests/test_monitoring.py`.

---

## Final Phase — Executive Business Presentation

| Expected deliverable | Location |
|----------------------|----------|
| `Healthcare_Insights_Report.docx` | `Final Phase_ Executive Business Presentation/Healthcare_Insights_Report.docx` |

A longer companion version is also available:
`Final Phase_ Executive Business Presentation/Healthcare_Insights_Report_Detailed.docx`.

---

### Notes
- `.ipynb` files are executed and contain saved outputs; open in Jupyter/VS Code.
- `.joblib` model artifacts were trained with scikit-learn 1.8.0 / numpy 2.3.4 — use the pinned `requirements.txt` in Phase 5/6 to load them without unpickle mismatches.
- Some "report" deliverables are notebooks (Phase 4) or markdown (Phase 2/6) rather than `.docx`, as permitted by each phase's spec.
