# Governance & Compliance Document

**System:** Hospital AI Classification System (Visit-Risk Model A + Claim-Outcome Model B)
**Version:** 1.0 · **Owner:** Data Science / MLOps · **Last reviewed:** 2026-06-27

This document defines how the AI system is monitored, validated, versioned, and
retrained to remain reliable, safe, and compliant as hospital operations and
payer behaviour evolve.

---

## 1. Scope & Intended Use

| Model | Purpose | Decision support — **not** automation |
|-------|---------|----------------------------------------|
| **A — Visit Risk** | Flag Low/Medium/High operational-clinical risk for triage prioritisation | A clinician always makes the final call |
| **B — Claim Outcome** | Predict Paid/Pending/Rejected before submission for denial management | A billing specialist reviews before action |

**Out of scope / prohibited uses:** automated discharge, medication or treatment
decisions, automated claim denial, or any use that overrides a qualified human.
Both models are advisory and require human-in-the-loop sign-off.

---

## 2. Roles & Responsibilities

| Role | Responsibility |
|------|----------------|
| **Model owner (Data Science)** | Model performance, retraining decisions, drift sign-off |
| **MLOps / Platform** | Pipeline health, deployment, audit-log retention, alerting |
| **Clinical lead** | Validates Model A behaviour is clinically safe; owns escalation |
| **Revenue-cycle lead** | Validates Model B behaviour; owns false-positive tolerance |
| **Compliance / Privacy officer** | PHI handling, audit reviews, regulatory reporting |

Monitoring runs are reviewed on a **monthly cadence**; significant drift triggers
an out-of-cycle review.

---

## 3. Data Validation Policy

Every incoming batch is screened by `monitoring/data_validation.py` **before**
inference. Checks (mirroring the Phase 5 API contract):

| Check | Rule | On failure |
|-------|------|-----------|
| Missing column | All model input columns must be present | Reject batch |
| Missing value | No nulls in required fields | Quarantine row |
| Numeric range | Values within documented bounds (e.g. `age` 0–120, `provider_rejection_rate` 0–1) | Quarantine row |
| Unseen category | Categoricals must match the training vocabulary | Quarantine row |

**Policy:** flagged records are quarantined, **not** silently served. A batch with
any missing-column issue is rejected wholesale. Quarantined records are routed to
the data-quality team. (Example real finding: `billing_lag` carries negative
values for recent records where the bill date precedes the visit date — caught by
the range check.)

---

## 4. Drift Monitoring Policy

`monitoring/drift_detection.py` compares a **reference window** (training era)
against the **current window** (recent operations).

| Signal | Method | Threshold |
|--------|--------|-----------|
| Numeric feature drift | PSI over quantile bins + KS test | PSI ≥ 0.25 **or** KS p < 0.05 |
| Categorical feature drift | PSI over categories + chi-square | PSI ≥ 0.25 **or** chi² p < 0.05 |
| Prediction drift | PSI + chi-square on output class mix | PSI ≥ 0.25 **or** chi² p < 0.05 |

**PSI bands:** `< 0.10` stable · `0.10–0.25` moderate (watch) · `≥ 0.25` significant (act).

**Cadence:** drift job runs monthly (`python run_monitoring.py drift`) and produces
[`reports/drift_report.md`](reports/drift_report.md) plus a machine-readable
`reports/drift_metrics.json`.

**Escalation:**
1. **Moderate** drift on any feature → log and watch next cycle.
2. **Significant** feature drift → investigate the upstream source; rule out a
   pipeline bug before concluding it is a true population shift.
3. **Prediction drift** → pull recent ground-truth labels and recompute live
   precision/recall against the Phase 4 targets (High-Risk recall ≥ 0.70,
   Rejected recall ≥ 0.65). Material degradation triggers retraining.
4. **Sustained drift across two consecutive cycles** → mandatory retraining.

> Note: some features (`days_since_registration`, `billing_lag`) drift *by design*
> because they are time-relative; reviewers interpret these in context rather than
> treating every high PSI as an incident.

---

## 5. Audit Logging & Traceability

Every prediction — online (Phase 5 API) or offline (batch) — emits one
append-only JSON record (`monitoring/audit_log.py`, same schema as the API):

```json
{"event":"prediction","request_id":"<uuid>","model_name":"risk_model_v2",
 "model_version":"v2.0.0","prediction":"High","probabilities":{...},
 "feature_hash":"<sha256>","timestamp":"<UTC ISO-8601>"}
```

- **`feature_hash`** (SHA-256 of the engineered feature vector) proves which exact
  inputs produced a decision **without storing PHI** in the log.
- **`model_version`** ties every decision to a specific artifact generation.
- **`request_id`** gives a unique handle for dispute/appeal traceability.
- Logs are shipped to CloudWatch (90-day retention minimum; extend per regulation)
  and summarised with `python run_monitoring.py audit <log>` (volume, version mix,
  latency percentiles, duplicate-id integrity check).

---

## 6. Model Versioning

- Artifacts are versioned with a semantic `MODEL_VERSION` (current: **v2.0.0**).
- The version is stamped on every API response and every audit record.
- A version bump is **required** whenever the model, feature schema, or tuned
  thresholds change. Old versions remain retrievable from ECR + Git for rollback
  and for reproducing historical decisions.

---

## 7. System Limitations

1. **Modest overall accuracy.** v2 prioritises minority-class recall (High-Risk,
   Rejected) via threshold tuning, trading majority-class precision. Reviewers
   should expect more false-positive alerts by design.
2. **Temporal sensitivity.** Time-relative features drift continuously; the model
   degrades on future data and must be retrained on a rolling window.
3. **Pending-claim ambiguity.** "Pending" is an interim state; some Pending claims
   later resolve to Paid/Rejected, capping Model B's achievable accuracy.
4. **Single model across payers.** One claim model serves four insurers with
   differing adjudication logic; per-insurer specialisation is a known future step.
5. **No free-text / clinical-notes features.** Models use structured fields only;
   nuance captured in notes is invisible to them.
6. **Label provenance.** Ground-truth `risk_score` and `claim_status` are assumed
   correct; systematic labelling errors would propagate.

---

## 8. Assumptions

- Training data is representative of the deployment population at release time.
- Feature values are available at decision time (no post-hoc leakage features —
  e.g. `approved_amount`, `payment_days` are excluded by design).
- Categorical vocabularies are stable; new categories surface via the validation
  layer rather than being silently encoded.
- `provider_rejection_rate` and `dept_avg_billed` aggregates are refreshed on a
  documented schedule (monthly) from historical data.
- Hospital and payer behaviour evolve slowly enough that a monthly monitoring
  cadence is sufficient to detect harmful drift before material impact.

---

## 9. Retraining Strategy

**Triggers (any one):**
- Significant feature **or** prediction drift sustained across two monitoring cycles.
- Live High-Risk recall < 0.70 or Rejected recall < 0.65 on fresh labelled data.
- A scheduled **quarterly** refresh, regardless of drift (staleness guard).
- Material change in upstream schema, payer mix, or clinical protocols.

**Procedure:**
1. Assemble a rolling **12-month** training window (most recent data).
2. Re-run Phase 3 training (time-based 80/20 split, SMOTE in CV folds, tuned
   thresholds) and Phase 4 evaluation (incl. fairness segmentation by gender,
   city, insurer).
3. Gate on: minority-class recall targets met, train/test gap ≤ 0.05, no new
   fairness regression.
4. Bump `MODEL_VERSION`, register new artifacts, deploy via the Phase 5 rolling
   update, and archive the previous version for rollback.
5. Record the retraining event, data window, and metrics in the model registry.

**Rollback:** if a new version regresses in production, re-point the service to the
previous task revision (see Phase 5 `DEPLOYMENT.md`); audit logs distinguish the
versions for impact analysis.

---

## 10. Compliance Considerations

- **Patient privacy (PHI):** logs store a feature **hash**, never raw patient
  attributes; access to raw data and logs is least-privilege and audited.
- **Human oversight:** both models are advisory; a qualified human signs off every
  consequential decision. This is enforced as policy, not just convention.
- **Fairness:** performance is segmented by gender, city, and insurer at every
  retraining (Phase 4). Material gaps block release pending mitigation.
- **Transparency:** feature-importance and per-decision probabilities are available
  to support clinician/finance trust and patient-appeal processes.
- **Accountability:** `request_id` + `model_version` + `feature_hash` provide an
  end-to-end audit trail for any contested prediction.
- **Retention & reporting:** audit logs retained ≥ 90 days (extend to satisfy local
  healthcare/financial regulation); drift reports retained for trend analysis.

---

## 11. Monitoring Runbook (quick reference)

```bash
# Validate an incoming batch before serving
python run_monitoring.py validate incoming_batch.csv --model claim_model

# Regenerate the drift detection report (monthly)
python run_monitoring.py drift

# Summarize the prediction audit trail for a governance review
python run_monitoring.py audit /var/log/hospital-ml/predictions.log
```

Outputs: `reports/drift_report.md`, `reports/drift_metrics.json`, and console
validation/audit summaries. Review monthly; escalate per §4.
