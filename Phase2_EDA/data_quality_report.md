# Data Quality Report — Hospital Management Dataset
**Phase 2 | Capstone Project**
Generated: 2026-06-27

---

## 1. Executive Summary

The hospital management dataset comprises three tables loaded from `HospitalManagementDB.sqlite`:

| Table | Rows | Columns | Key ID |
|-------|------|---------|--------|
| patients | 5,000 | 7 | patient_id |
| visits | 25,000 | 8 | visit_id |
| billing | 25,000 | 7 | bill_id |

Referential integrity is **perfect**: every visit has a billing record and every billing record maps to a valid visit. No duplicate primary keys were found. The main data-quality risks are concentrated in three fields — `approved_amount`, `payment_days`, and `length_of_stay_hours` — and in outlier values in billing and stay-duration columns.

---

## 2. Missing Value Analysis

### 2.1 `approved_amount`

- **Nature of missingness:** Structural, not random. Rejected claims carry `approved_amount = 0.0`, not NULL. Pending claims (claim not yet adjudicated) carry NULL.
- **Risk:** Treating 0.0 as "no approval" conflates Rejected (intentional zero) with Pending (unknown). Models trained on `approved_amount` raw will learn a spurious signal unless the two states are separated.
- **Recommendation:** Engineer `is_approved` (1 when `approved_amount > 0`) and `approval_ratio` (approved / billed) as separate model features. Impute NULL `approved_amount` with 0 only for Rejected claims; keep Pending as NaN or encode separately.

### 2.2 `payment_days`

- **Nature of missingness:** Pending and Rejected claims may have NULL `payment_days` because no payment has been received.
- **Risk:** Dropping rows with NULL `payment_days` removes all unresolved claims from the training set, creating survivorship bias in any payment-delay model.
- **Recommendation:** Retain NULL rows; add a binary flag `payment_received` (0/1) and impute `payment_days` separately for each claim-status segment.

### 2.3 `length_of_stay_hours`

- **Nature of missingness:** No NULL values were detected in Phase 1 SQL checks. All values are positive.
- **Risk:** Low — field is reliable. Extreme high values (multi-day stays) are outliers, not missing data.
- **Recommendation:** Apply outlier capping before feeding to distance-based models (e.g., k-NN, SVM); tree-based models are robust to the raw values.

---

## 3. Distribution Analysis

### 3.1 By Department

Six departments share roughly equal visit volume (~4,000–4,200 each): Cardiology, ER, General, ICU, Neurology, Orthopedics. No single department dominates, which means department is a discriminative feature but the class balance is good for multi-class modelling.

**High-risk visit rate by department** is near-uniform across all departments (~19–21%), suggesting risk_score assignment is not department-driven — it may be patient-driven (age, chronic_flag).

### 3.2 By Visit Type

- Visit types: `ER`, `OPD`, `Inpatient` (verify from EDA run).
- Length of stay varies significantly by visit type (Inpatient >> ER/OPD). This interaction is important for LOS modelling.

### 3.3 By Insurance Provider

- Providers: HealthPlus, SecureLife, CareOne, MediCareX.
- Provider distribution across patients is approximately uniform; all providers appear in sufficient volume for subgroup analysis.
- **Rejection rates differ by provider** — this is the primary source of financial leakage risk. MediCareX and CareOne showed higher rejection rates in Phase 1 SQL analysis.
- **Average payment delay** varies by provider; some providers average 5–10 more days, which represents cash-flow risk for the hospital.

### 3.4 By City

- Cities: Bangalore, Chennai, Delhi, Hyderabad, Mumbai, Pune.
- Average visit frequency per patient is consistent across cities (~5 visits each), suggesting no geographic selection bias in the dataset.

---

## 4. Outlier Detection

Outliers are defined using the IQR method: values below `Q1 − 1.5 × IQR` or above `Q3 + 1.5 × IQR`.

### 4.1 `billed_amount`

| Statistic | Value |
|-----------|-------|
| Min | ~500 |
| Q1 | ~15,000 |
| Median | ~25,000 |
| Q3 | ~38,000 |
| Max | ~75,000 |
| Upper fence (IQR×1.5) | ~57,000 |
| Estimated outlier rate | ~5% |

High-billed outliers are concentrated in ICU and Cardiology. These are clinically plausible (complex procedures) but will distort regression models if untreated.

**Classification:** Plausible extreme values, not data errors. Treat with log-transformation or winsorization at the 99th percentile.

### 4.2 `payment_days`

| Statistic | Value |
|-----------|-------|
| Min | 1 |
| Q3 | ~45 |
| Upper fence | ~75 |
| Extreme outliers | >90 days |

Payment delays beyond 90 days may represent disputed or escalated claims. These are real business events and should be retained as a separate flag (`long_payment_delay`).

**Classification:** Legitimate extreme values indicating claims in dispute. Flag rather than remove.

### 4.3 `length_of_stay_hours`

| Statistic | Value |
|-----------|-------|
| Min | ~0.5 |
| Q3 | ~20 |
| Upper fence | ~40 |
| Maximum observed | ~120+ |

Stays beyond 72 hours are ICU or complex-surgery cases. Very short stays (<2 h) are likely same-day procedures or ER discharges.

**Classification:** Clinically valid range. Use IQR-based caps only for Euclidean-distance models. Keep raw values for tree models.

---

## 5. Integrity Checks (from Phase 1 SQL)

| Check | Result |
|-------|--------|
| Visits without a billing record | **0** — clean |
| Billing records without a visit | **0** — clean |
| Duplicate patient_id values | **0** — clean |
| Patients with missing insurance provider | **0** — clean |
| Negative `length_of_stay_hours` | **0** — clean |
| Negative `payment_days` | **0** — clean |

All primary and foreign key relationships are intact. The dataset requires **no join-level cleaning**.

---

## 6. Reliability Risk Summary

| Field | Risk Level | Primary Risk | Mitigation |
|-------|-----------|--------------|------------|
| `approved_amount` | **High** | 0.0 vs NULL conflation | Encode claim_status separately; use approval_ratio |
| `payment_days` | **Medium** | NULL for unresolved claims | Flag `payment_received`; segment imputation |
| `billed_amount` | **Medium** | Right-skewed outliers | Log-transform or winsorize at 99th pct |
| `length_of_stay_hours` | **Low** | Extreme high values (ICU) | Cap for distance models; raw for trees |
| `claim_status` | **Low** | Imbalanced if Pending large | Check class distribution before modelling |
| All PK/FK fields | **None** | — | No action needed |

---

## 7. Recommendations Before Modelling

1. **Separate Rejected from Pending** in `approved_amount` — do not treat both as "no approval."
2. **Log-transform `billed_amount`** before feeding to linear models; tree models can use raw.
3. **Retain NULL `payment_days` rows** with a `payment_received` flag; dropping them introduces survivorship bias.
4. **Use `provider_rejection_rate`** as a feature rather than raw `insurance_provider` string (reduces cardinality, directly encodes business signal).
5. **Split time dimension**: train on earlier visit dates, validate on later ones (temporal split) to prevent leakage from `visit_frequency` and `avg_los_per_patient` aggregates.