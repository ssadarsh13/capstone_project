"""
Phase 2 – Feature Engineering
Loads hospital data from SQLite, engineers model-ready features, and writes
model_table.csv (and model_table.parquet when pyarrow is available).
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).parent / "HospitalManagementDB.sqlite"
OUTPUT_CSV = Path(__file__).parent / "model_table.csv"
OUTPUT_PARQUET = Path(__file__).parent / "model_table.parquet"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # --- Time-based features ---
    df["visit_month"] = df["visit_date"].dt.month
    df["visit_quarter"] = df["visit_date"].dt.quarter
    df["visit_dayofweek"] = df["visit_date"].dt.dayofweek   # 0 = Mon, 6 = Sun
    df["is_weekend"] = (df["visit_dayofweek"] >= 5).astype(int)
    df["days_since_registration"] = (df["visit_date"] - df["registration_date"]).dt.days
    df["billing_lag"] = (df["billing_date"] - df["visit_date"]).dt.days

    # --- Patient-level aggregates (computed on full dataset, not leaking targets) ---
    df["visit_frequency"] = df.groupby("patient_id")["visit_id"].transform("count")
    df["avg_los_per_patient"] = (
        df.groupby("patient_id")["length_of_stay_hours"]
        .transform("mean")
        .round(2)
    )

    # --- Insurance-provider rejection rate ---
    rejection_rates = (
        df.groupby("insurance_provider")["claim_status"]
        .apply(lambda s: (s == "Rejected").mean())
        .rename("provider_rejection_rate")
        .round(4)
        .reset_index()
    )
    df = df.merge(rejection_rates, on="insurance_provider", how="left")

    # --- Department-level average billed amount ---
    df["dept_avg_billed"] = (
        df.groupby("department")["billed_amount"].transform("mean").round(2)
    )

    # --- Approval ratio: approved vs billed (0 for rejected/pending) ---
    df["approval_ratio"] = np.where(
        df["billed_amount"] > 0,
        (df["approved_amount"].fillna(0) / df["billed_amount"]).round(4),
        np.nan,
    )

    return df


def main(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
       print("\n combained dataframe missing...")
       return None

    print("\nEngineering features...")
    df = build_features(df)
    new_cols = [
        "visit_month", "visit_quarter", "visit_dayofweek", "is_weekend",
        "days_since_registration", "billing_lag",
        "visit_frequency", "avg_los_per_patient",
        "provider_rejection_rate", "dept_avg_billed", "approval_ratio",
    ]
    print(f"  Final shape : {df.shape}")
    print(f"  New features: {new_cols}")

    print(f"\nSaving CSV     : {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)

    try:
        df.to_parquet(OUTPUT_PARQUET, index=False)
        print(f"Saving Parquet : {OUTPUT_PARQUET}")
    except ImportError:
        print("pyarrow not installed — parquet output skipped.")

    print("\nDone.")
    return df


if __name__ == "__main__":
    main()