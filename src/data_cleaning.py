"""
Data cleaning pipeline for LendingClub Loan Default Predictor.
Input:  data/raw/archive/accepted_2007_to_2018Q4.csv.gz
Output: data/processed/loans_cleaned.parquet
"""

import gzip
import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[1]
RAW  = BASE / "data/raw/archive/accepted_2007_to_2018Q4.csv.gz"
OUT  = BASE / "data/processed/loans_cleaned.parquet"
OUT.parent.mkdir(parents=True, exist_ok=True)


# ── 1. TARGET VARIABLE ─────────────────────────────────────────────────────────
# Binary: 1 = default, 0 = fully paid
# Drop ambiguous / still-active statuses
TARGET_MAP = {
    "Fully Paid": 0,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    "Does not meet the credit policy. Status:Charged Off": 1,
    # Late (31-120 days) is high-risk but not yet defaulted — kept as 1
    "Late (31-120 days)": 1,
}
# Dropped: "Current", "In Grace Period", "Late (16-30 days)" — status unknown at prediction time


# ── 2. COLUMNS TO DROP ────────────────────────────────────────────────────────
# 2a. >50% missing
HIGH_MISSING = [
    "member_id",
    # hardship fields
    "hardship_flag", "hardship_type", "hardship_reason", "hardship_status",
    "deferral_term", "hardship_amount", "hardship_start_date", "hardship_end_date",
    "payment_plan_start_date", "hardship_length", "hardship_dpd",
    "hardship_loan_status", "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount", "hardship_last_payment_amount",
    # settlement fields
    "debt_settlement_flag", "debt_settlement_flag_date", "settlement_status",
    "settlement_date", "settlement_amount", "settlement_percentage", "settlement_term",
    # joint-application secondary applicant (95%+ missing)
    "sec_app_mths_since_last_major_derog", "sec_app_revol_util", "revol_bal_joint",
    "sec_app_mort_acc", "sec_app_fico_range_low", "sec_app_fico_range_high",
    "sec_app_earliest_cr_line", "sec_app_inq_last_6mths", "sec_app_open_acc",
    "sec_app_open_act_il", "sec_app_num_rev_accts", "sec_app_chargeoff_within_12_mths",
    "sec_app_collections_12_mths_ex_med",
    # joint income/dti (94-95% missing)
    "verification_status_joint", "dti_joint", "annual_inc_joint",
    # other high-missing
    "desc",                          # 94% missing free-text
    "mths_since_last_record",        # 84% missing
    "mths_since_recent_bc_dlq",      # 77% missing
    "mths_since_last_major_derog",   # 74% missing
    "mths_since_recent_revol_delinq",# 67% missing
    "next_pymnt_d",                  # 60% missing
    "mths_since_last_delinq",        # 51% missing
]

# 2b. Post-origination / data-leakage (known only after loan is issued)
LEAKAGE = [
    "out_prncp", "out_prncp_inv",
    "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt",
    "next_pymnt_d",          # already in HIGH_MISSING, harmless duplicate
    "last_credit_pull_d",
    "last_fico_range_high", "last_fico_range_low",
]

# 2c. Identifiers / free-text / administrative
ADMIN = [
    "id", "url",
    "title",        # free-text loan title, redundant with purpose
    "zip_code",     # too granular; addr_state kept
    "policy_code",  # constant = 1
    "pymnt_plan",   # nearly all 'n'
]

COLS_TO_DROP = list(set(HIGH_MISSING + LEAKAGE + ADMIN))


# ── 3. FEATURE ENGINEERING helpers ────────────────────────────────────────────
def parse_term(s):
    """' 36 months' → 36"""
    return pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce")


def parse_emp_length(s):
    """'10+ years' → 10, '< 1 year' → 0, '2 years' → 2"""
    s = s.astype(str)
    out = s.str.extract(r"(\d+)")[0].astype(float)
    out[s.str.contains("< 1", na=False)] = 0
    return out


def parse_pct(s):
    """'13.5%' → 13.5 (handles already-numeric)"""
    if pd.api.types.is_numeric_dtype(s):
        return s
    return pd.to_numeric(s.str.replace("%", "", regex=False), errors="coerce")


def months_since(series, reference_date):
    """Convert month-year string (e.g. 'Jan-2015') to months since reference."""
    dt = pd.to_datetime(series, format="%b-%Y", errors="coerce")
    return ((reference_date - dt) / pd.Timedelta(days=30.44)).round().astype("Int64")


# ── 4. MAIN PIPELINE ──────────────────────────────────────────────────────────
def load_and_clean():
    print("Loading raw data …")
    with gzip.open(RAW, "rt") as f:
        df = pd.read_csv(f, low_memory=False)
    print(f"  Raw shape: {df.shape}")

    # ── 4.1 Filter to labelled statuses only ──────────────────────────────────
    df = df[df["loan_status"].isin(TARGET_MAP)].copy()
    df["default"] = df["loan_status"].map(TARGET_MAP).astype(int)
    print(f"  After filtering ambiguous statuses: {df.shape}")
    print(f"  Default rate: {df['default'].mean():.3%}")

    # ── 4.2 Drop unwanted columns ─────────────────────────────────────────────
    to_drop = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=to_drop + ["loan_status"])
    print(f"  After dropping {len(to_drop)} columns: {df.shape}")

    # ── 4.3 Parse / encode features ───────────────────────────────────────────
    df["term"]       = parse_term(df["term"])
    df["emp_length"] = parse_emp_length(df["emp_length"])
    df["int_rate"]   = parse_pct(df["int_rate"])
    df["revol_util"] = parse_pct(df["revol_util"])

    # Date features: issue_d → loan age in months; earliest_cr_line → cr_history_months
    # issue_d is preserved as a datetime column for chronological splitting downstream.
    reference_date = pd.Timestamp("2019-01-01")   # ~end of dataset
    if "issue_d" in df.columns:
        df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
        df["loan_age_months"] = (
            (reference_date - df["issue_d"]) / pd.Timedelta(days=30.44)
        ).round().astype("Int64")
    if "earliest_cr_line" in df.columns:
        df["cr_history_months"] = months_since(df["earliest_cr_line"], reference_date)
        df = df.drop(columns=["earliest_cr_line"])

    # ── 4.4 Outlier capping (99th percentile for skewed numerics) ─────────────
    CLIP_COLS = {
        "annual_inc": (0, None),    # cap at 99th pct below
        "dti":        (0, 100),     # DTI > 100 is data error
        "revol_util": (0, 100),     # percentage can't exceed 100
        "revol_bal":  (0, None),
    }
    for col, (lo, hi) in CLIP_COLS.items():
        if col not in df.columns:
            continue
        if hi is None:
            hi = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=lo, upper=hi)

    # ── 4.5 Missing value imputation ─────────────────────────────────────────
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()

    # Numeric → median (issue_d left untouched — it's a datetime)
    medians = {col: df[col].median() for col in num_cols if df[col].isna().any()}
    df = df.fillna(medians)

    # Categorical → "Unknown"
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    # ── 4.6 Basic dtype cleanup ───────────────────────────────────────────────
    # Downcast float64 to float32 to save memory
    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype("float32")

    print(f"  Final shape: {df.shape}")
    print(f"  Default rate (final): {df['default'].mean():.3%}")

    return df


def main():
    df = load_and_clean()

    print(f"\nNull check after cleaning:")
    nulls = df.isnull().sum()
    print(nulls[nulls > 0] if nulls.any() else "  No nulls remaining.")

    print(f"\nData types:")
    print(df.dtypes.value_counts())

    print(f"\nSaving to {OUT} …")
    df.to_parquet(OUT, index=False)
    print("Done.")

    # Quick summary
    print("\n=== CLEANING SUMMARY ===")
    print(f"Rows:          {len(df):,}")
    print(f"Features:      {df.shape[1] - 1}")
    print(f"Default rate:  {df['default'].mean():.3%}")
    print(f"Output:        {OUT}")


if __name__ == "__main__":
    main()
