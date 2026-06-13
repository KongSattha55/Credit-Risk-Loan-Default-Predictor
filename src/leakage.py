"""Centralized feature exclusions for origination-time modeling."""

from __future__ import annotations

from collections.abc import Iterable

TARGET = "default"

# Fields that are only known after loan origination or directly encode outcomes.
LEAKAGE_COLS = (
    "loan_status",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "last_credit_pull_d",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "out_prncp",
    "out_prncp_inv",
    "settlement_status",
    "settlement_date",
    "settlement_amount",
    "settlement_percentage",
    "settlement_term",
    "debt_settlement_flag",
    "hardship_flag",
    "loan_age_months",
)

# Non-leaky fields that are excluded because they are redundant or unwieldy.
NON_FEATURE_COLS = (
    "funded_amnt",
    "funded_amnt_inv",
    "fico_range_high",
    "emp_title",
)

ALWAYS_DROP = (TARGET, *NON_FEATURE_COLS, *LEAKAGE_COLS)


def feature_columns(
    columns: Iterable[str],
    extra_drop: Iterable[str] = (),
) -> list[str]:
    """Return final model feature columns after leakage and non-feature drops."""
    excluded = set(ALWAYS_DROP).union(extra_drop)
    return [col for col in columns if col not in excluded]


def leakage_columns_present(columns: Iterable[str]) -> list[str]:
    """Return leakage columns present in a candidate feature set."""
    column_set = set(columns)
    return [col for col in LEAKAGE_COLS if col in column_set]
