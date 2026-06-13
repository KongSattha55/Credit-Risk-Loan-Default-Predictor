import pandas as pd
import pytest

from src.leakage import LEAKAGE_COLS, feature_columns, leakage_columns_present


REQUESTED_LEAKAGE_COLS = {
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
    "loan_age_months",   # computed by old data_cleaning.py; must be permanently excluded
}


def test_central_leakage_list_contains_required_columns():
    assert REQUESTED_LEAKAGE_COLS.issubset(set(LEAKAGE_COLS))


def test_final_feature_cols_exclude_leakage_columns():
    columns = ["default", "loan_amnt", "issue_d", "grade", *LEAKAGE_COLS]

    final_feature_cols = feature_columns(columns, extra_drop=["issue_d"])

    assert "loan_amnt" in final_feature_cols
    assert "grade" in final_feature_cols
    assert "issue_d" not in final_feature_cols
    assert leakage_columns_present(final_feature_cols) == []


def test_train_load_data_excludes_leakage_columns(monkeypatch, tmp_path):
    from src import train

    data_path = tmp_path / "loans_features.parquet"
    pd.DataFrame(
        {
            "default": [0, 1],
            "issue_d": pd.to_datetime(["2015-12-01", "2017-01-01"]),
            "loan_amnt": [5_000.0, 12_000.0],
            "grade": ["A", "D"],
            "loan_age_months": [36, 72],    # leakage — must be excluded
            "last_pymnt_amnt": [250.0, 0.0],  # leakage — must be excluded
        }
    ).to_parquet(data_path)

    monkeypatch.setattr(train, "INTERIM_PATH", data_path)
    monkeypatch.setattr(train, "CLEANED_PATH", tmp_path / "missing.parquet")

    X, y, issue_d, source = train.load_data()

    assert source == "interim (feature-engineered)"
    assert y.tolist() == [0, 1]
    assert issue_d.tolist() == pd.to_datetime(["2015-12-01", "2017-01-01"]).tolist()
    assert "loan_amnt" in X.columns
    assert "grade" in X.columns
    assert "issue_d" not in X.columns
    assert "loan_age_months" not in X.columns, \
        "loan_age_months must be excluded from features — it is in LEAKAGE_COLS"
    assert leakage_columns_present(X.columns) == []


def test_loan_age_months_excluded_from_feature_columns():
    """feature_columns() must always exclude loan_age_months."""
    # Simulate a DataFrame that accidentally contains the column
    candidate_cols = [
        "loan_amnt", "int_rate", "grade", "loan_age_months",
        "cr_history_months", "default",
    ]
    result = feature_columns(candidate_cols)
    assert "loan_age_months" not in result, (
        "feature_columns() did not exclude loan_age_months. "
        "Verify it is in LEAKAGE_COLS."
    )
    assert "loan_amnt" in result
    assert "cr_history_months" in result


def test_loan_age_months_in_leakage_cols():
    """loan_age_months must remain in LEAKAGE_COLS permanently."""
    assert "loan_age_months" in set(LEAKAGE_COLS), (
        "loan_age_months was removed from LEAKAGE_COLS. "
        "It must stay there to prevent reintroduction."
    )


def test_feature_columns_excludes_all_requested_leakage():
    """feature_columns() must exclude every column in REQUESTED_LEAKAGE_COLS."""
    # Build a column list that contains all leakage + some safe columns
    safe = ["loan_amnt", "int_rate", "cr_history_months"]
    cols = safe + list(REQUESTED_LEAKAGE_COLS)
    result = feature_columns(cols)
    for leak in REQUESTED_LEAKAGE_COLS:
        assert leak not in result, f"feature_columns() failed to exclude {leak}"
    for s in safe:
        assert s in result
