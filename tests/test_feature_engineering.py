"""
Unit tests for src/feature_engineering.py.
Run: pytest tests/test_feature_engineering.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    GRADE_MAP,
    SUBGRADE_MAP,
    VERIFICATION_MAP,
    HOME_OWNERSHIP_MAP,
    LOG_TRANSFORM_COLS,
    COLLINEAR_DROPS,
    RAW_CAT_DROPS,
    _safe_div,
    _smoothed_target_rate,
    engineer_features,
)


# ══════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════

def _make_df(**overrides) -> pd.DataFrame:
    """Minimal cleaned DataFrame covering all engineering branches."""
    base = {
        # Loan terms
        "loan_amnt":       [10000.0],
        "funded_amnt":     [10000.0],   # collinear → dropped
        "funded_amnt_inv": [10000.0],   # collinear → dropped
        "term":            [36.0],
        "int_rate":        [12.5],
        "installment":     [335.0],
        # Grade
        "grade":           ["B"],
        "sub_grade":       ["B3"],
        # Borrower
        "emp_length":      [5.0],
        "emp_title":       ["Manager"],
        "home_ownership":  ["RENT"],
        "annual_inc":      [60000.0],
        "verification_status": ["Verified"],
        "purpose":         ["debt_consolidation"],
        "addr_state":      ["CA"],
        # Credit
        "dti":             [15.0],
        "delinq_2yrs":     [0.0],
        "fico_range_low":  [700.0],
        "fico_range_high": [704.0],   # collinear → dropped
        "inq_last_6mths":  [1.0],
        "open_acc":        [8.0],
        "pub_rec":         [0.0],
        "revol_bal":       [5000.0],
        "revol_util":      [30.0],
        "total_acc":       [20.0],
        "initial_list_status": ["w"],
        "application_type":    ["Individual"],
        "disbursement_method": ["Cash"],
        # loan_age_months is in LEAKAGE_COLS and must not appear in feature
        # engineering input; data_cleaning.py no longer creates it.
        "cr_history_months":   [120.0],
        # Skewed / bureau
        "tot_coll_amt":    [0.0],
        "delinq_amnt":     [0.0],
        "total_rev_hi_lim":[10000.0],
        "tot_hi_cred_lim": [50000.0],
        "total_bal_ex_mort":[5000.0],
        "total_bc_limit":  [5000.0],
        "total_il_high_credit_limit": [10000.0],
        "bc_open_to_buy":  [2000.0],
        "collections_12_mths_ex_med": [0.0],
        "bc_util":         [40.0],
        # Target
        "default":         [0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


@pytest.fixture(scope="module")
def df_out():
    df, _ = engineer_features(_make_df())
    return df


# ══════════════════════════════════════════════════════════════════════════
# Encoding maps
# ══════════════════════════════════════════════════════════════════════════

class TestEncodingMaps:
    def test_grade_map_covers_all_grades(self):
        assert set(GRADE_MAP.keys()) == set("ABCDEFG")

    def test_grade_map_is_strictly_ordered(self):
        grades = list("ABCDEFG")
        for i in range(len(grades) - 1):
            assert GRADE_MAP[grades[i]] < GRADE_MAP[grades[i + 1]]

    def test_subgrade_map_has_35_entries(self):
        assert len(SUBGRADE_MAP) == 35

    def test_subgrade_a1_is_1(self):
        assert SUBGRADE_MAP["A1"] == 1

    def test_subgrade_g5_is_35(self):
        assert SUBGRADE_MAP["G5"] == 35

    def test_subgrade_consecutive_within_grade(self):
        assert SUBGRADE_MAP["B1"] == SUBGRADE_MAP["A5"] + 1
        assert SUBGRADE_MAP["C1"] == SUBGRADE_MAP["B5"] + 1

    def test_verification_map_has_3_levels(self):
        assert len(VERIFICATION_MAP) == 3
        assert VERIFICATION_MAP["Not Verified"] < VERIFICATION_MAP["Source Verified"]
        assert VERIFICATION_MAP["Source Verified"] < VERIFICATION_MAP["Verified"]

    def test_home_ownership_mortgage_highest(self):
        assert HOME_OWNERSHIP_MAP["MORTGAGE"] == max(HOME_OWNERSHIP_MAP.values())

    def test_home_ownership_other_is_zero(self):
        assert HOME_OWNERSHIP_MAP["OTHER"] == 0
        assert HOME_OWNERSHIP_MAP["NONE"]  == 0


# ══════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════

class TestSafeDiv:
    def test_normal_division(self):
        a = pd.Series([10.0, 20.0])
        b = pd.Series([2.0,  4.0])
        result = _safe_div(a, b)
        assert list(result) == pytest.approx([5.0, 5.0])

    def test_division_by_zero_fills_zero(self):
        a = pd.Series([1.0])
        b = pd.Series([0.0])
        assert _safe_div(a, b).iloc[0] == 0.0

    def test_custom_fill_value(self):
        a = pd.Series([1.0])
        b = pd.Series([0.0])
        assert _safe_div(a, b, fill=-1.0).iloc[0] == -1.0

    def test_no_inf_in_output(self):
        a = pd.Series([1.0, 0.0, 5.0])
        b = pd.Series([0.0, 0.0, 2.0])
        result = _safe_div(a, b)
        assert not np.isinf(result).any()


class TestSmoothedTargetRate:
    def test_returns_float32(self):
        s = pd.Series(["A", "B", "A", "B"])
        y = pd.Series([1, 0, 1, 0])
        result = _smoothed_target_rate(s, y, k=1.0, global_rate=0.5)
        assert result.dtype == "float32"

    def test_rare_group_pulled_toward_global_rate(self):
        """A group with 1 observation should be heavily smoothed toward global rate."""
        # 999 A's defaulting at 10%, 1 B defaulting at 100%
        s = pd.Series(["A"] * 999 + ["B"] * 1)
        y = pd.Series([0] * 900 + [1] * 99 + [1])  # A ~ 9.9%, B = 100%
        global_rate = y.mean()
        result = _smoothed_target_rate(s, y, k=500, global_rate=global_rate)
        # B's smoothed rate must be between its raw rate and global_rate
        b_rate = float(result[result.index == 999].iloc[0])
        assert global_rate * 0.5 < b_rate < 1.0


# ══════════════════════════════════════════════════════════════════════════
# Step 1 — Grade encoding
# ══════════════════════════════════════════════════════════════════════════

class TestGradeEncoding:
    def test_grade_enc_present(self, df_out):
        assert "grade_enc" in df_out.columns

    def test_sub_grade_enc_present(self, df_out):
        assert "sub_grade_enc" in df_out.columns

    def test_grade_raw_dropped(self, df_out):
        assert "grade" not in df_out.columns

    def test_sub_grade_raw_dropped(self, df_out):
        assert "sub_grade" not in df_out.columns

    def test_grade_enc_value_for_B(self, df_out):
        assert df_out["grade_enc"].iloc[0] == GRADE_MAP["B"]

    def test_sub_grade_enc_value_for_B3(self, df_out):
        assert df_out["sub_grade_enc"].iloc[0] == SUBGRADE_MAP["B3"]

    def test_grade_enc_dtype(self, df_out):
        assert df_out["grade_enc"].dtype == "float32"


# ══════════════════════════════════════════════════════════════════════════
# Step 2 — Low-cardinality categorical encoding
# ══════════════════════════════════════════════════════════════════════════

class TestLowCardinalityEncoding:
    def test_verification_status_enc_present(self, df_out):
        assert "verification_status_enc" in df_out.columns

    def test_verification_verified_maps_to_2(self, df_out):
        assert df_out["verification_status_enc"].iloc[0] == VERIFICATION_MAP["Verified"]

    def test_home_ownership_enc_present(self, df_out):
        assert "home_ownership_enc" in df_out.columns

    def test_home_rent_maps_to_1(self, df_out):
        assert df_out["home_ownership_enc"].iloc[0] == HOME_OWNERSHIP_MAP["RENT"]

    def test_purpose_rate_enc_present(self, df_out):
        assert "purpose_rate_enc" in df_out.columns

    def test_purpose_rate_in_unit_interval(self, df_out):
        rate = df_out["purpose_rate_enc"].iloc[0]
        assert 0.0 <= rate <= 1.0

    def test_purpose_rate_reflects_default_rate(self):
        """High-default purpose group gets a higher smoothed rate than a zero-default group."""
        rows = pd.concat(
            [_make_df(purpose=["credit_card"],    default=[1])] * 20 +
            [_make_df(purpose=["home_improvement"], default=[0])] * 20,
            ignore_index=True,
        )
        out, _ = engineer_features(rows)
        cc_rate  = out.loc[out["purpose_rate_enc"] == out.loc[:19, "purpose_rate_enc"].iloc[0],
                           "purpose_rate_enc"].iloc[0]
        hi_rate  = out.loc[20:, "purpose_rate_enc"].iloc[0]
        assert cc_rate > hi_rate

    def test_term_60_is_zero_for_36_month(self, df_out):
        assert df_out["term_60"].iloc[0] == 0.0

    def test_term_60_is_one_for_60_month(self):
        out, _ = engineer_features(_make_df(term=[60.0]))
        assert out["term_60"].iloc[0] == 1.0

    def test_initial_list_status_enc_w_is_1(self, df_out):
        assert df_out["initial_list_status_enc"].iloc[0] == 1.0

    def test_application_type_joint_individual_is_0(self, df_out):
        assert df_out["application_type_joint"].iloc[0] == 0.0

    def test_application_type_joint_is_1(self):
        out, _ = engineer_features(_make_df(application_type=["Joint App"]))
        assert out["application_type_joint"].iloc[0] == 1.0

    def test_disbursement_direct_cash_is_0(self, df_out):
        assert df_out["disbursement_direct"].iloc[0] == 0.0

    def test_disbursement_direct_directpay_is_1(self):
        out, _ = engineer_features(_make_df(disbursement_method=["DirectPay"]))
        assert out["disbursement_direct"].iloc[0] == 1.0

    def test_raw_categoricals_dropped(self, df_out):
        for col in ["verification_status", "home_ownership", "purpose",
                    "initial_list_status", "application_type", "disbursement_method", "term"]:
            assert col not in df_out.columns, f"Raw column '{col}' was not dropped"


# ══════════════════════════════════════════════════════════════════════════
# Step 3 — High-cardinality encoding
# ══════════════════════════════════════════════════════════════════════════

class TestHighCardinalityEncoding:
    def test_state_default_rate_present(self, df_out):
        assert "state_default_rate" in df_out.columns

    def test_state_default_rate_in_unit_interval(self, df_out):
        rate = df_out["state_default_rate"].iloc[0]
        assert 0.0 <= rate <= 1.0

    def test_emp_title_log_freq_present(self, df_out):
        assert "emp_title_log_freq" in df_out.columns

    def test_emp_title_log_freq_is_non_negative(self, df_out):
        assert df_out["emp_title_log_freq"].iloc[0] >= 0.0

    def test_addr_state_raw_dropped(self, df_out):
        assert "addr_state" not in df_out.columns

    def test_emp_title_raw_dropped(self, df_out):
        assert "emp_title" not in df_out.columns

    def test_emp_title_freq_intermediate_dropped(self, df_out):
        assert "emp_title_freq" not in df_out.columns


# ══════════════════════════════════════════════════════════════════════════
# Step 4 — Log transforms
# ══════════════════════════════════════════════════════════════════════════

class TestLogTransforms:
    def test_log_columns_created(self, df_out):
        for col in LOG_TRANSFORM_COLS:
            if col in _make_df().columns:
                assert f"{col}_log" in df_out.columns, f"Missing {col}_log"

    def test_original_columns_kept(self, df_out):
        """Originals are kept alongside log versions for model selection."""
        for col in ["annual_inc", "revol_bal"]:
            assert col in df_out.columns, f"Original '{col}' was removed — should be kept"

    def test_log_annual_inc_value(self, df_out):
        import math
        expected = math.log1p(60000.0)
        assert pytest.approx(float(df_out["annual_inc_log"].iloc[0]), rel=1e-4) == expected

    def test_log_zero_gives_zero(self):
        out, _ = engineer_features(_make_df(tot_coll_amt=[0.0]))
        assert out["tot_coll_amt_log"].iloc[0] == pytest.approx(0.0)

    def test_log_columns_non_negative(self, df_out):
        log_cols = [c for c in df_out.columns if c.endswith("_log")]
        for col in log_cols:
            assert df_out[col].min() >= 0.0, f"{col} has negative values after log transform"

    def test_log_columns_dtype_float32(self, df_out):
        log_cols = [c for c in df_out.columns if c.endswith("_log")]
        for col in log_cols:
            assert df_out[col].dtype == "float32", f"{col} is not float32"


# ══════════════════════════════════════════════════════════════════════════
# Step 5 — Ratio features
# ══════════════════════════════════════════════════════════════════════════

class TestRatioFeatures:
    def test_loan_to_income_present(self, df_out):
        assert "loan_to_income" in df_out.columns

    def test_loan_to_income_value(self, df_out):
        # 10000 / 60000 ≈ 0.1667
        assert df_out["loan_to_income"].iloc[0] == pytest.approx(10000 / 60000, rel=1e-3)

    def test_installment_to_income_present(self, df_out):
        assert "installment_to_income" in df_out.columns

    def test_installment_to_income_value(self, df_out):
        # 335 / (60000/12) = 335 / 5000 = 0.067
        assert df_out["installment_to_income"].iloc[0] == pytest.approx(335 / 5000, rel=1e-2)

    def test_credit_util_total_present(self, df_out):
        assert "credit_util_total" in df_out.columns

    def test_bc_util_ratio_present(self, df_out):
        assert "bc_util_ratio" in df_out.columns

    def test_bc_util_ratio_value(self, df_out):
        # (5000 - 2000) / (5000 + 1) ≈ 0.5999
        expected = 3000 / 5001
        assert df_out["bc_util_ratio"].iloc[0] == pytest.approx(expected, rel=1e-2)

    def test_int_rate_x_term_present(self, df_out):
        assert "int_rate_x_term" in df_out.columns

    def test_int_rate_x_term_value(self, df_out):
        # 12.5 × 36 = 450
        assert df_out["int_rate_x_term"].iloc[0] == pytest.approx(450.0, rel=1e-3)

    def test_fico_dti_score_present(self, df_out):
        assert "fico_dti_score" in df_out.columns

    def test_fico_dti_score_value(self, df_out):
        # 700 / (15 + 1) = 43.75
        assert df_out["fico_dti_score"].iloc[0] == pytest.approx(700 / 16, rel=1e-3)

    def test_derog_ratio_present(self, df_out):
        assert "derog_ratio" in df_out.columns

    def test_derog_ratio_zero_for_clean_borrower(self, df_out):
        # pub_rec=0, collections=0 → derog_ratio = 0 / (20+1) = 0
        assert df_out["derog_ratio"].iloc[0] == pytest.approx(0.0)

    def test_inq_per_acc_present(self, df_out):
        assert "inq_per_acc" in df_out.columns

    def test_inq_per_acc_value(self, df_out):
        # 1 / (8 + 1) ≈ 0.111
        assert df_out["inq_per_acc"].iloc[0] == pytest.approx(1 / 9, rel=1e-3)

    def test_revolving_debt_share_present(self, df_out):
        assert "revolving_debt_share" in df_out.columns

    def test_revolving_debt_share_value(self, df_out):
        # 5000 / (5000 + 1) ≈ 0.9998
        assert df_out["revolving_debt_share"].iloc[0] == pytest.approx(5000 / 5001, rel=1e-3)

    def test_no_inf_in_ratio_features(self, df_out):
        ratio_cols = ["loan_to_income", "installment_to_income", "credit_util_total",
                      "bc_util_ratio", "int_rate_x_term", "fico_dti_score",
                      "derog_ratio", "inq_per_acc", "revolving_debt_share"]
        for col in ratio_cols:
            if col in df_out.columns:
                assert not np.isinf(df_out[col]).any(), f"{col} contains inf"


# ══════════════════════════════════════════════════════════════════════════
# Step 6 — Drop redundant columns
# ══════════════════════════════════════════════════════════════════════════

class TestDropRedundant:
    def test_funded_amnt_dropped(self, df_out):
        assert "funded_amnt" not in df_out.columns

    def test_funded_amnt_inv_dropped(self, df_out):
        assert "funded_amnt_inv" not in df_out.columns

    def test_fico_range_high_dropped(self, df_out):
        assert "fico_range_high" not in df_out.columns

    def test_raw_grade_dropped(self, df_out):
        assert "grade" not in df_out.columns

    def test_raw_addr_state_dropped(self, df_out):
        assert "addr_state" not in df_out.columns


# ══════════════════════════════════════════════════════════════════════════
# Step 7 — Validation
# ══════════════════════════════════════════════════════════════════════════

class TestValidation:
    def test_no_nulls_in_output(self, df_out):
        assert df_out.isnull().sum().sum() == 0

    def test_no_string_columns_in_output(self, df_out):
        str_cols = df_out.select_dtypes(include=["object", "category"]).columns.tolist()
        assert str_cols == [], f"String columns remain: {str_cols}"

    def test_no_inf_values_in_output(self, df_out):
        inf_count = np.isinf(df_out.select_dtypes(include="number")).sum().sum()
        assert inf_count == 0

    def test_target_column_preserved(self, df_out):
        assert "default" in df_out.columns
        assert set(df_out["default"].unique()).issubset({0, 1})

    def test_default_rate_unchanged(self):
        df_in      = _make_df(default=[1])
        df_out, _  = engineer_features(df_in)
        assert df_out["default"].iloc[0] == 1

    def test_all_columns_numeric(self, df_out):
        non_numeric = df_out.select_dtypes(exclude="number").columns.tolist()
        assert non_numeric == [], f"Non-numeric columns: {non_numeric}"


# ══════════════════════════════════════════════════════════════════════════
# Multi-row stability
# ══════════════════════════════════════════════════════════════════════════

class TestMultiRow:
    def test_runs_on_multi_row_dataframe(self):
        rows = pd.concat([_make_df(grade=["A"]),
                          _make_df(grade=["C"]),
                          _make_df(grade=["G"], dti=[50.0])],
                         ignore_index=True)
        out, _ = engineer_features(rows)
        assert len(out) == 3
        assert out["grade_enc"].tolist() == [
            GRADE_MAP["A"], GRADE_MAP["C"], GRADE_MAP["G"]
        ]

    def test_grade_enc_monotone_with_risk(self):
        """Higher grade letter → higher grade_enc → confirmed higher default rate."""
        a_enc = GRADE_MAP["A"]
        g_enc = GRADE_MAP["G"]
        assert a_enc < g_enc

    def test_high_risk_loan_has_higher_loan_to_income(self):
        low,  _ = engineer_features(_make_df(loan_amnt=[5000.0],  annual_inc=[100000.0]))
        high, _ = engineer_features(_make_df(loan_amnt=[30000.0], annual_inc=[30000.0]))
        assert high["loan_to_income"].iloc[0] > low["loan_to_income"].iloc[0]
