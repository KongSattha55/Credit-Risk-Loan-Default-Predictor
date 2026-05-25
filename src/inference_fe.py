"""
Inference-time feature engineering — single source of truth.

Previously this logic was duplicated in api/main.py and app/main.py. Any drift
between them (and src/feature_engineering.py) silently produced different
scores for the same input. This module is the canonical row-level transform.

Both the FastAPI service and the Streamlit app import `apply_feature_engineering`.
"""

from __future__ import annotations

import numpy as np

# Columns dropped from features before model training / scoring. Shared across
# train / tune / threshold / explain / app to avoid drift.
TARGET = "default"
ALWAYS_DROP = (
    TARGET,
    "funded_amnt",
    "funded_amnt_inv",
    "fico_range_high",
    "emp_title",
)

# Log1p columns must stay in sync with src/feature_engineering.LOG_TRANSFORM_COLS.
_LOG_COLS = (
    "annual_inc", "revol_bal", "tot_coll_amt", "delinq_amnt",
    "total_rev_hi_lim", "tot_hi_cred_lim", "total_bal_ex_mort",
    "total_bc_limit", "total_il_high_credit_limit",
)


def _safe_div(a: float, b: float, fill: float = 0.0) -> float:
    """Element-wise division safe against division-by-zero, NaN, inf."""
    try:
        result = a / b
    except ZeroDivisionError:
        return fill
    if result != result or abs(result) == float("inf"):
        return fill
    return result


def apply_feature_engineering(raw: dict, maps: dict) -> dict:
    """
    Replicate src/feature_engineering.py transforms on a single-row input dict.
    Returns a new dict with engineered features.

    maps: contents of mlruns/artifacts/encoding_maps.json.
    Missing keys (e.g. emp_title_log_freq after pruning) are tolerated; the
    corresponding feature is skipped and the model input dict will simply not
    contain it — callers align to `feature_columns` from model metadata.
    """
    r = dict(raw)

    # ── Ordinal encodings ─────────────────────────────────────────────────
    r["grade_enc"]               = float(maps["grade"].get(r.get("grade", ""), 4))
    r["sub_grade_enc"]           = float(maps["sub_grade"].get(r.get("sub_grade", ""), 18))
    r["verification_status_enc"] = float(maps["verification_status"].get(r.get("verification_status", ""), 0))
    r["home_ownership_enc"]      = float(maps["home_ownership"].get(r.get("home_ownership", ""), 0))

    # ── Binary flags ──────────────────────────────────────────────────────
    r["term_60"]                 = float(r.get("term", 36) == 60)
    r["initial_list_status_enc"] = float(r.get("initial_list_status", "w") == "w")
    r["application_type_joint"]  = float(r.get("application_type", "Individual") == "Joint App")
    r["disbursement_direct"]     = float(r.get("disbursement_method", "Cash") == "DirectPay")

    # ── Target encodings ──────────────────────────────────────────────────
    if "purpose_rate_enc" in maps:
        r["purpose_rate_enc"] = float(
            maps["purpose_rate_enc"].get(r.get("purpose", ""), maps.get("purpose_rate_enc_fallback", 0.2))
        )
    if "state_default_rate" in maps:
        r["state_default_rate"] = float(
            maps["state_default_rate"].get(r.get("addr_state", ""), maps.get("state_default_rate_fallback", 0.2))
        )
    if "emp_title_log_freq" in maps:
        r["emp_title_log_freq"] = float(
            maps["emp_title_log_freq"].get(r.get("emp_title", ""),
                                            maps.get("emp_title_log_freq_fallback", 0.0))
        )

    # ── Log1p transforms ──────────────────────────────────────────────────
    for col in _LOG_COLS:
        r[f"{col}_log"] = float(np.log1p(max(r.get(col, 0.0), 0.0)))

    # ── Ratio / interaction features ──────────────────────────────────────
    r["loan_to_income"]        = _safe_div(r.get("loan_amnt", 0),      r.get("annual_inc", 1))
    r["installment_to_income"] = _safe_div(r.get("installment", 0),    r.get("annual_inc", 12) / 12)
    r["credit_util_total"]     = _safe_div(r.get("revol_bal", 0),      r.get("tot_hi_cred_lim", 0) + 1)
    bc_used = max(r.get("total_bc_limit", 0) - r.get("bc_open_to_buy", 0), 0)
    r["bc_util_ratio"]         = _safe_div(bc_used,                     r.get("total_bc_limit", 0) + 1)
    r["int_rate_x_term"]       = float(r.get("int_rate", 0) * r.get("term", 36))
    r["fico_dti_score"]        = _safe_div(r.get("fico_range_low", 0), r.get("dti", 0) + 1)
    derogs = r.get("pub_rec", 0) + r.get("collections_12_mths_ex_med", 0)
    r["derog_ratio"]           = _safe_div(derogs,                      r.get("total_acc", 0) + 1)
    r["inq_per_acc"]           = _safe_div(r.get("inq_last_6mths", 0), r.get("open_acc", 0) + 1)
    r["revolving_debt_share"]  = _safe_div(r.get("revol_bal", 0),      r.get("total_bal_ex_mort", 0) + 1)

    return r


def risk_label(prob: float) -> str:
    if prob < 0.15:  return "Low"
    if prob < 0.30:  return "Medium"
    if prob < 0.50:  return "High"
    return "Very High"
