"""Business-friendly helpers for SHAP explanations."""

from __future__ import annotations

from typing import Any

import numpy as np


_FEATURE_LABELS = {
    "int_rate": "Interest rate",
    "fico_range_low": "FICO score",
    "dti": "Debt-to-income ratio",
    "term_60": "60-month loan term",
    "annual_inc": "Annual income",
    "annual_inc_log": "Annual income",
    "cr_history_months": "Credit history length",
    "loan_to_income": "Loan amount compared with income",
    "installment_to_income": "Monthly payment burden",
    "revol_util": "Revolving credit utilization",
    "bc_util": "Bankcard utilization",
    "bc_util_ratio": "Bankcard utilization",
    "grade_enc": "LendingClub grade",
    "sub_grade_enc": "LendingClub sub-grade",
    "purpose_rate_enc": "Loan purpose risk pattern",
    "state_default_rate": "State risk pattern",
    "inq_last_6mths": "Recent credit inquiries",
    "inq_per_acc": "Inquiries per open account",
    "open_acc": "Open credit accounts",
    "mort_acc": "Mortgage account count",
    "verification_status_enc": "Income verification status",
    "home_ownership_enc": "Home ownership",
    "int_rate_x_term": "Interest rate and term combination",
    "fico_dti_score": "Credit score compared with debt burden",
    "acc_open_past_24mths": "Recently opened accounts",
    "total_bc_limit": "Total bankcard credit limit",
    "total_il_high_credit_limit": "Total installment credit limit",
    "emp_title_log_freq": "Employment-title stability signal",
    "mo_sin_rcnt_rev_tl_op": "Months since recent revolving account opened",
}

_VALUE_HINTS = {
    "int_rate": ("high", "low"),
    "dti": ("high", "low"),
    "loan_to_income": ("high", "low"),
    "installment_to_income": ("high", "low"),
    "revol_util": ("high", "low"),
    "bc_util": ("high", "low"),
    "bc_util_ratio": ("high", "low"),
    "inq_last_6mths": ("high", "low"),
    "inq_per_acc": ("high", "low"),
    "fico_range_low": ("relatively low", "strong"),
    "annual_inc": ("lower", "high"),
    "annual_inc_log": ("lower", "high"),
    "cr_history_months": ("shorter", "longer"),
    "term_60": ("60 months", "not 60 months"),
    "fico_dti_score": ("weaker compared with debt burden", "strong compared with debt burden"),
    "acc_open_past_24mths": ("high", "low"),
    "total_bc_limit": ("lower", "higher"),
    "total_il_high_credit_limit": ("lower", "higher"),
    "mo_sin_rcnt_rev_tl_op": ("recent", "less recent"),
}


def _feature_label(feature_name: str) -> str:
    return _FEATURE_LABELS.get(feature_name, feature_name.replace("_", " "))


def _format_value(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(val) >= 1000:
        return f"{val:,.0f}"
    if abs(val) >= 10:
        return f"{val:.1f}"
    return f"{val:.3g}"


def explain_feature_direction(feature_name: str, value: Any, shap_value: float) -> str:
    """Return one readable sentence for a local feature contribution."""
    label = _feature_label(feature_name)
    value_text = _format_value(value)
    increases_risk = float(shap_value) > 0
    high_hint, low_hint = _VALUE_HINTS.get(feature_name, ("notable", "favorable"))
    descriptor = high_hint if increases_risk else low_hint
    direction = "increases" if increases_risk else "reduces"
    return f"{label} is {descriptor} ({value_text}), which {direction} predicted default risk."


def summarize_shap_local(
    shap_values,
    feature_values,
    feature_names,
    top_n: int = 5,
) -> dict[str, Any]:
    """Summarize positive and negative local drivers in business language."""
    shap_arr = np.asarray(shap_values, dtype=float)
    value_arr = np.asarray(feature_values)
    names = list(feature_names)

    if shap_arr.ndim != 1:
        raise ValueError("shap_values must be a one-dimensional array")
    if len(shap_arr) != len(names) or len(value_arr) != len(names):
        raise ValueError("shap_values, feature_values, and feature_names must have the same length")

    positive_idx = [i for i in np.argsort(shap_arr)[::-1] if shap_arr[i] > 0][:top_n]
    negative_idx = [i for i in np.argsort(shap_arr) if shap_arr[i] < 0][:top_n]

    def _driver(i: int) -> dict[str, Any]:
        return {
            "feature": names[i],
            "value": float(value_arr[i]) if np.issubdtype(np.asarray([value_arr[i]]).dtype, np.number) else str(value_arr[i]),
            "shap_value": float(shap_arr[i]),
            "explanation": explain_feature_direction(names[i], value_arr[i], shap_arr[i]),
        }

    positives = [_driver(i) for i in positive_idx]
    negatives = [_driver(i) for i in negative_idx]

    lines: list[str] = []
    if positives:
        lines.append("This loan is predicted as higher risk mainly because:")
        lines.extend(f"{i}. {d['explanation']}" for i, d in enumerate(positives, 1))
    if negatives:
        if lines:
            lines.append("")
        lines.append("Risk is partially reduced because:")
        lines.extend(f"{i}. {d['explanation']}" for i, d in enumerate(negatives, 1))

    return {
        "top_positive_drivers": positives,
        "top_negative_drivers": negatives,
        "explanation_text": "\n".join(lines),
    }


def global_business_interpretation(feature_name: str) -> str:
    """Return a stable, interview-friendly interpretation for common features."""
    known = {
        "int_rate": "Higher interest rates usually indicate riskier borrowers and increase predicted default risk.",
        "fico_range_low": "Higher FICO scores indicate stronger credit quality and usually lower predicted default risk.",
        "dti": "Higher debt-to-income ratios indicate more debt burden and usually raise default risk.",
        "term_60": "Longer 60-month loans have a longer repayment horizon and usually increase uncertainty.",
        "annual_inc": "Higher income can improve repayment capacity and reduce default risk.",
        "annual_inc_log": "Higher income can improve repayment capacity and reduce default risk.",
        "cr_history_months": "Longer credit history can indicate more established borrower behavior.",
        "loan_to_income": "A larger loan relative to income can make repayment harder.",
        "installment_to_income": "A higher monthly payment burden can increase default risk.",
        "revol_util": "Higher revolving utilization can indicate credit stress.",
        "grade_enc": "Weaker LendingClub grades indicate higher borrower risk.",
        "sub_grade_enc": "Weaker LendingClub sub-grades indicate higher borrower risk.",
        "purpose_rate_enc": "Some loan purposes historically default at higher rates than others.",
        "state_default_rate": "Borrower geography can capture regional economic or portfolio risk patterns.",
        "int_rate_x_term": "High interest rates combined with longer terms can indicate repayment uncertainty.",
        "acc_open_past_24mths": "Many recently opened accounts can indicate fast credit growth and higher risk.",
        "fico_dti_score": "A stronger credit score relative to debt burden usually lowers default risk.",
        "home_ownership_enc": "Home ownership status can proxy borrower stability and housing obligations.",
        "total_bc_limit": "Higher bankcard limits can indicate available liquidity and established credit access.",
        "total_il_high_credit_limit": "Higher installment credit limits can reflect broader credit access and capacity.",
        "emp_title_log_freq": "Employment-title patterns can capture occupation stability in the historical portfolio.",
        "mo_sin_rcnt_rev_tl_op": "More time since a recent revolving account was opened can indicate less recent credit expansion.",
    }
    return known.get(feature_name, f"{_feature_label(feature_name)} is an important model signal; review its SHAP direction and distribution.")
