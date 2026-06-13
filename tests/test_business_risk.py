import numpy as np
import pytest

from src.business_risk import (
    assign_risk_band,
    calculate_expected_loss,
    threshold_business_table,
)
from src.inference_fe import risk_label


REQUIRED_COLUMNS = {
    "threshold",
    "approval_rate",
    "rejection_rate",
    "default_rate_approved",
    "defaults_caught_rate",
    "expected_loss_approved",
    "expected_loss_rejected",
    "total_expected_loss",
}


def test_expected_loss_calculation_is_correct():
    assert calculate_expected_loss(0.20, lgd=0.45, ead=10_000) == pytest.approx(900.0)


def test_expected_loss_vectorized_calculation_is_correct():
    result = calculate_expected_loss(
        np.array([0.10, 0.25]),
        lgd=0.40,
        ead=np.array([10_000, 20_000]),
    )

    assert np.allclose(result, np.array([400.0, 2_000.0]))


def test_risk_bands_are_assigned_correctly():
    assert assign_risk_band(0.05) == "Low"
    assert assign_risk_band(0.10) == "Medium"
    assert assign_risk_band(0.20) == "High"
    assert assign_risk_band(0.35) == "Very High"


def test_vectorized_risk_bands_are_assigned_correctly():
    bands = assign_risk_band(np.array([0.05, 0.15, 0.25, 0.45]))

    assert bands.tolist() == ["Low", "Medium", "High", "Very High"]


def test_serving_risk_label_uses_business_risk_bands():
    for pd_value in [0.05, 0.10, 0.20, 0.35, 0.80]:
        assert risk_label(pd_value) == assign_risk_band(pd_value)


def test_threshold_table_has_required_columns():
    table = threshold_business_table(
        y_true=np.array([0, 1, 0, 1]),
        y_proba=np.array([0.05, 0.15, 0.30, 0.60]),
        loan_amounts=np.array([10_000, 12_000, 15_000, 20_000]),
        lgd=0.45,
        thresholds=np.array([0.20, 0.40]),
    )

    assert REQUIRED_COLUMNS.issubset(table.columns)


def test_approval_rate_decreases_as_threshold_becomes_stricter():
    table = threshold_business_table(
        y_true=np.array([0, 1, 0, 1]),
        y_proba=np.array([0.05, 0.15, 0.30, 0.60]),
        loan_amounts=np.array([10_000, 12_000, 15_000, 20_000]),
        thresholds=np.array([0.20, 0.50]),
    )

    stricter_approval_rate = table.loc[table["threshold"] == 0.20, "approval_rate"].iloc[0]
    looser_approval_rate = table.loc[table["threshold"] == 0.50, "approval_rate"].iloc[0]

    assert stricter_approval_rate < looser_approval_rate


def test_expected_loss_is_non_negative():
    table = threshold_business_table(
        y_true=np.array([0, 1, 0, 1]),
        y_proba=np.array([0.05, 0.15, 0.30, 0.60]),
        loan_amounts=np.array([10_000, 12_000, 15_000, 20_000]),
        thresholds=np.array([0.20, 0.40]),
    )

    assert (table["expected_loss_approved"] >= 0).all()
    assert (table["expected_loss_rejected"] >= 0).all()
    assert (table["total_expected_loss"] >= 0).all()
