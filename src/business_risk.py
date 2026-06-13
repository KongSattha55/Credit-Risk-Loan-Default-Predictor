"""Business risk analytics for calibrated loan default probabilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_LGD = 0.45
DEFAULT_THRESHOLDS = np.round(np.arange(0.05, 0.501, 0.025), 3)


def calculate_expected_loss(pd, lgd: float = DEFAULT_LGD, ead=None):
    """Calculate Expected Loss = PD * LGD * EAD."""
    pd_values = np.asarray(pd, dtype=float)
    ead_values = 1.0 if ead is None else np.asarray(ead, dtype=float)
    loss = pd_values * float(lgd) * ead_values
    if np.ndim(loss) == 0:
        return float(loss)
    return loss


def assign_risk_band(pd):
    """Assign business risk bands from calibrated probability of default."""
    pd_values = np.asarray(pd, dtype=float)
    bands = np.select(
        [
            pd_values < 0.10,
            (pd_values >= 0.10) & (pd_values < 0.20),
            (pd_values >= 0.20) & (pd_values < 0.35),
            pd_values >= 0.35,
        ],
        ["Low", "Medium", "High", "Very High"],
        default="Unknown",
    )
    if np.ndim(pd_values) == 0:
        return str(bands.item())
    return bands


def threshold_business_table(
    y_true,
    y_proba,
    loan_amounts,
    lgd: float = DEFAULT_LGD,
    thresholds=None,
) -> pd.DataFrame:
    """
    Build a business threshold table from calibrated PDs.

    Loans with PD below the threshold are treated as approved; loans at or
    above the threshold are treated as rejected/reviewed.
    """
    y = np.asarray(y_true, dtype=int)
    proba = np.asarray(y_proba, dtype=float)
    ead = np.asarray(loan_amounts, dtype=float)
    threshold_values = DEFAULT_THRESHOLDS if thresholds is None else np.asarray(thresholds, dtype=float)

    if not (len(y) == len(proba) == len(ead)):
        raise ValueError("y_true, y_proba, and loan_amounts must have the same length")
    if len(y) == 0:
        raise ValueError("threshold_business_table requires at least one row")
    if np.any((proba < 0) | (proba > 1)):
        raise ValueError("y_proba must contain probabilities in [0, 1]")
    if np.any(ead < 0):
        raise ValueError("loan_amounts/EAD must be non-negative")
    if not (0 <= float(lgd) <= 1):
        raise ValueError("lgd must be between 0 and 1")

    expected_loss = calculate_expected_loss(proba, lgd=lgd, ead=ead)
    total_defaults = max(int(y.sum()), 1)
    rows = []

    for threshold in threshold_values:
        approved = proba < threshold
        rejected = ~approved
        n_approved = int(approved.sum())
        n_rejected = int(rejected.sum())

        rows.append(
            {
                "threshold": float(threshold),
                "approval_rate": float(n_approved / len(y)),
                "rejection_rate": float(n_rejected / len(y)),
                "default_rate_approved": float(y[approved].mean()) if n_approved else 0.0,
                "defaults_caught_rate": float(y[rejected].sum() / total_defaults),
                "expected_loss_approved": float(expected_loss[approved].sum()),
                "expected_loss_rejected": float(expected_loss[rejected].sum()),
                "total_expected_loss": float(expected_loss.sum()),
            }
        )

    return pd.DataFrame(rows)
