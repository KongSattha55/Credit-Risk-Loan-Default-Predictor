import json
from pathlib import Path

import pandas as pd
import pytest

from src.leakage import leakage_columns_present


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "artifacts" / "model_comparison.csv"
JSON_PATH = PROJECT_ROOT / "artifacts" / "model_comparison.json"

pytestmark = pytest.mark.skipif(
    not (CSV_PATH.exists() and JSON_PATH.exists()),
    reason="Model comparison artifacts not found — run src/model_comparison.py first",
)

REQUIRED_COLUMNS = {
    "model",
    "roc_auc",
    "pr_auc",
    "average_precision",
    "brier_score",
    "log_loss",
    "f1",
    "training_time_seconds",
}


def test_model_comparison_outputs_exist():
    assert CSV_PATH.exists()
    assert JSON_PATH.exists()


def test_model_comparison_required_columns_exist():
    df = pd.read_csv(CSV_PATH)

    assert REQUIRED_COLUMNS.issubset(df.columns)


def test_model_comparison_uses_no_leakage_columns():
    payload = json.loads(JSON_PATH.read_text())

    assert payload["leakage_columns_present"] == []
    assert leakage_columns_present(payload["feature_columns"]) == []


def test_model_comparison_metrics_are_valid_ranges():
    df = pd.read_csv(CSV_PATH)

    for col in ["roc_auc", "pr_auc", "average_precision", "brier_score", "f1"]:
        assert ((df[col] >= 0) & (df[col] <= 1)).all(), f"{col} outside [0, 1]"

    assert (df["log_loss"] >= 0).all()
    assert (df["training_time_seconds"] >= 0).all()


def test_model_comparison_includes_all_required_models():
    df = pd.read_csv(CSV_PATH)
    models = set(df["model"])

    assert {
        "FICO/grade logistic baseline",
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "LightGBM",
        "Calibrated LightGBM sigmoid",
    }.issubset(models)
