import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.explain_shap import load_metadata, load_test_sample
from src.explain_utils import explain_feature_direction, summarize_shap_local
from src.leakage import leakage_columns_present


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
META_PATH = PROJECT_ROOT / "mlruns" / "artifacts" / "lightgbm_metadata.json"
DATA_EXISTS = any(
    path.exists()
    for path in (
        PROJECT_ROOT / "data" / "interim" / "loans_features.parquet",
        PROJECT_ROOT / "data" / "processed" / "loans_cleaned.parquet",
    )
)
SHAP_ARTIFACTS_EXIST = all(
    (ARTIFACTS_DIR / name).exists()
    for name in ("shap_top_features.csv", "shap_local_example.json")
)

requires_model_data = pytest.mark.skipif(
    not (META_PATH.exists() and DATA_EXISTS),
    reason="Model metadata or feature data not found — run training and feature engineering first",
)
requires_shap_artifacts = pytest.mark.skipif(
    not (META_PATH.exists() and SHAP_ARTIFACTS_EXIST),
    reason="SHAP artifacts not found — run src/explain_shap.py first",
)


@requires_model_data
def test_shap_feature_names_match_model_feature_names():
    metadata = load_metadata()
    X_sample, _ = load_test_sample(sample=5, metadata=metadata)

    assert X_sample.columns.tolist() == metadata["feature_columns"]


@requires_shap_artifacts
def test_no_leakage_features_appear_in_shap_output():
    metadata = load_metadata()
    top_features = pd.read_csv(ARTIFACTS_DIR / "shap_top_features.csv")

    assert leakage_columns_present(metadata["feature_columns"]) == []
    assert leakage_columns_present(top_features["feature"]) == []


@requires_shap_artifacts
def test_shap_top_features_csv_is_created():
    path = ARTIFACTS_DIR / "shap_top_features.csv"
    df = pd.read_csv(path)

    assert path.exists()
    assert {"feature", "mean_abs_shap", "rank", "direction_note", "business_interpretation"}.issubset(df.columns)
    assert len(df) > 0


@requires_shap_artifacts
def test_local_explanation_json_has_required_fields():
    path = ARTIFACTS_DIR / "shap_local_example.json"
    payload = json.loads(path.read_text())

    assert "predicted_pd" in payload
    assert "base_value" in payload
    assert "top_positive_drivers" in payload
    assert "top_negative_drivers" in payload
    assert isinstance(payload["top_positive_drivers"], list)
    assert isinstance(payload["top_negative_drivers"], list)


def test_explanation_helper_returns_readable_text():
    summary = summarize_shap_local(
        shap_values=np.array([0.4, -0.2, 0.1]),
        feature_values=np.array([24.5, 720, 1]),
        feature_names=["int_rate", "fico_range_low", "term_60"],
        top_n=2,
    )
    sentence = explain_feature_direction("dti", 35, 0.3)

    assert "higher risk" in summary["explanation_text"]
    assert "Risk is partially reduced" in summary["explanation_text"]
    assert "Interest rate" in summary["top_positive_drivers"][0]["explanation"]
    assert "Debt-to-income ratio" in sentence
    assert "increases predicted default risk" in sentence
