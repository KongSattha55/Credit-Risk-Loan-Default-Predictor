"""
Integration tests for api/main.py using FastAPI TestClient.
Requires the calibrated model to exist at models/lightgbm_calibrated_sigmoid.pkl.

Run: pytest tests/test_api.py -v
Skip if model not present:
    pytest tests/test_api.py -v -m "not requires_model"
"""

import pytest
import json
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATED_MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_calibrated_sigmoid.pkl"
META_PATH = PROJECT_ROOT / "mlruns" / "artifacts" / "lightgbm_metadata.json"
MODEL_EXISTS = CALIBRATED_MODEL_PATH.exists()

pytestmark = pytest.mark.skipif(
    not MODEL_EXISTS,
    reason="Calibrated model not found — run src/calibrate.py first",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


VALID_PAYLOAD = {
    "loan_amnt": 10000.0,
    "term": 36,
    "int_rate": 12.5,
    "installment": 334.87,
    "grade": "B",
    "sub_grade": "B3",
    "emp_length": 5.0,
    "home_ownership": "RENT",
    "annual_inc": 60000.0,
    "verification_status": "Verified",
    "purpose": "debt_consolidation",
    "addr_state": "CA",
    "dti": 15.0,
    "delinq_2yrs": 0.0,
    "fico_range_low": 700.0,
    "inq_last_6mths": 1.0,
    "open_acc": 8.0,
    "pub_rec": 0.0,
    "revol_bal": 5000.0,
    "revol_util": 30.0,
    "total_acc": 20.0,
    "initial_list_status": "w",
    "application_type": "Individual",
    "disbursement_method": "Cash",
    "cr_history_months": 120.0,
}


class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_has_roc_auc(self, client):
        data = client.get("/health").json()
        assert "test_roc_auc" in data
        assert isinstance(data["test_roc_auc"], float)

    def test_health_returns_503_when_model_is_unavailable(self, client, monkeypatch):
        from api import main

        def unavailable_model():
            raise RuntimeError("missing model")

        monkeypatch.setattr(main, "load_model", unavailable_model)
        resp = client.get("/health")

        assert resp.status_code == 503
        assert resp.json()["detail"] == "missing model"


class TestModelInfo:
    def test_model_info_returns_200(self, client):
        resp = client.get("/model-info")
        assert resp.status_code == 200

    def test_model_info_has_metrics(self, client):
        data = client.get("/model-info").json()
        assert "metrics" in data
        assert "test_roc_auc" in data["metrics"]

    def test_model_info_has_n_features(self, client):
        data = client.get("/model-info").json()
        assert data["n_features"] > 0

    def test_model_info_reports_calibrated_sigmoid(self, client):
        data = client.get("/model-info").json()
        assert data["probability_model"] == "calibrated_sigmoid"
        assert data["model_path"] == "models/lightgbm_calibrated_sigmoid.pkl"


class TestProductionProbabilityPipeline:
    def test_calibrated_model_artifact_exists(self):
        assert CALIBRATED_MODEL_PATH.exists()

    def test_calibrated_model_can_be_unpickled(self):
        from src.calibration_classes import LGBMBoosterWrapper, PreFitCalibratedClassifier  # noqa: F401

        with open(CALIBRATED_MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        assert hasattr(model, "predict_proba")
        assert getattr(model, "method", None) == "sigmoid"

    def test_api_loads_calibrated_sigmoid_model(self):
        from api.main import load_model

        load_model.cache_clear()
        model = load_model()

        assert hasattr(model, "predict_proba")
        assert getattr(model, "method", None) == "sigmoid"

    def test_no_leakage_columns_used_by_api_metadata(self):
        from src.leakage import leakage_columns_present

        meta = json.loads(META_PATH.read_text())
        assert leakage_columns_present(meta["feature_columns"]) == []


class TestPredict:
    def test_predict_returns_200(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_predict_response_fields(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "default_probability" in data
        assert "prediction" in data
        assert "threshold" in data
        assert "risk_label" in data
        assert "model_version" in data

    def test_predict_probability_in_range(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        prob = data["default_probability"]
        assert 0.0 <= prob <= 1.0

    def test_predict_binary_prediction(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert data["prediction"] in (0, 1)

    def test_predict_prediction_consistent_with_threshold(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        prob   = data["default_probability"]
        thresh = data["threshold"]
        pred   = data["prediction"]
        expected_pred = 1 if prob >= thresh else 0
        assert pred == expected_pred

    def test_predict_risk_label_valid(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert data["risk_label"] in ("Low", "Medium", "High", "Very High")

    def test_predict_missing_required_field_returns_422(self, client):
        bad_payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "loan_amnt"}
        resp = client.post("/predict", json=bad_payload)
        assert resp.status_code == 422

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("term", 48),
            ("grade", "Z"),
            ("sub_grade", "B9"),
        ],
    )
    def test_predict_rejects_invalid_credit_categories(self, client, field, value):
        resp = client.post("/predict", json={**VALID_PAYLOAD, field: value})
        assert resp.status_code == 422

    def test_predict_high_risk_loan(self, client):
        """A loan with high int_rate, high DTI, low FICO should have higher default prob."""
        high_risk = {**VALID_PAYLOAD, "int_rate": 29.99, "dti": 40.0, "fico_range_low": 580.0}
        low_risk  = {**VALID_PAYLOAD, "int_rate": 6.0,   "dti": 5.0,  "fico_range_low": 780.0}

        prob_high = client.post("/predict", json=high_risk).json()["default_probability"]
        prob_low  = client.post("/predict", json=low_risk).json()["default_probability"]

        assert prob_high > prob_low, (
            f"Expected high-risk ({prob_high:.3f}) > low-risk ({prob_low:.3f})"
        )
