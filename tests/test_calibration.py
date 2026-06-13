"""
tests/test_calibration.py
─────────────────────────
Tests for the probability calibration pipeline (src/calibrate.py).

Structure
---------
TestLeakageInvariant   — always runs; no pkl artifacts required.
                          Enforces that loan_age_months is fully removed.
TestArtifactsExist     — skipped if calibration pkl files are absent.
TestModelLoads         — skipped if calibration pkl files are absent.
TestPredictProbaOutput — skipped if calibration pkl files are absent.
TestCalibrationResults — skipped if calibration_results.json is absent.

Run:
    pytest tests/test_calibration.py -v
    pytest tests/test_calibration.py -v -k TestLeakageInvariant  # always-run subset
"""

from __future__ import annotations

import inspect
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT))

# Import the calibration classes so that pickle can resolve them
# when loading the .pkl files (pickle uses the class's module path).
from src.calibration_classes import LGBMBoosterWrapper, PreFitCalibratedClassifier  # noqa: F401
MODELS_DIR    = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
META_PATH     = PROJECT_ROOT / "mlruns" / "artifacts" / "tuning_metadata.json"
RESULTS_PATH  = ARTIFACTS_DIR / "calibration_results.json"

ARTIFACTS_EXIST = (
    (MODELS_DIR / "lightgbm_raw.pkl").exists()
    and (MODELS_DIR / "lightgbm_calibrated_sigmoid.pkl").exists()
    and (MODELS_DIR / "lightgbm_calibrated_isotonic.pkl").exists()
    and RESULTS_PATH.exists()
)

# Applied as a class-level decorator on every artifact-dependent class.
# TestLeakageInvariant carries NO skip marker — it always executes.
_skip_without_artifacts = pytest.mark.skipif(
    not ARTIFACTS_EXIST,
    reason="Calibration artifacts not found — run src/calibrate.py first",
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _load_pkl(name: str):
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_results() -> dict:
    return json.loads(RESULTS_PATH.read_text())


def _tiny_X():
    """Return a minimal DataFrame that matches the model's 99 feature columns."""
    import pandas as pd
    meta  = json.loads(META_PATH.read_text())
    feats = meta["feature_columns"]
    return pd.DataFrame([{col: 0.0 for col in feats}])


# ══════════════════════════════════════════════════════════════════════════
# 1. Artifact existence
# ══════════════════════════════════════════════════════════════════════════

@_skip_without_artifacts
class TestArtifactsExist:
    def test_raw_model_pkl_exists(self):
        assert (MODELS_DIR / "lightgbm_raw.pkl").exists()

    def test_sigmoid_model_pkl_exists(self):
        assert (MODELS_DIR / "lightgbm_calibrated_sigmoid.pkl").exists()

    def test_isotonic_model_pkl_exists(self):
        assert (MODELS_DIR / "lightgbm_calibrated_isotonic.pkl").exists()

    def test_calibration_curve_png_exists(self):
        assert (ARTIFACTS_DIR / "calibration_curve.png").exists()

    def test_calibration_results_json_exists(self):
        assert RESULTS_PATH.exists()


# ══════════════════════════════════════════════════════════════════════════
# 2. Model loads without error
# ══════════════════════════════════════════════════════════════════════════

@_skip_without_artifacts
class TestModelLoads:
    def test_raw_model_loads(self):
        obj = _load_pkl("lightgbm_raw")
        assert obj is not None

    def test_sigmoid_model_loads(self):
        obj = _load_pkl("lightgbm_calibrated_sigmoid")
        assert obj is not None

    def test_isotonic_model_loads(self):
        obj = _load_pkl("lightgbm_calibrated_isotonic")
        assert obj is not None

    def test_raw_model_has_predict_proba(self):
        obj = _load_pkl("lightgbm_raw")
        assert hasattr(obj, "predict_proba"), "LGBMBoosterWrapper missing predict_proba"

    def test_sigmoid_model_has_predict_proba(self):
        obj = _load_pkl("lightgbm_calibrated_sigmoid")
        assert hasattr(obj, "predict_proba")

    def test_isotonic_model_has_predict_proba(self):
        obj = _load_pkl("lightgbm_calibrated_isotonic")
        assert hasattr(obj, "predict_proba")

    def test_raw_model_has_classes(self):
        obj = _load_pkl("lightgbm_raw")
        assert hasattr(obj, "classes_")
        assert list(obj.classes_) == [0, 1]


# ══════════════════════════════════════════════════════════════════════════
# 3. predict_proba returns valid probabilities
# ══════════════════════════════════════════════════════════════════════════

@_skip_without_artifacts
class TestPredictProbaOutput:
    @pytest.fixture(scope="class")
    def sample_X(self):
        return _tiny_X()

    def test_raw_proba_shape(self, sample_X):
        obj    = _load_pkl("lightgbm_raw")
        result = obj.predict_proba(sample_X)
        assert result.shape == (1, 2)

    def test_sigmoid_proba_shape(self, sample_X):
        obj    = _load_pkl("lightgbm_calibrated_sigmoid")
        result = obj.predict_proba(sample_X)
        assert result.shape == (1, 2)

    def test_isotonic_proba_shape(self, sample_X):
        obj    = _load_pkl("lightgbm_calibrated_isotonic")
        result = obj.predict_proba(sample_X)
        assert result.shape == (1, 2)

    def test_raw_proba_in_unit_interval(self, sample_X):
        obj  = _load_pkl("lightgbm_raw")
        p    = obj.predict_proba(sample_X)
        assert np.all(p >= 0.0), "Raw probabilities below 0"
        assert np.all(p <= 1.0), "Raw probabilities above 1"

    def test_sigmoid_proba_in_unit_interval(self, sample_X):
        obj = _load_pkl("lightgbm_calibrated_sigmoid")
        p   = obj.predict_proba(sample_X)
        assert np.all(p >= 0.0) and np.all(p <= 1.0)

    def test_isotonic_proba_in_unit_interval(self, sample_X):
        obj = _load_pkl("lightgbm_calibrated_isotonic")
        p   = obj.predict_proba(sample_X)
        assert np.all(p >= 0.0) and np.all(p <= 1.0)

    def test_proba_rows_sum_to_one(self, sample_X):
        """Both class probabilities must sum to 1.0 for each model."""
        for name in ("lightgbm_raw",
                     "lightgbm_calibrated_sigmoid",
                     "lightgbm_calibrated_isotonic"):
            obj = _load_pkl(name)
            p   = obj.predict_proba(sample_X)
            assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5), \
                f"{name}: row probabilities do not sum to 1.0"

    def test_proba_no_nan(self, sample_X):
        for name in ("lightgbm_raw",
                     "lightgbm_calibrated_sigmoid",
                     "lightgbm_calibrated_isotonic"):
            obj = _load_pkl(name)
            p   = obj.predict_proba(sample_X)
            assert not np.isnan(p).any(), f"{name}: NaN in predict_proba output"


# ══════════════════════════════════════════════════════════════════════════
# 4. Calibration results JSON content
# ══════════════════════════════════════════════════════════════════════════

@_skip_without_artifacts
class TestCalibrationResults:
    @pytest.fixture(scope="class")
    def results(self):
        return _load_results()

    def test_results_has_required_keys(self, results):
        for key in ("raw", "sigmoid", "isotonic", "best_method",
                    "n_val", "n_test", "split_strategy"):
            assert key in results, f"Missing key: {key}"

    def test_split_strategy_is_time_based(self, results):
        assert results["split_strategy"] == "time_based_issue_d"

    def test_val_period_is_2016(self, results):
        assert results["val_period"] == "2016"

    def test_test_period_is_2017_2018(self, results):
        assert results["test_period"] == "2017-2018"

    def test_n_val_is_positive(self, results):
        assert results["n_val"] > 0

    def test_n_test_is_positive(self, results):
        assert results["n_test"] > 0

    def test_best_method_is_sigmoid_or_isotonic(self, results):
        assert results["best_method"] in ("sigmoid", "isotonic")

    def test_roc_auc_above_random(self, results):
        for method in ("raw", "sigmoid", "isotonic"):
            assert results[method]["roc_auc"] > 0.5, \
                f"{method} ROC-AUC is not above random baseline"

    def test_brier_below_no_skill(self, results):
        """No-skill Brier score = base_rate*(1-base_rate) ≈ 0.19 at 26% default rate."""
        base_rate    = results["test_default_rate"]
        no_skill_brier = base_rate * (1 - base_rate)
        for method in ("raw", "sigmoid", "isotonic"):
            assert results[method]["brier"] < no_skill_brier, \
                f"{method} Brier score is worse than no-skill baseline"

    def test_calibration_reduces_brier(self, results):
        """At least one calibrated model must improve Brier over raw."""
        raw_brier = results["raw"]["brier"]
        best_cal  = min(results["sigmoid"]["brier"], results["isotonic"]["brier"])
        assert best_cal <= raw_brier + 1e-6, \
            "Neither calibration method improves Brier score"

    def test_calibration_reduces_log_loss(self, results):
        """At least one calibrated model must improve Log Loss over raw."""
        raw_ll   = results["raw"]["log_loss"]
        best_cal = min(results["sigmoid"]["log_loss"], results["isotonic"]["log_loss"])
        assert best_cal <= raw_ll + 1e-6, \
            "Neither calibration method improves Log Loss"

    def test_roc_auc_not_degraded_by_calibration(self, results):
        """Calibration must not reduce ROC-AUC by more than 0.005."""
        raw_auc = results["raw"]["roc_auc"]
        for method in ("sigmoid", "isotonic"):
            cal_auc = results[method]["roc_auc"]
            assert cal_auc >= raw_auc - 0.005, \
                f"{method} calibration degraded ROC-AUC by more than 0.005"


# ══════════════════════════════════════════════════════════════════════════
# 5. Leakage invariant — NO pytestmark skip; runs unconditionally.
#    These tests guard against loan_age_months being reintroduced at the
#    code, metadata, or artifact level.
# ══════════════════════════════════════════════════════════════════════════

class TestLeakageInvariant:
    """
    Runs regardless of whether calibration artifacts exist.
    No skip marker — leakage invariants must always be verified.

    These tests enforce the permanent invariant that loan_age_months (and all
    other LEAKAGE_COLS) can never reach model features.  No whitelist,
    no KNOWN_ARTIFACT_ISSUES exceptions — the rule is absolute.
    """

    def test_tuning_metadata_feature_columns_contain_no_leakage(self):
        """tuning_metadata.json must not list any LEAKAGE_COLS as model features."""
        from src.leakage import leakage_columns_present
        if not META_PATH.exists():
            pytest.skip("tuning_metadata.json not present — run src/tune_lightgbm.py")
        meta     = json.loads(META_PATH.read_text())
        features = meta.get("feature_columns", [])
        leaks    = leakage_columns_present(features)
        assert leaks == [], (
            f"LEAKAGE COLUMNS FOUND in tuning_metadata.json feature_columns: {leaks}.\n"
            "Remove them from the metadata and retrain src/tune_lightgbm.py."
        )

    def test_loan_age_months_not_in_tuning_metadata_features(self):
        """Explicit guard: loan_age_months must be absent from tuning metadata."""
        if not META_PATH.exists():
            pytest.skip("tuning_metadata.json not present — run src/tune_lightgbm.py")
        meta     = json.loads(META_PATH.read_text())
        features = meta.get("feature_columns", [])
        assert "loan_age_months" not in features, (
            "loan_age_months is still listed in tuning_metadata.json feature_columns. "
            "Remove it from the metadata and retrain src/tune_lightgbm.py."
        )

    def test_calibrate_py_does_not_reconstruct_loan_age_months(self):
        """calibrate.py must not contain code that assigns loan_age_months as a column."""
        calibrate_src = (PROJECT_ROOT / "src" / "calibrate.py").read_text()
        # Check for the column-assignment patterns that would reconstruct leakage
        bad_patterns = [
            'df["loan_age_months"]',
            "df['loan_age_months']",
        ]
        for pattern in bad_patterns:
            assert pattern not in calibrate_src, (
                f"src/calibrate.py contains '{pattern}'. "
                "The backward-compat reconstruction block must be removed."
            )

    def test_data_cleaning_does_not_create_loan_age_months(self):
        """data_cleaning.py must not contain code that creates loan_age_months."""
        cleaning_src = (PROJECT_ROOT / "src" / "data_cleaning.py").read_text()
        # Ensure it's not assigned as a new column
        assert 'df["loan_age_months"]' not in cleaning_src and \
               "df['loan_age_months']" not in cleaning_src, (
            "src/data_cleaning.py assigns loan_age_months as a column. "
            "Remove that code — it belongs in LEAKAGE_COLS, not in the output."
        )

    def test_feature_engineering_drops_leakage_columns(self):
        """_drop_redundant in feature_engineering.py must reference LEAKAGE_COLS."""
        fe_src = (PROJECT_ROOT / "src" / "feature_engineering.py").read_text()
        assert "LEAKAGE_COLS" in fe_src, (
            "src/feature_engineering.py does not reference LEAKAGE_COLS. "
            "The _drop_redundant() function must explicitly drop leakage columns."
        )

    def test_calibration_results_leakage_in_model_is_empty(self):
        """artifacts/calibration_results.json must report no leakage in model."""
        if not RESULTS_PATH.exists():
            pytest.skip("calibration_results.json not present — run src/calibrate.py")
        results = json.loads(RESULTS_PATH.read_text())
        assert "leakage_in_model" in results, \
            "calibration_results.json missing 'leakage_in_model' field"
        assert results["leakage_in_model"] == [], (
            f"calibration_results.json reports leakage in model: {results['leakage_in_model']}.\n"
            "Retrain src/tune_lightgbm.py then re-run src/calibrate.py."
        )


class TestCalibrationInputValidation:
    @staticmethod
    def _write_frame(path):
        import pandas as pd

        pd.DataFrame(
            {
                "issue_d": pd.to_datetime(["2015-06-01", "2016-06-01", "2017-06-01"]),
                "default": [0, 1, 0],
                "loan_amnt": [5_000.0, 10_000.0, 15_000.0],
                "loan_age_months": [12.0, 24.0, 36.0],
            }
        ).to_parquet(path)

    def test_load_splits_rejects_missing_model_features(self, monkeypatch, tmp_path):
        from src import calibrate

        data_path = tmp_path / "features.parquet"
        meta_path = tmp_path / "metadata.json"
        self._write_frame(data_path)
        meta_path.write_text(json.dumps({"feature_columns": ["loan_amnt", "missing_feature"]}))

        monkeypatch.setattr(calibrate, "INTERIM_PATH", data_path)
        monkeypatch.setattr(calibrate, "CLEANED_PATH", tmp_path / "missing.parquet")
        monkeypatch.setattr(calibrate, "META_PATH", meta_path)

        with pytest.raises(ValueError, match="missing 1 feature"):
            calibrate._load_splits()

    def test_load_splits_rejects_leakage_features(self, monkeypatch, tmp_path):
        from src import calibrate

        data_path = tmp_path / "features.parquet"
        meta_path = tmp_path / "metadata.json"
        self._write_frame(data_path)
        meta_path.write_text(json.dumps({"feature_columns": ["loan_amnt", "loan_age_months"]}))

        monkeypatch.setattr(calibrate, "INTERIM_PATH", data_path)
        monkeypatch.setattr(calibrate, "CLEANED_PATH", tmp_path / "missing.parquet")
        monkeypatch.setattr(calibrate, "META_PATH", meta_path)

        with pytest.raises(ValueError, match="contains leakage features"):
            calibrate._load_splits()
