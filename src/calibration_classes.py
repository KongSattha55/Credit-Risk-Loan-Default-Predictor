"""
src/calibration_classes.py
──────────────────────────
Picklable calibration classes — imported by src/calibrate.py and tests.

Keeping these in a dedicated module (instead of in calibrate.py's __main__)
ensures pickle can resolve the class names when the .pkl files are loaded
from test runners, the API, or the dashboard.
"""

from __future__ import annotations

import numpy as np

# LightGBM is an optional dependency at import time — the classes
# annotate lgb.Booster in __init__ only, so we can tolerate the import
# being deferred.
try:
    import lightgbm as lgb
except ImportError:
    lgb = None  # type: ignore[assignment]

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42


# ══════════════════════════════════════════════════════════════════════════
# LGBMBoosterWrapper
# ══════════════════════════════════════════════════════════════════════════

class LGBMBoosterWrapper:
    """
    Thin sklearn-compatible wrapper around a fitted lgb.Booster.

    Exposes the predict_proba / predict / classes_ interface expected by
    sklearn calibration utilities and the PreFitCalibratedClassifier below.
    """

    def __init__(self, booster, best_iteration: int = 0) -> None:
        self.booster_       = booster
        self.best_iteration = best_iteration
        self.classes_       = np.array([0, 1])

    def predict_proba(self, X) -> np.ndarray:
        prob = self.booster_.predict(X, num_iteration=self.best_iteration or 0)
        return np.column_stack([1.0 - prob, prob])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_params(self, deep: bool = True) -> dict:
        return {"best_iteration": self.best_iteration}


# ══════════════════════════════════════════════════════════════════════════
# PreFitCalibratedClassifier
# ══════════════════════════════════════════════════════════════════════════

class PreFitCalibratedClassifier:
    """
    Equivalent to sklearn's CalibratedClassifierCV(estimator, cv="prefit"),
    which was removed in sklearn 1.8.

    The base estimator is already trained.  fit(X_val, y_val) trains only
    the thin calibration layer on the provided validation data; the base
    model weights are never modified.

    Calibration methods
    -------------------
    "isotonic"  IsotonicRegression — non-parametric, preferred with large N
    "sigmoid"   LogisticRegression on raw scores (Platt scaling)

    Interface
    ---------
    .fit(X, y)          fits the calibration layer
    .predict_proba(X)   returns (n_samples, 2) array summing to 1.0
    .predict(X)         binary labels at threshold 0.5
    .classes_           np.array([0, 1])
    """

    def __init__(
        self,
        estimator: LGBMBoosterWrapper,
        method: str = "isotonic",
    ) -> None:
        if method not in ("isotonic", "sigmoid"):
            raise ValueError(f"method must be 'isotonic' or 'sigmoid', got {method!r}")
        self.estimator   = estimator
        self.method      = method
        self.classes_    = estimator.classes_
        self._calibrator = None

    def fit(self, X, y) -> "PreFitCalibratedClassifier":
        """Fit the calibration layer on pre-scored (X, y)."""
        raw = self.estimator.predict_proba(X)[:, 1]
        y   = np.asarray(y)
        if self.method == "isotonic":
            self._calibrator = IsotonicRegression(out_of_bounds="clip")
            self._calibrator.fit(raw, y)
        else:  # sigmoid / Platt scaling
            self._calibrator = LogisticRegression(random_state=RANDOM_STATE)
            self._calibrator.fit(raw.reshape(-1, 1), y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self._calibrator is None:
            raise RuntimeError("Call fit() before predict_proba().")
        raw = self.estimator.predict_proba(X)[:, 1]
        if self.method == "isotonic":
            cal = self._calibrator.transform(raw)
        else:
            cal = self._calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_params(self, deep: bool = True) -> dict:
        return {"method": self.method}
