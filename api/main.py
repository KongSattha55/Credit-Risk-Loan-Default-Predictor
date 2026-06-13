"""
FastAPI inference service for the LendingClub Loan Default Predictor.

Loads the calibrated sigmoid LightGBM model for production probability output.
Applies feature engineering transforms at inference time using
encoding maps saved by src/feature_engineering.py.

Run:
    uvicorn api.main:app --reload
    # Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import json
import pickle
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))  # allow `from src.xxx import`
ARTIFACTS_DIR = PROJECT_ROOT / "mlruns" / "artifacts"
MODEL_PATH    = PROJECT_ROOT / "models" / "lightgbm_calibrated_sigmoid.pkl"
META_PATH     = ARTIFACTS_DIR / "lightgbm_metadata.json"
MAPS_PATH     = ARTIFACTS_DIR / "encoding_maps.json"

from src.calibration_classes import LGBMBoosterWrapper, PreFitCalibratedClassifier  # noqa: F401
from src.inference_fe import apply_feature_engineering, risk_label
from src.leakage import leakage_columns_present


# ══════════════════════════════════════════════════════════════════════════
# Model / metadata loading (cached — loaded once at startup)
# ══════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Calibrated model not found at {MODEL_PATH}. Run `python src/calibrate.py` first.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def load_metadata() -> dict:
    if not META_PATH.exists():
        raise RuntimeError(f"Metadata not found at {META_PATH}.")
    meta = json.loads(META_PATH.read_text())
    leaks = leakage_columns_present(meta.get("feature_columns", []))
    if leaks:
        raise RuntimeError(f"Metadata contains leakage features and cannot be served: {leaks}")
    return meta


@lru_cache(maxsize=1)
def load_encoding_maps() -> dict:
    if not MAPS_PATH.exists():
        raise RuntimeError(
            f"Encoding maps not found at {MAPS_PATH}. Run src/feature_engineering.py first."
        )
    return json.loads(MAPS_PATH.read_text())


# ══════════════════════════════════════════════════════════════════════════
# Request / Response schemas
# ══════════════════════════════════════════════════════════════════════════

class LoanApplication(BaseModel):
    """Raw loan application fields available at origination time."""

    # Core loan terms
    loan_amnt:   float = Field(..., gt=0,        description="Requested loan amount ($)")
    term:        Literal[36, 60] = Field(...,      description="Loan term in months (36 or 60)")
    int_rate:    float = Field(..., gt=0,         description="Interest rate (%)")
    installment: float = Field(..., gt=0,         description="Monthly payment ($)")

    # Borrower grade / sub-grade (strings — encoded internally)
    grade:     Literal["A", "B", "C", "D", "E", "F", "G"] = Field(..., description="LendingClub grade (A–G)")
    sub_grade: str = Field(..., pattern=r"^[A-G][1-5]$", description="LendingClub sub-grade (A1–G5)")

    # Borrower profile
    emp_length:          float = Field(0.0,  ge=0, le=10, description="Employment length (years)")
    home_ownership:      str   = Field(...,               description="RENT / OWN / MORTGAGE / OTHER")
    annual_inc:          float = Field(...,  gt=0,        description="Annual income ($)")
    verification_status: str   = Field(...,               description="Income verification status")
    purpose:             str   = Field(...,               description="Loan purpose (e.g. debt_consolidation)")
    addr_state:          str   = Field(...,               description="Borrower state (2-letter code)")
    emp_title:           str   = Field("",                description="Job title (optional)")

    # Credit metrics
    dti:              float = Field(...,  ge=0,          description="Debt-to-income ratio (%)")
    fico_range_low:   float = Field(...,                 description="FICO score (low end of range)")
    delinq_2yrs:      float = Field(0.0, ge=0)
    inq_last_6mths:   float = Field(0.0, ge=0)
    open_acc:         float = Field(0.0, ge=0)
    pub_rec:          float = Field(0.0, ge=0)
    revol_bal:        float = Field(0.0, ge=0)
    revol_util:       float = Field(0.0, ge=0, le=100)
    total_acc:        float = Field(0.0, ge=0)

    # Loan metadata
    initial_list_status: str   = Field("w",          description="w or f")
    application_type:    str   = Field("Individual",  description="Individual or Joint App")
    disbursement_method: str   = Field("Cash",        description="Cash or DirectPay")
    cr_history_months:   float = Field(0.0, ge=0)

    # Optional bureau fields — default to 0
    collections_12_mths_ex_med: float = Field(0.0, ge=0)
    acc_now_delinq:              float = Field(0.0, ge=0)
    tot_coll_amt:                float = Field(0.0, ge=0)
    tot_cur_bal:                 float = Field(0.0, ge=0)
    open_acc_6m:                 float = Field(0.0, ge=0)
    open_act_il:                 float = Field(0.0, ge=0)
    open_il_12m:                 float = Field(0.0, ge=0)
    open_il_24m:                 float = Field(0.0, ge=0)
    mths_since_rcnt_il:          float = Field(0.0, ge=0)
    total_bal_il:                float = Field(0.0, ge=0)
    il_util:                     float = Field(0.0, ge=0)
    open_rv_12m:                 float = Field(0.0, ge=0)
    open_rv_24m:                 float = Field(0.0, ge=0)
    max_bal_bc:                  float = Field(0.0, ge=0)
    all_util:                    float = Field(0.0, ge=0)
    total_rev_hi_lim:            float = Field(0.0, ge=0)
    inq_fi:                      float = Field(0.0, ge=0)
    total_cu_tl:                 float = Field(0.0, ge=0)
    inq_last_12m:                float = Field(0.0, ge=0)
    acc_open_past_24mths:        float = Field(0.0, ge=0)
    avg_cur_bal:                 float = Field(0.0, ge=0)
    bc_open_to_buy:              float = Field(0.0, ge=0)
    bc_util:                     float = Field(0.0, ge=0)
    chargeoff_within_12_mths:    float = Field(0.0, ge=0)
    delinq_amnt:                 float = Field(0.0, ge=0)
    mo_sin_old_il_acct:          float = Field(0.0, ge=0)
    mo_sin_old_rev_tl_op:        float = Field(0.0, ge=0)
    mo_sin_rcnt_rev_tl_op:       float = Field(0.0, ge=0)
    mo_sin_rcnt_tl:              float = Field(0.0, ge=0)
    mort_acc:                    float = Field(0.0, ge=0)
    mths_since_recent_bc:        float = Field(0.0, ge=0)
    mths_since_recent_inq:       float = Field(0.0, ge=0)
    num_accts_ever_120_pd:       float = Field(0.0, ge=0)
    num_actv_bc_tl:              float = Field(0.0, ge=0)
    num_actv_rev_tl:             float = Field(0.0, ge=0)
    num_bc_sats:                 float = Field(0.0, ge=0)
    num_bc_tl:                   float = Field(0.0, ge=0)
    num_il_tl:                   float = Field(0.0, ge=0)
    num_op_rev_tl:               float = Field(0.0, ge=0)
    num_rev_accts:               float = Field(0.0, ge=0)
    num_rev_tl_bal_gt_0:         float = Field(0.0, ge=0)
    num_sats:                    float = Field(0.0, ge=0)
    num_tl_120dpd_2m:            float = Field(0.0, ge=0)
    num_tl_30dpd:                float = Field(0.0, ge=0)
    num_tl_90g_dpd_24m:          float = Field(0.0, ge=0)
    num_tl_op_past_12m:          float = Field(0.0, ge=0)
    pct_tl_nvr_dlq:              float = Field(100.0, ge=0, le=100)
    percent_bc_gt_75:            float = Field(0.0, ge=0, le=100)
    pub_rec_bankruptcies:        float = Field(0.0, ge=0)
    tax_liens:                   float = Field(0.0, ge=0)
    tot_hi_cred_lim:             float = Field(0.0, ge=0)
    total_bal_ex_mort:           float = Field(0.0, ge=0)
    total_bc_limit:              float = Field(0.0, ge=0)
    total_il_high_credit_limit:  float = Field(0.0, ge=0)


class PredictionResponse(BaseModel):
    default_probability: float = Field(..., description="Predicted probability of default (0–1)")
    prediction:          int   = Field(..., description="1 = default, 0 = fully paid")
    threshold:           float = Field(..., description="Decision threshold used")
    risk_label:          str   = Field(..., description="Low / Medium / High / Very High")
    model_version:       str   = Field(..., description="MLflow run ID of the serving model")


def _build_model_input(application: LoanApplication, feature_columns: list[str]) -> pd.DataFrame:
    """Convert a raw LoanApplication into a model-ready single-row DataFrame."""
    maps    = load_encoding_maps()
    raw_row = application.model_dump()
    eng_row = apply_feature_engineering(raw_row, maps)

    # Build DataFrame with exactly the columns the model expects
    df = pd.DataFrame([{col: eng_row.get(col, 0.0) for col in feature_columns}])
    return df.astype("float32")


# ══════════════════════════════════════════════════════════════════════════
# App
# ══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Loan Default Predictor",
    description="Predicts the probability that a LendingClub loan will default.",
    version="2.0.0",
)


@app.get("/health")
def health():
    """Health check — verifies model and encoding maps are loaded."""
    try:
        meta = load_metadata()
        load_model()           # raises if missing or unreadable
        load_encoding_maps()   # raises if missing
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {
        "status":       "ok",
        "model_type":   meta.get("model_type"),
        "probability_model": "calibrated_sigmoid",
        "test_roc_auc": meta.get("test_roc_auc"),
        "n_features":   len(meta.get("feature_columns", [])),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    """
    Predict default probability for a loan application.

    Supply the raw loan application fields (grade as 'A'–'G', etc.).
    Feature engineering (encoding, log transforms, ratios) is applied
    internally before the model scores the application.
    """
    try:
        model    = load_model()
        meta     = load_metadata()
        load_encoding_maps()
    except RuntimeError as exc:
        # Missing model / metadata / encoding maps — operator-visible, safe to surface.
        raise HTTPException(status_code=503, detail=str(exc))

    features = meta["feature_columns"]
    thresh   = meta["threshold"]

    try:
        df   = _build_model_input(application, features)
        prob = float(model.predict_proba(df)[0, 1])
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid input: {exc}")

    pred = int(prob >= thresh)
    return PredictionResponse(
        default_probability=round(prob, 6),
        prediction=pred,
        threshold=thresh,
        risk_label=risk_label(prob),
        model_version=meta["mlflow_run_id"],
    )


@app.get("/model-info")
def model_info():
    """Return model metadata and performance metrics."""
    try:
        meta = load_metadata()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {
        "model_type":    meta.get("model_type"),
        "probability_model": "calibrated_sigmoid",
        "model_path":    str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "mlflow_run_id": meta.get("mlflow_run_id"),
        "threshold":     meta.get("threshold"),
        "n_features":    len(meta.get("feature_columns", [])),
        "metrics": {
            "val_roc_auc":        meta.get("val_roc_auc"),
            "val_avg_precision":  meta.get("val_avg_precision"),
            "test_roc_auc":       meta.get("test_roc_auc"),
            "test_avg_precision": meta.get("test_avg_precision"),
            "test_f1":            meta.get("test_f1"),
            "test_brier":         meta.get("test_brier"),
        },
    }
