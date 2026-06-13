"""
Streamlit dashboard for the Loan Default Predictor.

Run:
    streamlit run app/main.py

Tabs:
  1. Predict      — single loan application scoring
  2. Threshold Analysis — precision/recall trade-off explorer
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))   # allow `from src.xxx import`

ARTIFACTS_DIR = PROJECT_ROOT / "mlruns" / "artifacts"
CALIBRATED_MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_calibrated_sigmoid.pkl"
RAW_MODEL_PATH = ARTIFACTS_DIR / "lightgbm_model.pkl"
META_PATH     = ARTIFACTS_DIR / "lightgbm_metadata.json"

INTERIM_PATH  = PROJECT_ROOT / "data" / "interim"   / "loans_features.parquet"
CLEANED_PATH  = PROJECT_ROOT / "data" / "processed"  / "loans_cleaned.parquet"
MAPS_PATH     = ARTIFACTS_DIR / "encoding_maps.json"

from src.explain_utils import summarize_shap_local
from src.calibration_classes import LGBMBoosterWrapper, PreFitCalibratedClassifier  # noqa: F401
from src.business_risk import assign_risk_band, calculate_expected_loss, threshold_business_table
from src.inference_fe import ALWAYS_DROP as _ALWAYS_DROP_T, TARGET as _TARGET
from src.leakage import feature_columns as leakage_free_feature_columns, leakage_columns_present
from src.splits import time_split_masks

RANDOM_STATE = 42
TARGET       = _TARGET
ALWAYS_DROP  = list(_ALWAYS_DROP_T)


# ══════════════════════════════════════════════════════════════════════════
# Cached resources
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_calibrated_model():
    with open(CALIBRATED_MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_raw_lgb_model():
    with open(RAW_MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_shap_explainer():
    """TreeExplainer is the right choice for LightGBM boosters — O(TD²L)
    per row, no background sample needed."""
    import shap
    return shap.TreeExplainer(load_raw_lgb_model())


@st.cache_data
def load_metadata() -> dict:
    meta = json.loads(META_PATH.read_text())
    leaks = leakage_columns_present(meta.get("feature_columns", []))
    if leaks:
        raise RuntimeError(f"Metadata contains leakage features and cannot be served: {leaks}")
    return meta


@st.cache_data
def load_encoding_maps() -> dict:
    return json.loads(MAPS_PATH.read_text())


@st.cache_data(show_spinner="Loading data and computing predictions …")
def load_test_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct the 2017+ time-based test split used by training and return
    (y_true, y_proba). Results are cached so the tab loads fast.
    """
    if INTERIM_PATH.exists():
        df = pd.read_parquet(INTERIM_PATH)
    elif CLEANED_PATH.exists():
        df = pd.read_parquet(CLEANED_PATH)
    else:
        return None, None, None

    if "issue_d" not in df.columns:
        return None, None, None

    meta     = load_metadata()
    features = meta["feature_columns"]
    available_features = leakage_free_feature_columns(df.columns, extra_drop=["issue_d"])
    missing = [c for c in features if c not in available_features]
    if missing:
        return None, None, None

    _, _, test_mask = time_split_masks(df["issue_d"])
    X_test = df.loc[test_mask, features].copy()
    y_test = df.loc[test_mask, TARGET].astype("int32")
    loan_amounts = df.loc[test_mask, "loan_amnt"].astype("float32")

    model   = load_calibrated_model()
    y_proba = model.predict_proba(X_test)[:, 1]

    return y_test.values, y_proba, loan_amounts.values


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

from src.inference_fe import apply_feature_engineering, risk_label


def predict_single(inputs: dict, feature_columns: list[str]) -> float:
    model = load_calibrated_model()
    maps  = load_encoding_maps()
    eng   = apply_feature_engineering(inputs, maps)
    row   = {col: [eng.get(col, 0.0)] for col in feature_columns}
    df    = pd.DataFrame(row).astype("float32")
    return float(model.predict_proba(df)[0, 1])


def positive_class_shap_values(raw_values) -> np.ndarray:
    if isinstance(raw_values, list):
        raw_values = raw_values[1] if len(raw_values) > 1 else raw_values[0]
    values = np.asarray(raw_values)
    if values.ndim == 3:
        if values.shape[-1] == 2:
            values = values[:, :, 1]
        elif values.shape[0] == 2:
            values = values[1]
    return values


def metrics_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, thresh: float) -> dict:
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
    y_pred = (y_proba >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "specificity": spec,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


# ══════════════════════════════════════════════════════════════════════════
# Page setup
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Credit Risk — Loan Default Predictor")

if not CALIBRATED_MODEL_PATH.exists():
    st.error("Calibrated model not found. Run `python src/calibrate.py` first.")
    st.stop()

if not RAW_MODEL_PATH.exists():
    st.error("Raw LightGBM model not found. Run `python src/train.py --model lightgbm` first.")
    st.stop()

meta      = load_metadata()
features  = meta["feature_columns"]
saved_thr = meta["threshold"]

with st.sidebar:
    st.header("Model Info")
    st.metric("Test ROC-AUC",       f"{meta['test_roc_auc']:.4f}")
    st.metric("Test Avg Precision", f"{meta['test_avg_precision']:.4f}")
    st.metric("Test F1",            f"{meta['test_f1']:.4f}")
    st.metric("Saved Threshold",    f"{saved_thr:.4f}")
    st.caption("PD model: calibrated sigmoid")
    st.caption(f"Data: {meta.get('data_source', '—')}")
    st.caption(f"Run ID: {meta['mlflow_run_id'][:8]}…")

tab_predict, tab_threshold, tab_business, tab_batch = st.tabs([
    "Predict",
    "Threshold Analysis",
    "Business Risk Analytics",
    "Batch Score (CSV)",
])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — Predict
# ══════════════════════════════════════════════════════════════════════════

with tab_predict:
    st.subheader("Loan Application Details")

    # ── Example presets ────────────────────────────────────────────────────
    # Three canonical loans to showcase the model without typing 30+ fields.
    PRESETS = {
        "Safe (A-grade, low DTI)": {
            "loan_amnt": 8000, "term": 36, "int_rate": 6.5, "installment": 245.0,
            "grade": "A", "sub_grade": "A2", "emp_length": 10,
            "home_ownership": "MORTGAGE", "annual_inc": 95000,
            "verification_status": "Verified", "purpose": "credit_card",
            "addr_state": "CA", "dti": 8.0, "fico_range_low": 780,
            "delinq_2yrs": 0, "inq_last_6mths": 0, "open_acc": 6, "pub_rec": 0,
            "revol_bal": 1500, "revol_util": 8.0, "total_acc": 22,
            "cr_history_months": 240, "mort_acc": 1,
        },
        "Borderline (C-grade, medium DTI)": {
            "loan_amnt": 15000, "term": 36, "int_rate": 13.0, "installment": 505.0,
            "grade": "C", "sub_grade": "C2", "emp_length": 5,
            "home_ownership": "RENT", "annual_inc": 52000,
            "verification_status": "Source Verified", "purpose": "debt_consolidation",
            "addr_state": "TX", "dti": 22.0, "fico_range_low": 680,
            "delinq_2yrs": 0, "inq_last_6mths": 1, "open_acc": 9, "pub_rec": 0,
            "revol_bal": 8000, "revol_util": 45.0, "total_acc": 18,
            "cr_history_months": 120, "mort_acc": 0,
        },
        "Risky (F-grade, high DTI)": {
            "loan_amnt": 30000, "term": 60, "int_rate": 26.0, "installment": 895.0,
            "grade": "F", "sub_grade": "F3", "emp_length": 1,
            "home_ownership": "RENT", "annual_inc": 35000,
            "verification_status": "Not Verified", "purpose": "small_business",
            "addr_state": "NV", "dti": 38.0, "fico_range_low": 620,
            "delinq_2yrs": 1, "inq_last_6mths": 4, "open_acc": 12, "pub_rec": 1,
            "revol_bal": 14000, "revol_util": 88.0, "total_acc": 15,
            "cr_history_months": 80, "mort_acc": 0,
        },
    }

    st.caption("Load a preset, or fill the form manually:")
    pc1, pc2, pc3, _ = st.columns([2, 2, 2, 3])
    for (label, preset_data), col in zip(PRESETS.items(), (pc1, pc2, pc3)):
        with col:
            if st.button(label, use_container_width=True, key=f"preset_{label}"):
                for k, v in preset_data.items():
                    st.session_state[f"fld_{k}"] = v
                st.session_state["_preset_applied"] = label
    if st.session_state.get("_preset_applied"):
        st.info(f"Loaded preset: **{st.session_state['_preset_applied']}**")

    st.divider()
    col1, col2, col3 = st.columns(3)

    def _ss(key: str, default):
        return st.session_state.get(f"fld_{key}", default)

    with col1:
        st.markdown("**Loan Terms**")
        loan_amnt   = st.number_input("Loan Amount ($)", 500, 40000, _ss("loan_amnt", 10000), 500, key="fld_loan_amnt")
        term        = st.selectbox("Term (months)", [36, 60], index=[36, 60].index(_ss("term", 36)), key="fld_term")
        int_rate    = st.slider("Interest Rate (%)", 5.0, 31.0, float(_ss("int_rate", 12.5)), 0.25, key="fld_int_rate")
        installment = st.number_input("Monthly Installment ($)", 10.0, 2000.0, float(_ss("installment", 335.0)), key="fld_installment")
        _purposes = ["debt_consolidation", "credit_card", "home_improvement", "other",
                     "major_purchase", "small_business", "car", "medical", "moving",
                     "vacation", "house", "wedding", "educational", "renewable_energy"]
        purpose     = st.selectbox("Purpose", _purposes,
                                   index=_purposes.index(_ss("purpose", "debt_consolidation")),
                                   key="fld_purpose")

    with col2:
        st.markdown("**Borrower Profile**")
        _grades = list("ABCDEFG")
        grade       = st.selectbox("Grade", _grades, index=_grades.index(_ss("grade", "C")), key="fld_grade")
        _subs = [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]
        sub_grade   = st.selectbox("Sub-grade", _subs, index=_subs.index(_ss("sub_grade", "C3")), key="fld_sub_grade")
        emp_length  = st.slider("Employment Length (years)", 0, 10, int(_ss("emp_length", 5)), key="fld_emp_length")
        _ho = ["RENT", "MORTGAGE", "OWN", "OTHER"]
        home_ownership = st.selectbox("Home Ownership", _ho, index=_ho.index(_ss("home_ownership", "RENT")), key="fld_home_ownership")
        annual_inc  = st.number_input("Annual Income ($)", 1000, 10_000_000, int(_ss("annual_inc", 60000)), 1000, key="fld_annual_inc")
        _vs = ["Not Verified", "Verified", "Source Verified"]
        verification_status = st.selectbox("Verification Status", _vs,
                                           index=_vs.index(_ss("verification_status", "Verified")),
                                           key="fld_verification_status")
        _states = [
            "CA","NY","TX","FL","IL","NJ","PA","OH","GA","MI","NC","VA",
            "WA","MA","AZ","CO","MD","MN","IN","TN","MO","WI","CT","OR",
            "SC","LA","KY","AL","OK","UT","NV","AR","MS","KS","NE","NM",
            "WV","ID","HI","NH","ME","RI","MT","DE","SD","ND","AK","VT",
            "WY","DC","IA",
        ]
        addr_state  = st.selectbox("State", _states, index=_states.index(_ss("addr_state", "CA")), key="fld_addr_state")

    with col3:
        st.markdown("**Credit Profile**")
        fico_range_low = st.slider("FICO Score (low)", 580, 850, int(_ss("fico_range_low", 700)), key="fld_fico_range_low")
        dti         = st.slider("Debt-to-Income Ratio (%)", 0.0, 100.0, float(_ss("dti", 15.0)), 0.5, key="fld_dti")
        revol_util  = st.slider("Revolving Utilisation (%)", 0.0, 100.0, float(_ss("revol_util", 30.0)), 0.5, key="fld_revol_util")
        revol_bal   = st.number_input("Revolving Balance ($)", 0, 500_000, int(_ss("revol_bal", 5000)), 100, key="fld_revol_bal")
        open_acc    = st.number_input("Open Credit Lines", 0, 100, int(_ss("open_acc", 8)), key="fld_open_acc")
        total_acc   = st.number_input("Total Credit Lines", 0, 200, int(_ss("total_acc", 20)), key="fld_total_acc")
        inq_last_6mths = st.number_input("Inquiries (last 6 mo.)", 0, 30, int(_ss("inq_last_6mths", 1)), key="fld_inq_last_6mths")
        delinq_2yrs    = st.number_input("Delinquencies (last 2 yr.)", 0, 30, int(_ss("delinq_2yrs", 0)), key="fld_delinq_2yrs")
        pub_rec        = st.number_input("Public Derogatory Records", 0, 20, int(_ss("pub_rec", 0)), key="fld_pub_rec")

    with st.expander("Advanced / Bureau Fields (optional)", expanded=False):
        ac1, ac2 = st.columns(2)
        with ac1:
            cr_history_months = st.number_input("Credit History (months)", 0, 1200, int(_ss("cr_history_months", 120)), key="fld_cr_history_months")
            mort_acc          = st.number_input("Mortgage Accounts", 0, 50, int(_ss("mort_acc", 0)), key="fld_mort_acc")
            tot_cur_bal       = st.number_input("Total Current Balance ($)", 0, 5_000_000, int(_ss("tot_cur_bal", 20000)), 1000, key="fld_tot_cur_bal")
            avg_cur_bal       = st.number_input("Avg Current Balance ($)", 0, 1_000_000, int(_ss("avg_cur_bal", 2000)), 100, key="fld_avg_cur_bal")
        with ac2:
            bc_util           = st.slider("Bankcard Utilisation (%)", 0.0, 100.0, float(_ss("bc_util", 40.0)), key="fld_bc_util")
            pct_tl_nvr_dlq    = st.slider("% Accounts Never Delinquent", 0.0, 100.0, float(_ss("pct_tl_nvr_dlq", 95.0)), key="fld_pct_tl_nvr_dlq")
            acc_open_past_24mths = st.number_input("Accounts Opened (last 24 mo.)", 0, 50, int(_ss("acc_open_past_24mths", 3)), key="fld_acc_open_past_24mths")
            num_actv_rev_tl   = st.number_input("Active Revolving Accounts", 0, 50, int(_ss("num_actv_rev_tl", 4)), key="fld_num_actv_rev_tl")
            num_bc_sats       = st.number_input("Bankcard Accounts — Satisfactory", 0, 30, int(_ss("num_bc_sats", 3)), key="fld_num_bc_sats")

    inputs = {
        "loan_amnt": loan_amnt, "term": term, "int_rate": int_rate,
        "installment": installment, "grade": grade, "sub_grade": sub_grade,
        "emp_length": emp_length, "home_ownership": home_ownership,
        "annual_inc": annual_inc, "verification_status": verification_status,
        "purpose": purpose, "addr_state": addr_state, "dti": dti,
        "delinq_2yrs": delinq_2yrs, "fico_range_low": fico_range_low,
        "inq_last_6mths": inq_last_6mths, "open_acc": open_acc, "pub_rec": pub_rec,
        "revol_bal": revol_bal, "revol_util": revol_util, "total_acc": total_acc,
        "initial_list_status": "w", "application_type": "Individual",
        "disbursement_method": "Cash",
        "cr_history_months": cr_history_months,
        "mort_acc": mort_acc, "tot_cur_bal": tot_cur_bal, "avg_cur_bal": avg_cur_bal,
        "bc_util": bc_util, "pct_tl_nvr_dlq": pct_tl_nvr_dlq,
        "acc_open_past_24mths": acc_open_past_24mths,
        "num_actv_rev_tl": num_actv_rev_tl, "num_bc_sats": num_bc_sats,
    }

    st.divider()
    if st.button("Predict Default Risk", type="primary", use_container_width=True):
        with st.spinner("Running model …"):
            prob = predict_single(inputs, features)

        label = risk_label(prob)
        pred  = int(prob >= saved_thr)

        r1, r2, r3 = st.columns(3)
        with r1: st.metric("Default Probability", f"{prob:.1%}")
        with r2: st.metric("Prediction", "Default" if pred else "Fully Paid")
        with r3: st.metric("Risk Tier", label)

        if pred == 1:
            st.error(f"**HIGH RISK** — model predicts default (prob = {prob:.1%})")
        else:
            st.success(f"**LOW RISK** — model predicts repayment (prob = {prob:.1%})")

        st.markdown("**Default Probability**")
        st.progress(min(prob, 1.0))
        st.caption(
            f"Saved threshold: {saved_thr:.4f}  |  "
            f"Predicted: {'Default (1)' if pred else 'Fully Paid (0)'}"
        )

        # ── Individual model explanation ────────────────────────────────────
        st.divider()
        st.markdown("**Model explanation**")
        try:
            maps  = load_encoding_maps()
            eng   = apply_feature_engineering(inputs, maps)
            row   = pd.DataFrame(
                [{col: eng.get(col, 0.0) for col in features}]
            ).astype("float32")

            explainer  = load_shap_explainer()
            shap_vals  = positive_class_shap_values(explainer.shap_values(row))[0]
            feat_vals  = row.iloc[0].values
            summary    = summarize_shap_local(shap_vals, feat_vals, features, top_n=5)

            c_inc, c_dec = st.columns(2)
            with c_inc:
                st.markdown("**Factors increasing risk**")
                if summary["top_positive_drivers"]:
                    for driver in summary["top_positive_drivers"]:
                        st.markdown(f"- {driver['explanation']}")
                else:
                    st.caption("No strong risk-increasing factors found.")
            with c_dec:
                st.markdown("**Factors reducing risk**")
                if summary["top_negative_drivers"]:
                    for driver in summary["top_negative_drivers"]:
                        st.markdown(f"- {driver['explanation']}")
                else:
                    st.caption("No strong risk-reducing factors found.")

            st.markdown("**Plain-English explanation**")
            st.text(summary["explanation_text"])

            top_n = 10
            order = np.argsort(np.abs(shap_vals))[::-1][:top_n]

            top_feats = [features[i] for i in order][::-1]
            top_shap  = [float(shap_vals[i]) for i in order][::-1]
            top_raw   = [float(feat_vals[i]) for i in order][::-1]
            colors    = ["#E53935" if v > 0 else "#1E88E5" for v in top_shap]
            labels    = [f"{f} = {v:.3g}" for f, v in zip(top_feats, top_raw)]

            shap_fig = go.Figure(go.Bar(
                x=top_shap, y=labels, orientation="h",
                marker=dict(color=colors),
                text=[f"{v:+.3f}" for v in top_shap],
                textposition="outside",
            ))
            shap_fig.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=20, b=10),
                title="Main factors for this prediction",
                xaxis_title="Risk impact",
                yaxis_title="",
                showlegend=False,
            )
            st.plotly_chart(shap_fig, use_container_width=True)
        except Exception as exc:
            st.warning(f"Model explanation unavailable: {exc}")


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — Threshold Analysis
# ══════════════════════════════════════════════════════════════════════════

with tab_threshold:
    st.subheader("Precision / Recall Threshold Explorer")
    st.markdown(
        "Adjust the decision threshold to balance **precision** (fewer false rejections) "
        "against **recall** (fewer missed defaults). The current saved threshold is "
        f"**{saved_thr:.4f}** (max-F1 strategy)."
    )

    # ── Load test-set predictions ──────────────────────────────────────────
    y_true, y_proba, _loan_amounts = load_test_predictions()

    if y_true is None:
        st.warning("No data found to compute curves. Run src/data_cleaning.py first.")
        st.stop()

    # ── Strategy buttons ───────────────────────────────────────────────────
    from src.threshold_refinement import ThresholdRefiner

    @st.cache_data
    def get_refiner_sweep():
        """Pre-compute the 500-point sweep once and cache."""
        ref = ThresholdRefiner(y_true, y_proba, n_thresholds=500)
        return ref.sweep, ref.roc_auc, ref.pr_auc, ref.ks_stat, ref.pr_curve, ref.roc_curve

    sweep_df, roc_auc, pr_auc, ks_stat, pr_curve_data, roc_curve_data = get_refiner_sweep()
    pr_prec, pr_rec, _ = pr_curve_data
    roc_fpr, roc_tpr, _ = roc_curve_data

    # ── Strategy selector + constraint inputs ──────────────────────────────
    st.markdown("#### Choose a Threshold Strategy")
    sc1, sc2, sc3 = st.columns([2, 2, 3])
    with sc1:
        strategy = st.selectbox(
            "Strategy",
            options=["max_f1", "max_precision", "max_recall", "youden", "cost_sensitive", "max_ks"],
            format_func=lambda x: {
                "max_f1":          "Max F1 (default)",
                "max_precision":   "Max Precision  (↓ false rejections)",
                "max_recall":      "Max Recall  (↓ missed defaults)",
                "youden":          "Youden's J  (ROC-optimal)",
                "cost_sensitive":  "Cost-Sensitive  (FP/FN costs)",
                "max_ks":          "Max KS  (credit-risk separation)",
            }[x],
        )
    with sc2:
        if strategy == "max_precision":
            min_recall = st.slider("Minimum Recall", 0.10, 0.95, 0.50, 0.05)
        elif strategy == "max_recall":
            min_precision = st.slider("Minimum Precision", 0.10, 0.80, 0.35, 0.05)
        elif strategy == "cost_sensitive":
            fp_cost = st.number_input("FP Cost (false rejection)", 0.1, 20.0, 1.0, 0.5)
            fn_cost = st.number_input("FN Cost (missed default)",  0.1, 20.0, 5.0, 0.5)
    with sc3:
        if strategy == "max_f1":
            st.info("Maximises the harmonic mean of precision and recall.")
        elif strategy == "max_precision":
            st.info("Highest precision while catching at least the chosen share of defaults.")
        elif strategy == "max_recall":
            st.info("Highest recall while maintaining at least the chosen precision.")
        elif strategy == "youden":
            st.info("Maximises sensitivity + specificity − 1. Balanced, curve-derived.")
        elif strategy == "max_ks":
            st.info(f"Threshold at max separation (KS = {ks_stat:.4f}) between defaulter and non-defaulter score CDFs.")
        else:
            st.info("Minimises FP×cost_fp + FN×cost_fn. Set costs to reflect business impact.")

    # ── Compute recommended threshold ─────────────────────────────────────
    ref = ThresholdRefiner(y_true, y_proba, n_thresholds=500)

    try:
        if strategy == "max_f1":
            result = ref.max_f1()
        elif strategy == "max_precision":
            result = ref.max_precision(min_recall)
        elif strategy == "max_recall":
            result = ref.max_recall(min_precision)
        elif strategy == "youden":
            result = ref.youden()
        elif strategy == "max_ks":
            result = ref.max_ks()
        else:
            result = ref.cost_sensitive(fp_cost, fn_cost)
        strategy_ok = True
    except ValueError as e:
        st.error(str(e))
        strategy_ok = False

    # ── Manual slider ──────────────────────────────────────────────────────
    st.markdown("#### Manual Threshold Slider")
    default_slider = float(result["threshold"]) if strategy_ok else saved_thr
    manual_thresh  = st.slider(
        "Threshold",
        min_value=0.01, max_value=0.99,
        value=default_slider, step=0.005,
        format="%.3f",
    )

    m = metrics_at_threshold(y_true, y_proba, manual_thresh)
    saved_m = metrics_at_threshold(y_true, y_proba, saved_thr)

    # ── Metric cards ───────────────────────────────────────────────────────
    st.markdown(f"#### Metrics at threshold **{manual_thresh:.3f}**")
    mc1, mc2, mc3, mc4 = st.columns(4)

    def _delta(new, old):
        d = new - old
        return f"{d:+.4f}"

    mc1.metric("Precision",    f"{m['precision']:.4f}",    _delta(m['precision'],    saved_m['precision']))
    mc2.metric("Recall",       f"{m['recall']:.4f}",       _delta(m['recall'],       saved_m['recall']))
    mc3.metric("F1",           f"{m['f1']:.4f}",           _delta(m['f1'],           saved_m['f1']))
    mc4.metric("Specificity",  f"{m['specificity']:.4f}",  _delta(m['specificity'],  saved_m['specificity']))

    st.caption(f"Delta vs saved threshold ({saved_thr:.4f})")

    # ── Confusion matrix ───────────────────────────────────────────────────
    st.markdown("#### Confusion Matrix")
    cm_vals = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
    cm_fig = go.Figure(go.Heatmap(
        z=cm_vals,
        x=["Pred: Fully Paid", "Pred: Default"],
        y=["True: Fully Paid", "True: Default"],
        colorscale="Blues",
        showscale=False,
        text=[[f"{v:,}" for v in row] for row in cm_vals],
        texttemplate="%{text}",
        textfont={"size": 18},
    ))
    cm_fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Predicted", yaxis_title="Actual",
    )
    st.plotly_chart(cm_fig, use_container_width=True)

    # ── Precision / Recall / F1 vs Threshold ──────────────────────────────
    st.markdown("#### Precision, Recall & F1 vs Threshold")
    fig_sweep = go.Figure()
    fig_sweep.add_trace(go.Scatter(
        x=sweep_df["threshold"], y=sweep_df["precision"],
        name="Precision", line=dict(color="#1E88E5", width=2),
    ))
    fig_sweep.add_trace(go.Scatter(
        x=sweep_df["threshold"], y=sweep_df["recall"],
        name="Recall", line=dict(color="#E53935", width=2),
    ))
    fig_sweep.add_trace(go.Scatter(
        x=sweep_df["threshold"], y=sweep_df["f1"],
        name="F1", line=dict(color="#43A047", width=2, dash="dot"),
    ))
    # Saved threshold
    fig_sweep.add_vline(
        x=saved_thr, line_dash="dash", line_color="grey",
        annotation_text=f"Saved ({saved_thr:.3f})", annotation_position="top left",
    )
    # Manual threshold
    fig_sweep.add_vline(
        x=manual_thresh, line_dash="solid", line_color="#FF6F00",
        annotation_text=f"Current ({manual_thresh:.3f})", annotation_position="top right",
    )
    # Strategy recommendation
    if strategy_ok and abs(result["threshold"] - manual_thresh) > 0.002:
        fig_sweep.add_vline(
            x=result["threshold"], line_dash="dot", line_color="#6A1B9A",
            annotation_text=f"{strategy} ({result['threshold']:.3f})",
            annotation_position="bottom right",
        )
    fig_sweep.update_layout(
        height=380,
        xaxis_title="Threshold",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_sweep, use_container_width=True)

    # ── Precision-Recall Curve + ROC Curve ────────────────────────────────
    st.markdown("#### Precision-Recall Curve & ROC Curve")
    fig_curves = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Precision-Recall Curve  (AUC = {pr_auc:.4f})",
            f"ROC Curve  (AUC = {roc_auc:.4f}, KS = {ks_stat:.4f})",
        ),
    )

    # PR curve
    fig_curves.add_trace(go.Scatter(
        x=pr_rec, y=pr_prec, mode="lines",
        name="PR curve", line=dict(color="#5C6BC0", width=2),
    ), row=1, col=1)
    fig_curves.add_trace(go.Scatter(
        x=[m["recall"]], y=[m["precision"]], mode="markers",
        name=f"Threshold {manual_thresh:.3f}",
        marker=dict(color="#FF6F00", size=12, symbol="circle"),
        showlegend=True,
    ), row=1, col=1)
    fig_curves.add_hline(
        y=y_true.mean(), line_dash="dash", line_color="grey",
        annotation_text=f"Baseline ({y_true.mean():.2%})",
        row=1, col=1,
    )

    # ROC curve
    fig_curves.add_trace(go.Scatter(
        x=roc_fpr, y=roc_tpr, mode="lines",
        name="ROC curve", line=dict(color="#5C6BC0", width=2), showlegend=False,
    ), row=1, col=2)
    fig_curves.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="grey", dash="dash"), showlegend=False,
        name="Random",
    ), row=1, col=2)
    # Mark current threshold on ROC
    spec = m["specificity"]
    rec  = m["recall"]
    fig_curves.add_trace(go.Scatter(
        x=[1 - spec], y=[rec], mode="markers",
        name=f"Threshold {manual_thresh:.3f}",
        marker=dict(color="#FF6F00", size=12, symbol="circle"), showlegend=False,
    ), row=1, col=2)

    fig_curves.update_xaxes(title_text="Recall",      row=1, col=1, range=[0, 1])
    fig_curves.update_yaxes(title_text="Precision",   row=1, col=1, range=[0, 1])
    fig_curves.update_xaxes(title_text="FPR (1-Spec)", row=1, col=2, range=[0, 1])
    fig_curves.update_yaxes(title_text="TPR (Recall)", row=1, col=2, range=[0, 1])
    fig_curves.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_curves, use_container_width=True)

    # ── Strategy comparison table ──────────────────────────────────────────
    st.markdown("#### All-Strategy Comparison")
    all_results = ref.compare_all(
        min_recall=0.50,
        min_precision=0.35,
        fp_cost=1.0,
        fn_cost=5.0,
    )
    rows = []
    for r in all_results:
        rows.append({
            "Strategy":    r["strategy"],
            "Threshold":   r["threshold"],
            "Precision":   r["precision"],
            "Recall":      r["recall"],
            "F1":          r["f1"],
            "Specificity": r["specificity"],
            "TP": r["tp"], "FP": r["fp"], "FN": r["fn"], "TN": r["tn"],
        })
    cmp_df = pd.DataFrame(rows).set_index("Strategy")

    # Highlight the currently selected strategy
    def highlight_strategy(row):
        return ["background-color: #FFF3E0" if row.name == strategy else "" for _ in row]

    st.dataframe(
        cmp_df.style
        .apply(highlight_strategy, axis=1)
        .format({
            "Threshold": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
            "F1": "{:.4f}", "Specificity": "{:.4f}",
            "TP": "{:,}", "FP": "{:,}", "FN": "{:,}", "TN": "{:,}",
        }),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown(
        "**Interpretation guide**\n\n"
        "- **Precision** — of all loans flagged as default, what % truly default "
        "(high = fewer good borrowers wrongly rejected)\n"
        "- **Recall** — of all actual defaults, what % the model catches "
        "(high = fewer bad loans slip through approval)\n"
        "- **Specificity** — of all good loans, what % are correctly approved\n"
        "- **Youden's J** — maximises (sensitivity + specificity − 1), curve-derived optimum\n"
        "- **Cost-sensitive** — minimises FP×cost_fp + FN×cost_fn "
        "(default 5:1 reflects that a missed default costs ~5× a wrongful rejection)"
    )


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — Business Risk Analytics
# ══════════════════════════════════════════════════════════════════════════

with tab_business:
    st.subheader("Business Risk Analytics")
    st.markdown(
        "Analyze approval/rejection trade-offs using calibrated probabilities, "
        "loan amount as exposure at default, and the selected loss-given-default assumption."
    )

    y_true_b, y_proba_b, loan_amounts_b = load_test_predictions()

    if y_true_b is None:
        st.warning("No data found to compute business analytics. Run src/data_cleaning.py first.")
        st.stop()

    bc1, bc2 = st.columns([2, 1])
    with bc1:
        business_threshold = st.slider(
            "Approval threshold",
            min_value=0.01,
            max_value=0.99,
            value=float(saved_thr),
            step=0.005,
            format="%.3f",
            help="Approve loans with calibrated PD below this threshold.",
        )
    with bc2:
        lgd = st.number_input(
            "Loss Given Default",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.05,
            format="%.2f",
        )

    business_table = threshold_business_table(
        y_true_b,
        y_proba_b,
        loan_amounts_b,
        lgd=lgd,
        thresholds=np.array([business_threshold]),
    )
    selected = business_table.iloc[0]
    approved = y_proba_b < business_threshold
    rejected = ~approved
    expected_loss = calculate_expected_loss(y_proba_b, lgd=lgd, ead=loan_amounts_b)

    st.markdown(f"#### Portfolio View at threshold **{business_threshold:.3f}**")
    br1, br2, br3, br4, br5 = st.columns(5)
    br1.metric("Approval Rate", f"{selected['approval_rate']:.1%}")
    br2.metric("Rejected Rate", f"{selected['rejection_rate']:.1%}")
    br3.metric("Default Rate Approved", f"{selected['default_rate_approved']:.1%}")
    br4.metric("Defaults Caught", f"{selected['defaults_caught_rate']:.1%}")
    br5.metric("Expected Loss", f"${selected['expected_loss_approved']:,.0f}")

    el1, el2, el3 = st.columns(3)
    el1.metric("Approved Expected Loss", f"${selected['expected_loss_approved']:,.0f}")
    el2.metric("Rejected Expected Loss", f"${selected['expected_loss_rejected']:,.0f}")
    el3.metric("Total Expected Loss", f"${selected['total_expected_loss']:,.0f}")

    st.caption(
        "Expected Loss = calibrated PD x LGD x loan amount. "
        "Rejected expected loss is risk avoided or routed to review under the selected threshold."
    )

    risk_bands = assign_risk_band(y_proba_b)
    band_order = ["Low", "Medium", "High", "Very High"]
    band_df = (
        pd.DataFrame(
            {
                "risk_band": risk_bands,
                "loan_amount": loan_amounts_b,
                "expected_loss": expected_loss,
            }
        )
        .groupby("risk_band", observed=False)
        .agg(loans=("risk_band", "size"), exposure=("loan_amount", "sum"), expected_loss=("expected_loss", "sum"))
        .reindex(band_order, fill_value=0)
        .reset_index()
    )

    st.markdown("#### Risk-Band Distribution")
    band_fig = go.Figure()
    band_fig.add_trace(
        go.Bar(
            x=band_df["risk_band"],
            y=band_df["loans"],
            marker=dict(color=["#2E7D32", "#F9A825", "#EF6C00", "#C62828"]),
            text=[f"{v:,}" for v in band_df["loans"]],
            textposition="outside",
        )
    )
    band_fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Risk Band",
        yaxis_title="Loan Count",
        showlegend=False,
    )
    st.plotly_chart(band_fig, use_container_width=True)

    st.markdown("#### Threshold Business Table")
    scenario_thresholds = np.round(np.arange(0.05, 0.501, 0.025), 3)
    scenario_table = threshold_business_table(
        y_true_b,
        y_proba_b,
        loan_amounts_b,
        lgd=lgd,
        thresholds=scenario_thresholds,
    )
    st.dataframe(
        scenario_table.style.format(
            {
                "threshold": "{:.3f}",
                "approval_rate": "{:.1%}",
                "rejection_rate": "{:.1%}",
                "default_rate_approved": "{:.1%}",
                "defaults_caught_rate": "{:.1%}",
                "expected_loss_approved": "${:,.0f}",
                "expected_loss_rejected": "${:,.0f}",
                "total_expected_loss": "${:,.0f}",
            }
        ),
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — Batch CSV scoring
# ══════════════════════════════════════════════════════════════════════════

with tab_batch:
    st.subheader("Batch Score — CSV Upload")
    st.markdown(
        "Upload a CSV of loan applications (columns matching the single-loan form). "
        "The app will run each row through the full feature-engineering + model "
        "pipeline and return a downloadable CSV with default probabilities, "
        "predictions, and risk tiers."
    )

    sample_cols = [
        "loan_amnt","term","int_rate","installment","grade","sub_grade",
        "emp_length","home_ownership","annual_inc","verification_status",
        "purpose","addr_state","dti","fico_range_low","revol_util","revol_bal",
        "open_acc","total_acc","inq_last_6mths","delinq_2yrs","pub_rec",
    ]
    with st.expander("Expected column names", expanded=False):
        st.code(", ".join(sample_cols), language="text")
        st.caption("Additional bureau columns are also accepted but optional.")

    uploaded = st.file_uploader("CSV file", type=["csv"], accept_multiple_files=False)

    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not parse CSV: {exc}")
            st.stop()

        st.success(f"Loaded {len(df_in):,} rows × {df_in.shape[1]} columns")
        st.dataframe(df_in.head(5), use_container_width=True)

        n_cap = 50_000
        if len(df_in) > n_cap:
            st.warning(f"Capping to first {n_cap:,} rows for this demo.")
            df_in = df_in.head(n_cap)

        if st.button("Score all rows", type="primary"):
            with st.spinner(f"Scoring {len(df_in):,} rows …"):
                maps = load_encoding_maps()
                engineered = [
                    {col: apply_feature_engineering(row, maps).get(col, 0.0)
                     for col in features}
                    for row in df_in.to_dict(orient="records")
                ]
                X_batch = pd.DataFrame(engineered).astype("float32")
                probs   = load_calibrated_model().predict_proba(X_batch)[:, 1]

                out = df_in.copy()
                out["default_probability"] = probs
                out["prediction"]          = (probs >= saved_thr).astype(int)
                out["risk_tier"]           = [risk_label(p) for p in probs]

            st.markdown("#### Results")
            b1, b2, b3 = st.columns(3)
            b1.metric("Flagged as default", f"{int(out['prediction'].sum()):,}")
            b2.metric("Mean probability", f"{probs.mean():.3f}")
            b3.metric("High / Very High tier", f"{int((out['risk_tier'].isin(['High','Very High'])).sum()):,}")

            st.dataframe(out.head(100), use_container_width=True)

            st.download_button(
                "Download scored CSV",
                data=out.to_csv(index=False).encode(),
                file_name="scored_loans.csv",
                mime="text/csv",
            )
