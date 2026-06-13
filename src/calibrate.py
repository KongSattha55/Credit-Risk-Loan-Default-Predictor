"""
src/calibrate.py
────────────────
Probability calibration for the tuned LightGBM model.

Compares three variants on the held-out test set:
  1. Raw LightGBM        — direct booster output
  2. Sigmoid calibration — PreFitCalibratedClassifier(method="sigmoid")
  3. Isotonic calibration— PreFitCalibratedClassifier(method="isotonic")

sklearn 1.8 removed cv="prefit" from CalibratedClassifierCV.
PreFitCalibratedClassifier provides the identical behaviour:
  • the base LightGBM model is already trained (not re-fitted),
  • only the calibration layer is fit on provided (X_val, y_val),
  • CalibratedClassifierCV is used internally for the fit step.

Split strategy (src/splits.py — centralized time-based split):
  • Train  : issue_d < 2016-01-01
  • Val    : 2016-01-01 <= issue_d < 2017-01-01  ← calibrators fit here
  • Test   : issue_d >= 2017-01-01               ← all metrics reported here

Usage:
    python src/calibrate.py
    python src/calibrate.py --n-bins 15   # calibration curve resolution

Outputs:
    • artifacts/calibration_curve.png
    • artifacts/calibration_results.json
    • models/lightgbm_raw.pkl
    • models/lightgbm_calibrated_sigmoid.pkl
    • models/lightgbm_calibrated_isotonic.pkl
    • tuning_metadata.json updated with calibration results
    • MLflow run logged under "loan-default-calibration" experiment
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration_classes import LGBMBoosterWrapper, PreFitCalibratedClassifier  # noqa: E402
from src.leakage import TARGET, leakage_columns_present   # noqa: E402
from src.splits import print_split_summary, split_xy_by_issue_date  # noqa: E402

INTERIM_PATH  = PROJECT_ROOT / "data" / "interim"   / "loans_features.parquet"
CLEANED_PATH  = PROJECT_ROOT / "data" / "processed" / "loans_cleaned.parquet"
MODEL_PATH    = PROJECT_ROOT / "mlruns" / "artifacts" / "lightgbm_tuned.txt"
META_PATH     = PROJECT_ROOT / "mlruns" / "artifacts" / "tuning_metadata.json"
MODELS_DIR    = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MLFLOW_URI    = str(PROJECT_ROOT / "mlruns")

RANDOM_STATE  = 42
EXPERIMENT    = "loan-default-calibration"


# ══════════════════════════════════════════════════════════════════════════
# Data loading — uses src/splits.py for reproducible time-based splits
# ══════════════════════════════════════════════════════════════════════════

def _load_splits() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series,    pd.Series,    pd.Series,
    list[str], str,
]:
    """
    Load feature-engineered (or cleaned) data and apply the centralized
    time-based split from src/splits.py.

    Feature columns are read from tuning_metadata.json so the booster
    receives exactly the same columns it was trained on.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, source
    """
    if INTERIM_PATH.exists():
        df     = pd.read_parquet(INTERIM_PATH)
        source = "interim (feature-engineered)"
    elif CLEANED_PATH.exists():
        df     = pd.read_parquet(CLEANED_PATH)
        source = "processed (cleaned)"
    else:
        raise FileNotFoundError(
            "No data found. Run src/data_cleaning.py first, "
            "then optionally src/feature_engineering.py."
        )

    if "issue_d" not in df.columns:
        raise ValueError(
            "issue_d column not found. Re-run src/data_cleaning.py to preserve it."
        )

    meta      = json.loads(META_PATH.read_text())

    feat_cols = list(meta.get("feature_columns", []))
    if not feat_cols:
        raise ValueError("Tuning metadata does not contain `feature_columns`.")

    leaks = leakage_columns_present(feat_cols)
    if leaks:
        raise ValueError(
            f"Tuning metadata contains leakage features: {leaks}. "
            "Retrain the LightGBM model before calibration."
        )

    missing = [col for col in feat_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Data is missing {len(missing)} feature(s) required by the tuned model: "
            f"{missing[:10]}. Regenerate features or retrain the model."
        )

    X       = df[feat_cols].copy()
    y       = df[TARGET].astype("int32")
    issue_d = df["issue_d"]

    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category")

    X_train, X_val, X_test, y_train, y_val, y_test = split_xy_by_issue_date(
        X, y, issue_d
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, feat_cols, source


# ══════════════════════════════════════════════════════════════════════════
# Metrics helper
# ══════════════════════════════════════════════════════════════════════════

def _compute_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    return {
        "roc_auc":  round(float(roc_auc_score(y_true, y_proba)),           6),
        "pr_auc":   round(float(average_precision_score(y_true, y_proba)), 6),
        "brier":    round(float(brier_score_loss(y_true, y_proba)),        6),
        "log_loss": round(float(log_loss(y_true, y_proba)),                6),
    }


# ══════════════════════════════════════════════════════════════════════════
# Calibration plot
# ══════════════════════════════════════════════════════════════════════════

def _plot_calibration(
    y_true:   np.ndarray,
    probas:   dict[str, np.ndarray],
    metrics:  dict[str, dict],
    n_bins:   int,
    out_path: Path,
) -> None:
    """
    Two-panel reliability diagram + probability distribution histogram.
    """
    fig, (ax_cal, ax_hist) = plt.subplots(1, 2, figsize=(14, 5))

    palette = {
        "Raw LightGBM":        "#E53935",
        "Sigmoid Calibration": "#1E88E5",
        "Isotonic Calibration":"#43A047",
    }
    dashes = {
        "Raw LightGBM":        "--",
        "Sigmoid Calibration": "-",
        "Isotonic Calibration":"-",
    }

    for label, proba in probas.items():
        m = metrics[label]
        display_name = (
            f"{label}  "
            f"(AUC={m['roc_auc']:.4f}, Brier={m['brier']:.4f})"
        )
        CalibrationDisplay.from_predictions(
            y_true, proba,
            n_bins=n_bins,
            name=display_name,
            ax=ax_cal,
            color=palette.get(label, "grey"),
            linestyle=dashes.get(label, "-"),
        )

    ax_cal.set_title(
        "Reliability Diagram — Test Set (2017–2018)\n"
        "Well-calibrated model follows the diagonal",
        fontsize=11, fontweight="bold",
    )
    ax_cal.set_xlabel("Mean Predicted Probability (PD)")
    ax_cal.set_ylabel("Fraction of Positives (Observed Default Rate)")
    ax_cal.legend(fontsize=8, loc="upper left")

    for label, proba in probas.items():
        ax_hist.hist(
            proba, bins=50, alpha=0.45,
            label=label, color=palette.get(label, "grey"),
            density=True,
        )
    ax_hist.axvline(
        y_true.mean(), color="black", linestyle=":", linewidth=1.5,
        label=f"Base rate ({y_true.mean():.2%})",
    )
    ax_hist.set_title(
        "Predicted Probability Distribution — Test Set",
        fontsize=11, fontweight="bold",
    )
    ax_hist.set_xlabel("Predicted Default Probability (PD)")
    ax_hist.set_ylabel("Density")
    ax_hist.legend(fontsize=8)

    plt.suptitle(
        "Probability Calibration — LendingClub Loan Default Predictor\n"
        "Val: 2016  |  Test: 2017–2018  |  Split: time_based_issue_d",
        fontsize=11, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {out_path.relative_to(PROJECT_ROOT)}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main(n_bins: int = 10) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  Probability Calibration — Loan Default Predictor")
    print("=" * 64)

    # ── 1. Load trained model ─────────────────────────────────────────────
    print("\n[1/6] Loading model …")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run src/tune_lightgbm.py first."
        )
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    meta    = json.loads(META_PATH.read_text())
    best_it = meta.get("best_iteration", 0) or 0
    print(f"      Best iteration : {best_it}")
    print(f"      Val ROC-AUC    : {meta.get('val_roc_auc', 'N/A')}")

    leaks = leakage_columns_present(meta.get("feature_columns", []))
    if leaks:
        raise ValueError(
            f"Model artifact uses post-origination feature(s): {leaks}. "
            "Retrain after removing them before calibration."
        )

    # ── 2. Load data + time-based split ───────────────────────────────────
    print("\n[2/6] Applying time-based split (src/splits.py) …")
    df_tmp = (
        pd.read_parquet(INTERIM_PATH) if INTERIM_PATH.exists()
        else pd.read_parquet(CLEANED_PATH)
    )
    print_split_summary(df_tmp["issue_d"], df_tmp[TARGET])
    del df_tmp

    _, X_val, X_test, _, y_val, y_test, feat_cols, source = _load_splits()
    print(f"\n      Data source : {source}")
    print(f"      Features    : {len(feat_cols)}")
    print(f"      Val  size   : {len(X_val):,}")
    print(f"      Test size   : {len(X_test):,}")
    print(f"      Test default rate : {y_test.mean():.3%}")

    # ── 3. Build wrapper and fit calibrators on val set ───────────────────
    print("\n[3/6] Fitting calibrators on 2016 validation set …")
    wrapper = LGBMBoosterWrapper(booster, best_iteration=best_it)

    # PreFitCalibratedClassifier = CalibratedClassifierCV(cv="prefit") equivalent
    # cv="prefit" was removed in sklearn 1.8; PreFitCalibratedClassifier is the
    # direct replacement with identical semantics.
    cal_sigmoid  = PreFitCalibratedClassifier(wrapper, method="sigmoid")
    cal_isotonic = PreFitCalibratedClassifier(wrapper, method="isotonic")

    cal_sigmoid.fit(X_val,  y_val)
    cal_isotonic.fit(X_val, y_val)
    print("      Sigmoid  (Platt scaling)    — fitted on 2016 val set")
    print("      Isotonic (non-parametric)   — fitted on 2016 val set")

    # ── 4. Evaluate on held-out test set ──────────────────────────────────
    print("\n[4/6] Evaluating on 2017–2018 test set …")
    raw_proba = booster.predict(X_test, num_iteration=best_it)
    sig_proba = cal_sigmoid.predict_proba(X_test)[:, 1]
    iso_proba = cal_isotonic.predict_proba(X_test)[:, 1]

    raw_m = _compute_metrics(y_test.values, raw_proba)
    sig_m = _compute_metrics(y_test.values, sig_proba)
    iso_m = _compute_metrics(y_test.values, iso_proba)
    all_m = {
        "Raw LightGBM":        raw_m,
        "Sigmoid Calibration": sig_m,
        "Isotonic Calibration":iso_m,
    }

    # ── Print comparison table ────────────────────────────────────────────
    col_w = 24
    sep   = "  " + "─" * 66
    hdr   = (
        f"  {'Model':<{col_w}}"
        f"{'ROC-AUC':>10}"
        f"{'PR-AUC':>10}"
        f"{'Brier':>10}"
        f"{'Log Loss':>10}"
    )

    print()
    print("=" * 68)
    print("  CALIBRATION RESULTS  (test set: 2017–2018 loans)")
    print("=" * 68)
    print(hdr)
    print(sep)
    for label, m in all_m.items():
        print(
            f"  {label:<{col_w}}"
            f"{m['roc_auc']:>10.5f}"
            f"{m['pr_auc']:>10.5f}"
            f"{m['brier']:>10.5f}"
            f"{m['log_loss']:>10.5f}"
        )
    print(sep)

    candidates  = {"sigmoid": sig_m, "isotonic": iso_m}
    best_method = min(candidates,
                      key=lambda k: (candidates[k]["brier"], candidates[k]["log_loss"]))
    best_m      = candidates[best_method]

    print(f"\n  Best calibration method   : {best_method.upper()}")
    print(f"  Brier  improvement vs raw : {raw_m['brier']    - best_m['brier']:+.6f}")
    print(f"  LogLoss improvement vs raw: {raw_m['log_loss'] - best_m['log_loss']:+.6f}")
    print(f"  ROC-AUC delta vs raw      : {best_m['roc_auc'] - raw_m['roc_auc']:+.6f}")
    print("=" * 68)

    # ── 5. Calibration plot ───────────────────────────────────────────────
    print("\n[5/6] Generating calibration plots …")
    _plot_calibration(
        y_true=y_test.values,
        probas={
            "Raw LightGBM":        raw_proba,
            "Sigmoid Calibration": sig_proba,
            "Isotonic Calibration":iso_proba,
        },
        metrics=all_m,
        n_bins=n_bins,
        out_path=ARTIFACTS_DIR / "calibration_curve.png",
    )

    # ── 6. Save models + persist results + log to MLflow ─────────────────
    print("\n[6/6] Saving models and logging to MLflow …")

    saved_objs = {
        "lightgbm_raw":                 wrapper,
        "lightgbm_calibrated_sigmoid":  cal_sigmoid,
        "lightgbm_calibrated_isotonic": cal_isotonic,
    }
    for name, obj in saved_objs.items():
        path = MODELS_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print(f"  models/{name}.pkl")

    # Standalone JSON for tests and downstream scripts
    cal_results = {
        "split_strategy":    "time_based_issue_d",
        "val_period":        "2016",
        "test_period":       "2017-2018",
        "n_val":             int(len(X_val)),
        "n_test":            int(len(X_test)),
        "test_default_rate": round(float(y_test.mean()), 6),
        "best_method":       best_method,
        "n_bins":            n_bins,
        "leakage_in_model":  leaks,
        "raw":               raw_m,
        "sigmoid":           sig_m,
        "isotonic":          iso_m,
    }
    results_path = ARTIFACTS_DIR / "calibration_results.json"
    results_path.write_text(json.dumps(cal_results, indent=2))
    print(f"  artifacts/calibration_results.json")

    meta.update({
        "calibration_split_strategy":       "time_based_issue_d",
        "calibration_n_bins":               n_bins,
        "calibration_best_method":          best_method,
        "calibration_brier_raw":            raw_m["brier"],
        "calibration_brier_sigmoid":        sig_m["brier"],
        "calibration_brier_isotonic":       iso_m["brier"],
        "calibration_logloss_raw":          raw_m["log_loss"],
        "calibration_logloss_sigmoid":      sig_m["log_loss"],
        "calibration_logloss_isotonic":     iso_m["log_loss"],
        "calibration_roc_auc_raw":          raw_m["roc_auc"],
        "calibration_roc_auc_sigmoid":      sig_m["roc_auc"],
        "calibration_roc_auc_isotonic":     iso_m["roc_auc"],
        "calibration_pr_auc_raw":           raw_m["pr_auc"],
        "calibration_pr_auc_sigmoid":       sig_m["pr_auc"],
        "calibration_pr_auc_isotonic":      iso_m["pr_auc"],
    })
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"  tuning_metadata.json updated")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name="calibration_comparison") as run:
        mlflow.log_params({
            "n_bins":             n_bins,
            "best_method":        best_method,
            "data_source":        source,
            "split_strategy":     "time_based_issue_d",
            "n_val":              len(X_val),
            "n_test":             len(X_test),
            "base_run_id":        meta.get("mlflow_run_id", ""),
            "leakage_in_model":   str(leaks),
        })
        for prefix, m in [
            ("raw",      raw_m),
            ("sigmoid",  sig_m),
            ("isotonic", iso_m),
        ]:
            mlflow.log_metrics({
                f"{prefix}_roc_auc":  m["roc_auc"],
                f"{prefix}_pr_auc":   m["pr_auc"],
                f"{prefix}_brier":    m["brier"],
                f"{prefix}_log_loss": m["log_loss"],
            })
        mlflow.log_artifact(str(ARTIFACTS_DIR / "calibration_curve.png"),
                            artifact_path="calibration")
        mlflow.log_artifact(str(results_path),
                            artifact_path="calibration")
        for name in saved_objs:
            mlflow.log_artifact(str(MODELS_DIR / f"{name}.pkl"),
                                artifact_path="models")
        run_id = run.info.run_id

    print(f"\n  MLflow run ID : {run_id}")
    print(f"  Experiment    : {EXPERIMENT}")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare raw vs calibrated LightGBM probabilities"
    )
    parser.add_argument(
        "--n-bins", type=int, default=10,
        help="Bins for CalibrationDisplay reliability diagram (default: 10)",
    )
    args = parser.parse_args()
    main(n_bins=args.n_bins)
