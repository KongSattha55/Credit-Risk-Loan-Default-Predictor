"""
SHAP explainability for the current leakage-free LightGBM model.

Outputs:
    artifacts/shap_summary_beeswarm.png
    artifacts/shap_summary_bar.png
    artifacts/shap_top_features.csv
    artifacts/shap_local_example_waterfall.png
    artifacts/shap_local_example.json

Usage:
    python src/explain_shap.py --sample 2000
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.explain_utils import global_business_interpretation, summarize_shap_local  # noqa: E402
from src.inference_fe import risk_label  # noqa: E402
from src.leakage import TARGET, feature_columns, leakage_columns_present  # noqa: E402
from src.splits import time_split_masks  # noqa: E402

INTERIM_PATH = PROJECT_ROOT / "data" / "interim" / "loans_features.parquet"
CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / "loans_cleaned.parquet"

MODEL_PATH = PROJECT_ROOT / "mlruns" / "artifacts" / "lightgbm_model.pkl"
META_PATH = PROJECT_ROOT / "mlruns" / "artifacts" / "lightgbm_metadata.json"
CALIBRATOR_PATH = PROJECT_ROOT / "mlruns" / "artifacts" / "calibrator.pkl"
MLFLOW_URI = str(PROJECT_ROOT / "mlruns")

OUT_DIR = PROJECT_ROOT / "artifacts"
RANDOM_STATE = 42


def _require_file(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}\n{hint}")


def load_metadata() -> dict[str, Any]:
    _require_file(META_PATH, "Run `python src/train.py --model lightgbm` first.")
    return json.loads(META_PATH.read_text())


def load_model():
    _require_file(MODEL_PATH, "Run `python src/train.py --model lightgbm` first.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def load_calibrator():
    if not CALIBRATOR_PATH.exists():
        return None
    with open(CALIBRATOR_PATH, "rb") as f:
        payload = pickle.load(f)
    return payload.get("calibrator") if isinstance(payload, dict) else payload


def predict_pd(model, X: pd.DataFrame, calibrator=None) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.predict(X)
    probs = np.asarray(probs, dtype=float)
    if calibrator is not None:
        probs = calibrator.transform(probs)
    return probs


def load_test_sample(sample: int, metadata: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    if INTERIM_PATH.exists():
        df = pd.read_parquet(INTERIM_PATH)
        print(f"  Data source: {INTERIM_PATH.relative_to(PROJECT_ROOT)} {df.shape}")
    elif CLEANED_PATH.exists():
        df = pd.read_parquet(CLEANED_PATH)
        print(f"  Data source: {CLEANED_PATH.relative_to(PROJECT_ROOT)} {df.shape}")
    else:
        raise FileNotFoundError("No data found. Run `python src/data_cleaning.py` and `python src/feature_engineering.py` first.")

    if "issue_d" not in df.columns:
        raise ValueError("SHAP requires `issue_d` for the time-based test split. Re-run data cleaning and feature engineering.")

    metadata_features = list(metadata.get("feature_columns", []))
    if not metadata_features:
        raise ValueError("Model metadata does not contain `feature_columns`.")

    leaked = leakage_columns_present(metadata_features)
    if leaked:
        raise ValueError(f"Model metadata contains leakage features: {leaked}")

    available_features = feature_columns(df.columns, extra_drop=["issue_d"])
    missing = [feature for feature in metadata_features if feature not in available_features]
    if missing:
        raise ValueError(f"Data is missing model features required for SHAP: {missing[:10]}")

    _, _, test_mask = time_split_masks(df["issue_d"])
    X_test = df.loc[test_mask, metadata_features].copy()
    y_test = df.loc[test_mask, TARGET].astype("int32")

    if sample and len(X_test) > sample:
        rng = np.random.default_rng(RANDOM_STATE)
        positions = rng.choice(len(X_test), sample, replace=False)
        X_test = X_test.iloc[positions].reset_index(drop=True)
        y_test = y_test.iloc[positions].reset_index(drop=True)
        print(f"  Sampled {sample:,} rows from the 2017+ test set")

    return X_test, y_test


def positive_class_shap_values(raw_values) -> np.ndarray:
    if isinstance(raw_values, list):
        raw_values = raw_values[1] if len(raw_values) > 1 else raw_values[0]
    values = np.asarray(raw_values)
    if values.ndim == 3:
        if values.shape[-1] == 2:
            values = values[:, :, 1]
        elif values.shape[0] == 2:
            values = values[1]
    if values.ndim != 2:
        raise ValueError(f"Expected 2D SHAP values, got shape {values.shape}")
    return values


def positive_expected_value(explainer) -> float:
    expected = explainer.expected_value
    if isinstance(expected, (list, tuple, np.ndarray)):
        arr = np.asarray(expected)
        return float(arr[1] if arr.size > 1 else arr[0])
    return float(expected)


def direction_note(feature_values: pd.Series, shap_values: np.ndarray) -> str:
    if feature_values.nunique(dropna=False) <= 1:
        return "Direction is not stable in this sample."
    corr = np.corrcoef(feature_values.astype(float), shap_values.astype(float))[0, 1]
    if np.isnan(corr) or abs(corr) < 0.05:
        return "Effect varies by borrower context."
    if corr > 0:
        return "Higher values tend to increase predicted default risk."
    return "Higher values tend to reduce predicted default risk."


def save_global_artifacts(shap_values: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
    feature_names = X.columns.tolist()
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    top_rows = []
    for rank, idx in enumerate(order[:30], start=1):
        feature = feature_names[idx]
        top_rows.append(
            {
                "feature": feature,
                "mean_abs_shap": float(mean_abs[idx]),
                "rank": rank,
                "direction_note": direction_note(X[feature], shap_values[:, idx]),
                "business_interpretation": global_business_interpretation(feature),
            }
        )
    top_df = pd.DataFrame(top_rows)
    top_df.to_csv(OUT_DIR / "shap_top_features.csv", index=False)

    plt.figure(figsize=(11, 8))
    shap.summary_plot(shap_values, X, max_display=25, show=False, plot_size=None)
    plt.title("Global Explanation: Factors Driving Default Risk")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    top_bar = top_df.sort_values("rank", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_bar["feature"], top_bar["mean_abs_shap"], color="#2F6B9A")
    ax.set_xlabel("Average contribution magnitude")
    ax.set_title("Top Global Risk Drivers")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return top_df


def save_local_artifacts(
    explainer,
    shap_values: np.ndarray,
    X: pd.DataFrame,
    y: pd.Series,
    predicted_pd: np.ndarray,
) -> dict[str, Any]:
    idx = int(np.argmax(predicted_pd))
    row = X.iloc[idx]
    row_shap = shap_values[idx]
    base_value = positive_expected_value(explainer)
    summary = summarize_shap_local(row_shap, row.values, X.columns.tolist(), top_n=5)

    local_json = {
        "sample_index": idx,
        "actual_default": int(y.iloc[idx]),
        "predicted_pd": float(predicted_pd[idx]),
        "risk_label": risk_label(float(predicted_pd[idx])),
        "base_value": float(np.mean(predicted_pd)),
        "raw_shap_base_value": base_value,
        "top_positive_drivers": summary["top_positive_drivers"],
        "top_negative_drivers": summary["top_negative_drivers"],
        "plain_english_explanation": summary["explanation_text"],
    }
    (OUT_DIR / "shap_local_example.json").write_text(json.dumps(local_json, indent=2))

    explanation = shap.Explanation(
        values=row_shap,
        base_values=base_value,
        data=row.values,
        feature_names=X.columns.tolist(),
    )
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.title(f"Individual Loan Explanation (PD={predicted_pd[idx]:.3f})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_local_example_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()

    return local_json


def log_to_mlflow(paths: list[Path], metadata: dict[str, Any], sample: int) -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("loan-default-explainability")
    with mlflow.start_run(run_name="shap_explainability") as run:
        mlflow.log_params(
            {
                "model_metadata_run_id": metadata.get("mlflow_run_id"),
                "split_strategy": metadata.get("split_strategy"),
                "sample": sample,
                "n_features": len(metadata.get("feature_columns", [])),
            }
        )
        for path in paths:
            mlflow.log_artifact(str(path), artifact_path="shap")
        print(f"  MLflow run: {run.info.run_id}")


def main(sample: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("  SHAP Explainability - Credit Risk / Loan Default Predictor")
    print("=" * 68)

    metadata = load_metadata()
    model = load_model()
    calibrator = load_calibrator()
    if calibrator is None:
        print("  Calibration: no calibrator found; explaining uncalibrated model probabilities.")
    else:
        print("  Calibration: calibrator found; local predicted_pd is calibrated.")

    print(f"  Model artifact: {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  Metadata: {META_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  Split strategy: {metadata.get('split_strategy')}")
    print(f"  Features: {len(metadata.get('feature_columns', []))}")

    X_test, y_test = load_test_sample(sample, metadata)
    print(f"  SHAP sample shape: {X_test.shape}")

    explainer = shap.TreeExplainer(model)
    shap_values = positive_class_shap_values(explainer.shap_values(X_test))
    if shap_values.shape[1] != len(metadata["feature_columns"]):
        raise ValueError("SHAP feature count does not match model metadata feature count.")
    if X_test.columns.tolist() != metadata["feature_columns"]:
        raise ValueError("SHAP feature names do not match model metadata feature names.")

    predicted_pd = predict_pd(model, X_test, calibrator=calibrator)

    top_df = save_global_artifacts(shap_values, X_test)
    local = save_local_artifacts(explainer, shap_values, X_test, y_test, predicted_pd)

    artifact_paths = [
        OUT_DIR / "shap_summary_beeswarm.png",
        OUT_DIR / "shap_summary_bar.png",
        OUT_DIR / "shap_top_features.csv",
        OUT_DIR / "shap_local_example_waterfall.png",
        OUT_DIR / "shap_local_example.json",
    ]
    log_to_mlflow(artifact_paths, metadata, sample)

    print("\nTop global features:")
    print(top_df[["rank", "feature", "mean_abs_shap", "direction_note"]].head(10).to_string(index=False))
    print("\nLocal example:")
    print(f"  predicted_pd: {local['predicted_pd']:.4f}")
    print(f"  risk_label  : {local['risk_label']}")
    print(f"\nDone. Artifacts written to {OUT_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SHAP explanations for the latest LightGBM model")
    parser.add_argument("--sample", type=int, default=2000, help="Number of test-set rows to explain")
    args = parser.parse_args()
    main(sample=args.sample)
