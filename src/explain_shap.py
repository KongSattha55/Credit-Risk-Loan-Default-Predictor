"""
src/explain_shap.py
───────────────────
SHAP explainability for the LightGBM loan-default model.

Produces:
  1. Global feature importance  — mean |SHAP| bar chart
  2. SHAP summary (beeswarm)    — feature impact direction & distribution
  3. Top-feature dependence plots — how each top feature affects predictions
  4. Single-loan waterfall       — explains one individual prediction
  5. CSV of SHAP values          — for downstream analysis

Outputs (saved to mlruns/artifacts/shap/):
  shap_importance.png
  shap_summary_beeswarm.png
  shap_dependence_<feature>.png  (top 5 features)
  shap_waterfall_sample.png
  shap_values.csv                (sampled, ~5k rows)

Usage:
    python src/explain_shap.py
    python src/explain_shap.py --sample 2000   # smaller sample for speed
    python src/explain_shap.py --no-plots       # CSV only
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[1]
INTERIM_PATH  = PROJECT_ROOT / "data" / "interim"   / "loans_features.parquet"
CLEANED_PATH  = PROJECT_ROOT / "data" / "processed" / "loans_cleaned.parquet"
MODEL_PATH    = PROJECT_ROOT / "mlruns" / "artifacts" / "lightgbm_tuned.txt"
META_PATH     = PROJECT_ROOT / "mlruns" / "artifacts" / "tuning_metadata.json"
OUT_DIR       = PROJECT_ROOT / "mlruns" / "artifacts" / "shap"

import sys as _sys
_sys.path.insert(0, str(PROJECT_ROOT))
from src.inference_fe import ALWAYS_DROP as _ALWAYS_DROP_T, TARGET as _TARGET  # noqa: E402

TARGET       = _TARGET
RANDOM_STATE = 42
ALWAYS_DROP  = list(_ALWAYS_DROP_T)


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def load_test_split(sample: int) -> tuple[pd.DataFrame, pd.Series]:
    """Reconstruct the same 20% test set used during training."""
    if INTERIM_PATH.exists():
        df = pd.read_parquet(INTERIM_PATH)
        print(f"  Data source : interim (feature-engineered)  {df.shape}")
    elif CLEANED_PATH.exists():
        df = pd.read_parquet(CLEANED_PATH)
        print(f"  Data source : processed (cleaned)  {df.shape}")
    else:
        raise FileNotFoundError("Run src/data_cleaning.py first.")

    drop_cols    = [c for c in (ALWAYS_DROP + ["issue_d"]) if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols + [TARGET]]

    X = df[feature_cols].copy()
    y = df[TARGET].astype("int32")

    # Cast categoricals so LightGBM is happy
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category")

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    # Sub-sample for speed — SHAP on 1.3M rows is slow
    if sample and len(X_test) > sample:
        idx = np.random.default_rng(RANDOM_STATE).choice(len(X_test), sample, replace=False)
        X_test = X_test.iloc[idx].reset_index(drop=True)
        y_test = y_test.iloc[idx].reset_index(drop=True)
        print(f"  Sub-sampled : {sample} rows from test set")

    return X_test, y_test


def load_model() -> lgb.Booster:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run src/tune_lightgbm.py first.")
    return lgb.Booster(model_file=str(MODEL_PATH))


# ══════════════════════════════════════════════════════════════════════════
# Plot helpers
# ══════════════════════════════════════════════════════════════════════════

def save_importance(shap_values: np.ndarray, feature_names: list[str], out_dir: Path) -> None:
    mean_abs = np.abs(shap_values).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1][:30]   # top 30
    top_feat = [feature_names[i] for i in order]
    top_vals = mean_abs[order]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(top_feat[::-1], top_vals[::-1], color="#2196F3")
    ax.set_xlabel("Mean |SHAP value|  (impact on model output)")
    ax.set_title("Top 30 Features — Global SHAP Importance", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = out_dir / "shap_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def save_beeswarm(shap_values: np.ndarray, X: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    shap.summary_plot(
        shap_values, X,
        max_display=25,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP Summary — Feature Impact on Default Probability", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "shap_summary_beeswarm.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def save_dependence_plots(
    shap_values: np.ndarray, X: pd.DataFrame, out_dir: Path, n_top: int = 5
) -> None:
    mean_abs  = np.abs(shap_values).mean(axis=0)
    top_feats = X.columns[np.argsort(mean_abs)[::-1][:n_top]].tolist()

    for feat in top_feats:
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(
            feat, shap_values, X,
            ax=ax, show=False,
            dot_size=8, alpha=0.4,
        )
        ax.set_title(f"SHAP Dependence — {feat}", fontsize=11, fontweight="bold")
        plt.tight_layout()
        safe = feat.replace("/", "_").replace(" ", "_")
        path = out_dir / f"shap_dependence_{safe}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path.name}")


def save_waterfall(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    y: pd.Series,
    out_dir: Path,
) -> None:
    """Waterfall plot for the highest-confidence predicted default."""
    model    = explainer.model
    proba    = model.predict(X)
    # pick the sample with the highest predicted default probability
    idx      = int(np.argmax(proba))
    sv       = explainer(X.iloc[[idx]])

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(sv[0], max_display=20, show=False)
    plt.title(
        f"SHAP Waterfall — Sample #{idx}  "
        f"(pred={proba[idx]:.3f}, actual={'Default' if y.iloc[idx] else 'Paid'})",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    path = out_dir / "shap_waterfall_sample.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}  (sample #{idx}, prob={proba[idx]:.3f})")


def save_csv(shap_values: np.ndarray, feature_names: list[str], out_dir: Path) -> None:
    df_shap = pd.DataFrame(shap_values, columns=feature_names)
    path    = out_dir / "shap_values.csv"
    df_shap.to_csv(path, index=False)
    print(f"  Saved: {path.name}  ({df_shap.shape[0]} rows × {df_shap.shape[1]} cols)")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main(sample: int, no_plots: bool) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  SHAP Explainability — Loan Default Predictor")
    print("=" * 60)

    # 1. Load model & data
    print("\n[1/5] Loading model …")
    model = load_model()
    meta  = json.loads(META_PATH.read_text())
    print(f"  Model threshold : {meta['threshold']}")
    print(f"  Val ROC-AUC     : {meta['val_roc_auc']}")

    print("\n[2/5] Loading test data …")
    X_test, y_test = load_test_split(sample)
    print(f"  Test shape : {X_test.shape}")

    # 2. Build SHAP explainer
    print("\n[3/5] Computing SHAP values  (TreeExplainer — fast) …")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)   # shape: (n_samples, n_features)
    print(f"  SHAP array : {shap_values.shape}")

    # 3. Summary stats
    mean_abs     = np.abs(shap_values).mean(axis=0)
    top_idx      = np.argsort(mean_abs)[::-1][:10]
    feature_names = X_test.columns.tolist()

    print("\n  Top 10 features by mean |SHAP|:")
    print(f"  {'Feature':<35} {'Mean |SHAP|':>12}")
    print("  " + "-" * 50)
    for i in top_idx:
        print(f"  {feature_names[i]:<35} {mean_abs[i]:>12.5f}")

    # 4. Save CSV
    print("\n[4/5] Saving SHAP values CSV …")
    save_csv(shap_values, feature_names, OUT_DIR)

    # 5. Plots
    if no_plots:
        print("\n[5/5] Skipping plots (--no-plots).")
    else:
        print("\n[5/5] Generating plots …")
        save_importance(shap_values, feature_names, OUT_DIR)
        save_beeswarm(shap_values, X_test, OUT_DIR)
        save_dependence_plots(shap_values, X_test, OUT_DIR, n_top=5)
        save_waterfall(explainer, X_test, y_test, OUT_DIR)

    print(f"\n{'=' * 60}")
    print(f"  Done.  All outputs → {OUT_DIR.relative_to(PROJECT_ROOT)}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP explainability for loan default model")
    parser.add_argument(
        "--sample", type=int, default=5000,
        help="Number of test-set rows to use for SHAP (default: 5000)"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation, output CSV only"
    )
    args = parser.parse_args()
    main(sample=args.sample, no_plots=args.no_plots)
