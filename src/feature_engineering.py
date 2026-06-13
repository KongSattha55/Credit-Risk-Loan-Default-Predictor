"""
Feature engineering pipeline for LendingClub Loan Default Predictor.
Input:  data/processed/loans_cleaned.parquet  (1.37M rows × 85 cols)
Output: data/interim/loans_features.parquet

Transformations (in order):
  1. Ordinal encoding    — grade (A=1…G=7), sub_grade (A1=1…G5=35)
  2. Categorical encoding — verification_status, home_ownership (ordinal)
                            purpose (smoothed target encoding, K=300)
                            Binary flags: term_60, initial_list_status,
                            application_type, disbursement_method
  3. High-cardinality    — addr_state (smoothed target encoding, K=500)
                            emp_title  (log-frequency encoding)
  4. Log1p transforms    — 9 skewed numeric columns (originals kept)
  5. Ratio / interaction — 9 engineered ratio features
  6. Drop redundant cols — collinear duplicates + raw strings post-encoding
  7. Validation          — null check, inf check, dtype audit

Usage:
    python src/feature_engineering.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE    = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
DATA_IN = BASE / "data" / "processed" / "loans_cleaned.parquet"
DATA_OUT= BASE / "data" / "interim"   / "loans_features.parquet"
DATA_OUT.parent.mkdir(parents=True, exist_ok=True)

from src.leakage import LEAKAGE_COLS  # noqa: E402
from src.splits import SPLIT_STRATEGY, time_split_masks  # noqa: E402

TARGET = "default"


# ══════════════════════════════════════════════════════════════════════════
# Encoding maps  (public — imported by tests and api)
# ══════════════════════════════════════════════════════════════════════════

GRADE_MAP: dict[str, int] = {
    "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7,
}

SUBGRADE_MAP: dict[str, int] = {
    f"{g}{n}": i * 5 + n
    for i, g in enumerate("ABCDEFG")
    for n in range(1, 6)
}

VERIFICATION_MAP: dict[str, int] = {
    "Not Verified": 0,
    "Source Verified": 1,
    "Verified": 2,
}

HOME_OWNERSHIP_MAP: dict[str, int] = {
    "MORTGAGE": 3,
    "OWN":      2,
    "RENT":     1,
    "OTHER":    0,
    "NONE":     0,
    "ANY":      0,
}

# Columns replaced by log-transformed versions (originals kept alongside)
LOG_TRANSFORM_COLS: list[str] = [
    "annual_inc",
    "revol_bal",
    "tot_coll_amt",
    "delinq_amnt",
    "total_rev_hi_lim",
    "tot_hi_cred_lim",
    "total_bal_ex_mort",
    "total_bc_limit",
    "total_il_high_credit_limit",
]

# Collinear / leaky duplicates
COLLINEAR_DROPS: list[str] = [
    "funded_amnt",      # corr ≈ 1.0 with loan_amnt
    "funded_amnt_inv",  # corr ≈ 1.0 with loan_amnt
    "fico_range_high",  # corr > 0.999 with fico_range_low
]

# Raw string columns that are superseded by engineered encodings
RAW_CAT_DROPS: list[str] = [
    "grade",               # → grade_enc
    "sub_grade",           # → sub_grade_enc
    "verification_status", # → verification_status_enc
    "home_ownership",      # → home_ownership_enc
    "purpose",             # → purpose_rate_enc
    "addr_state",          # → state_default_rate
    "emp_title",           # → emp_title_log_freq
    "initial_list_status", # → initial_list_status_enc
    "application_type",    # → application_type_joint
    "disbursement_method", # → disbursement_direct
    "term",                # → term_60  (only two values: 36 / 60)
]


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def _safe_div(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    """Element-wise division; replaces NaN / ±inf with `fill`."""
    result = a / b
    return result.replace([np.inf, -np.inf], np.nan).fillna(fill)


def _smoothed_target_rate(
    series: pd.Series,
    target: pd.Series,
    k: float,
    global_rate: float,
    fit_mask: pd.Series | np.ndarray | None = None,
) -> pd.Series:
    """
    Smoothed (James-Stein-style) target encoding.
        smooth_rate = (group_sum + k × global_rate) / (group_count + k)

    k controls shrinkage toward the global mean:
      large k → groups with few observations get pulled strongly toward global mean.

    fit_mask : bool array-like of len(series). If provided, group stats are
    computed ONLY on rows where mask is True (train rows). Encodings are then
    broadcast to all rows. This prevents target leakage from val/test into the
    learned mapping.
    """
    if fit_mask is None:
        fit_series, fit_target = series, target
    else:
        mask = np.asarray(fit_mask, dtype=bool)
        fit_series = series[mask]
        fit_target = target[mask]

    stats = (
        pd.DataFrame({"group": fit_series, "y": fit_target})
        .groupby("group")["y"]
        .agg(["sum", "count"])
        .assign(smooth_rate=lambda x: (x["sum"] + k * global_rate) / (x["count"] + k))
    )
    # Unseen groups in val/test fall back to global rate
    return series.map(stats["smooth_rate"]).fillna(global_rate).astype("float32")


# ══════════════════════════════════════════════════════════════════════════
# Step 1 — Ordinal encoding: grade & sub_grade
# ══════════════════════════════════════════════════════════════════════════

def _encode_grade(df: pd.DataFrame) -> pd.DataFrame:
    """
    grade A–G → integer 1–7 (higher = riskier, confirmed by EDA default rate plot).
    sub_grade A1–G5 → integer 1–35.
    """
    if "grade" in df.columns:
        df["grade_enc"] = df["grade"].map(GRADE_MAP).astype("float32")

    if "sub_grade" in df.columns:
        df["sub_grade_enc"] = df["sub_grade"].map(SUBGRADE_MAP).astype("float32")

    return df


# ══════════════════════════════════════════════════════════════════════════
# Step 2 — Low-cardinality categorical encoding
# ══════════════════════════════════════════════════════════════════════════

def _encode_low_cardinality(
    df: pd.DataFrame,
    global_rate: float,
    fit_mask: pd.Series | np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    verification_status : ordinal (Not Verified=0, Source Verified=1, Verified=2)
    home_ownership      : ordinal stability proxy (MORTGAGE=3 … OTHER=0)
    purpose             : smoothed target encoding (K=300)
    term                : binary flag  term_60 = 1 if 60-month else 0
    initial_list_status : binary flag  (w=1, f=0)
    application_type    : binary flag  (Joint App=1, Individual=0)
    disbursement_method : binary flag  (DirectPay=1, Cash=0)
    """
    if "verification_status" in df.columns:
        df["verification_status_enc"] = (
            df["verification_status"].map(VERIFICATION_MAP).astype("float32")
        )

    if "home_ownership" in df.columns:
        df["home_ownership_enc"] = (
            df["home_ownership"].map(HOME_OWNERSHIP_MAP).fillna(0).astype("float32")
        )

    if "purpose" in df.columns:
        df["purpose_rate_enc"] = _smoothed_target_rate(
            df["purpose"], df[TARGET], k=300, global_rate=global_rate,
            fit_mask=fit_mask,
        )

    if "term" in df.columns:
        df["term_60"] = (df["term"] == 60).astype("float32")

    if "initial_list_status" in df.columns:
        df["initial_list_status_enc"] = (
            df["initial_list_status"] == "w"
        ).astype("float32")

    if "application_type" in df.columns:
        df["application_type_joint"] = (
            df["application_type"] == "Joint App"
        ).astype("float32")

    if "disbursement_method" in df.columns:
        df["disbursement_direct"] = (
            df["disbursement_method"] == "DirectPay"
        ).astype("float32")

    # Save purpose encoding map for inference
    encoding_maps: dict = {}
    if "purpose" in df.columns:
        encoding_maps["purpose_rate_enc"] = (
            df.groupby("purpose")["purpose_rate_enc"].first().to_dict()
        )
        encoding_maps["purpose_rate_enc_fallback"] = float(global_rate)

    return df, encoding_maps


# ══════════════════════════════════════════════════════════════════════════
# Step 3 — High-cardinality columns: addr_state, emp_title
# ══════════════════════════════════════════════════════════════════════════

def _encode_high_cardinality(
    df: pd.DataFrame,
    global_rate: float,
    fit_mask: pd.Series | np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    addr_state (51 unique) : smoothed target encoding (K=500).
                             Larger K than purpose because states have more rows
                             so we can trust group means more.

    emp_title (383K unique): frequency encoding (log-scaled). NOTE: emp_title is
                             dropped in ALWAYS_DROP at training time, so the
                             mapping is not persisted to encoding_maps.json (was
                             previously bloating the file to ~19 MB unused).

    fit_mask : bool mask restricting group-statistic computation to train rows
               only. Prevents target leakage from val/test.
    """
    encoding_maps: dict = {}

    if "addr_state" in df.columns:
        df["state_default_rate"] = _smoothed_target_rate(
            df["addr_state"], df[TARGET], k=500, global_rate=global_rate,
            fit_mask=fit_mask,
        )
        encoding_maps["state_default_rate"] = (
            df.groupby("addr_state")["state_default_rate"].first().to_dict()
        )
        encoding_maps["state_default_rate_fallback"] = float(global_rate)

    if "emp_title" in df.columns:
        # Frequency on fit rows only (ties into leakage prevention)
        fit_slice = df["emp_title"] if fit_mask is None else df.loc[np.asarray(fit_mask, dtype=bool), "emp_title"]
        freq = fit_slice.value_counts(normalize=True)
        df["emp_title_freq"]     = df["emp_title"].map(freq).fillna(freq.min() if len(freq) else 0.0).astype("float32")
        df["emp_title_log_freq"] = np.log1p(df["emp_title_freq"]).astype("float32")
        df = df.drop(columns=["emp_title_freq"])
        # emp_title is dropped before training — we skip persisting the map to
        # save ~19 MB of disk / API cold-start memory. Re-enable here if you
        # ever remove emp_title from ALWAYS_DROP.

    return df, encoding_maps


# ══════════════════════════════════════════════════════════════════════════
# Step 4 — Log1p transforms for skewed numerics
# ══════════════════════════════════════════════════════════════════════════

def _log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log1p to each skewed column.  Original columns are KEPT so the
    model can choose between raw and log-scaled versions.
    Suffix convention: <col>_log  (e.g. annual_inc → annual_inc_log).
    """
    transformed: list[str] = []
    for col in LOG_TRANSFORM_COLS:
        if col not in df.columns:
            continue
        df[f"{col}_log"] = np.log1p(df[col].clip(lower=0)).astype("float32")
        transformed.append(col)
    return df


# ══════════════════════════════════════════════════════════════════════════
# Step 5 — Ratio & interaction features
# ══════════════════════════════════════════════════════════════════════════

def _ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    All engineered ratios confirmed to have |Pearson corr with default| > 0.05.

    Feature                 Formula                             Rationale
    ──────────────────────  ──────────────────────────────────  ─────────────────────────────
    loan_to_income          loan_amnt / annual_inc              Affordability proxy
    installment_to_income   installment / (annual_inc/12)       Monthly payment burden
    credit_util_total       revol_bal / (tot_hi_cred_lim + 1)  Overall credit utilisation
    bc_util                 bc_used / (total_bc_limit + 1)      Bankcard utilisation
    int_rate_x_term         int_rate × term                     Total interest burden proxy
    fico_dti_score          fico_range_low / (dti + 1)          Combined creditworthiness
    derog_ratio             (pub_rec + collections) / (total_acc + 1)  Derogatory density
    inq_per_acc             inq_last_6mths / (open_acc + 1)    Inquiry intensity
    revolving_debt_share    revol_bal / (total_bal_ex_mort + 1) Revolving vs total debt
    """
    # ── Affordability ──────────────────────────────────────────────────────
    if "loan_amnt" in df.columns and "annual_inc" in df.columns:
        df["loan_to_income"] = _safe_div(
            df["loan_amnt"], df["annual_inc"]
        ).astype("float32")

    if "installment" in df.columns and "annual_inc" in df.columns:
        df["installment_to_income"] = _safe_div(
            df["installment"], df["annual_inc"] / 12
        ).astype("float32")

    # ── Credit utilisation ─────────────────────────────────────────────────
    if "revol_bal" in df.columns and "tot_hi_cred_lim" in df.columns:
        df["credit_util_total"] = _safe_div(
            df["revol_bal"], df["tot_hi_cred_lim"] + 1
        ).astype("float32")

    if "total_bc_limit" in df.columns and "bc_open_to_buy" in df.columns:
        bc_used = (df["total_bc_limit"] - df["bc_open_to_buy"]).clip(lower=0)
        df["bc_util_ratio"] = _safe_div(
            bc_used, df["total_bc_limit"] + 1
        ).astype("float32")

    # ── Interest burden ────────────────────────────────────────────────────
    if "int_rate" in df.columns and "term" in df.columns:
        # Use raw term (36/60) for this multiplication — meaningful magnitude
        df["int_rate_x_term"] = (df["int_rate"] * df["term"]).astype("float32")

    # ── Combined creditworthiness ──────────────────────────────────────────
    if "fico_range_low" in df.columns and "dti" in df.columns:
        df["fico_dti_score"] = _safe_div(
            df["fico_range_low"], df["dti"] + 1
        ).astype("float32")

    # ── Derogatory mark density ────────────────────────────────────────────
    if "pub_rec" in df.columns and "total_acc" in df.columns:
        derogs = df["pub_rec"].fillna(0)
        if "collections_12_mths_ex_med" in df.columns:
            derogs = derogs + df["collections_12_mths_ex_med"].fillna(0)
        df["derog_ratio"] = _safe_div(derogs, df["total_acc"] + 1).astype("float32")

    # ── Inquiry intensity ──────────────────────────────────────────────────
    if "inq_last_6mths" in df.columns and "open_acc" in df.columns:
        df["inq_per_acc"] = _safe_div(
            df["inq_last_6mths"], df["open_acc"] + 1
        ).astype("float32")

    # ── Revolving debt share ───────────────────────────────────────────────
    if "revol_bal" in df.columns and "total_bal_ex_mort" in df.columns:
        df["revolving_debt_share"] = _safe_div(
            df["revol_bal"], df["total_bal_ex_mort"] + 1
        ).astype("float32")

    return df


# ══════════════════════════════════════════════════════════════════════════
# Step 6 — Drop redundant columns
# ══════════════════════════════════════════════════════════════════════════

def _drop_redundant(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove:
      • Collinear duplicates (funded_amnt ≈ loan_amnt, fico_range_high ≈ fico_range_low)
      • Raw string columns that have been replaced by numeric encodings
      • Any leakage columns that should never reach the model (safety net)
    Log-transformed columns: originals are KEPT so the model / regularisation
    can select between raw and log scales.
    """
    explicit = COLLINEAR_DROPS + RAW_CAT_DROPS
    # Leakage columns must never reach feature engineering output.  This guard
    # catches any column from LEAKAGE_COLS that was accidentally left in the
    # data (e.g. loan_age_months created by an older version of data_cleaning.py).
    leakage_present = [c for c in LEAKAGE_COLS if c in df.columns]
    all_drops = list(set(c for c in explicit + leakage_present if c in df.columns))
    return df.drop(columns=all_drops)


# ══════════════════════════════════════════════════════════════════════════
# Step 7 — Validation
# ══════════════════════════════════════════════════════════════════════════

def _validate(df: pd.DataFrame) -> None:
    """
    Assert the output meets quality requirements.
    Raises AssertionError with a descriptive message on failure.
    """
    # No remaining string / category columns
    str_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    assert not str_cols, (
        f"String/category columns remain after encoding: {str_cols}\n"
        "All categoricals must be numeric before saving."
    )

    # No missing values
    null_counts = df.isnull().sum()
    null_cols   = null_counts[null_counts > 0]
    assert null_cols.empty, (
        f"Missing values found after feature engineering:\n{null_cols.to_string()}"
    )

    # No ±inf values
    inf_mask = df.replace([np.inf, -np.inf], np.nan).isnull()
    orig_null = df.isnull()
    inf_cols  = (inf_mask & ~orig_null).sum()
    inf_cols  = inf_cols[inf_cols > 0]
    assert inf_cols.empty, (
        f"±Inf values found after feature engineering:\n{inf_cols.to_string()}"
    )

    # Target unchanged — values must be drawn from {0, 1} (subset OK for small fixtures)
    assert set(df[TARGET].unique()).issubset({0, 1}), \
        f"Target column contains values outside {{0, 1}}: {sorted(df[TARGET].unique())}"

    # No leakage columns may survive into the output
    surviving_leakage = [c for c in LEAKAGE_COLS if c in df.columns]
    assert not surviving_leakage, (
        f"Leakage columns present in feature-engineering output: {surviving_leakage}. "
        "These must be removed before this data is used for training."
    )


# ══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════

def engineer_features(
    df: pd.DataFrame,
    fit_mask: pd.Series | np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Full feature engineering pipeline.  Accepts a cleaned loans DataFrame
    and returns a fully numeric, model-ready DataFrame plus encoding maps
    needed to replicate the transforms at inference time.

    Parameters
    ----------
    df : pd.DataFrame  — output of data_cleaning.py (must contain TARGET column)
    fit_mask : bool array (len(df)) indicating TRAIN rows. Target encodings and
        frequency encodings are fit on these rows only and broadcast to all rows.
        If None, fits on all rows — only safe for EDA, NOT for training.

    Returns
    -------
    (pd.DataFrame, dict) — feature-engineered DataFrame + encoding maps
    """
    df = df.copy()
    if fit_mask is None:
        global_rate = float(df[TARGET].mean())
    else:
        mask = np.asarray(fit_mask, dtype=bool)
        global_rate = float(df.loc[mask, TARGET].mean())

    n_cols_in = df.shape[1]

    encoding_maps: dict = {
        "global_default_rate": global_rate,
        "grade":               GRADE_MAP,
        "sub_grade":           SUBGRADE_MAP,
        "verification_status": VERIFICATION_MAP,
        "home_ownership":      HOME_OWNERSHIP_MAP,
    }

    # ── Steps ─────────────────────────────────────────────────────────────
    print("  [1/7] Ordinal encoding: grade, sub_grade …")
    df = _encode_grade(df)

    print("  [2/7] Low-cardinality categorical encoding …")
    df, low_maps = _encode_low_cardinality(df, global_rate, fit_mask=fit_mask)
    encoding_maps.update(low_maps)

    print("  [3/7] High-cardinality encoding: addr_state, emp_title …")
    df, high_maps = _encode_high_cardinality(df, global_rate, fit_mask=fit_mask)
    encoding_maps.update(high_maps)

    print("  [4/7] Log1p transforms …")
    df = _log_transform(df)

    print("  [5/7] Ratio & interaction features …")
    df = _ratio_features(df)

    print("  [6/7] Dropping redundant / collinear columns …")
    df = _drop_redundant(df)

    print("  [7/7] Validation …")
    _validate(df)

    n_cols_out = df.shape[1]
    print(f"        Columns: {n_cols_in} → {n_cols_out}  "
          f"(+{n_cols_out - n_cols_in} net engineered)")

    return df, encoding_maps


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def build_train_mask(
    df: pd.DataFrame,
    strategy: str = SPLIT_STRATEGY,
) -> np.ndarray:
    """
    Build a boolean mask marking TRAIN rows (True) vs val+test rows (False).
    Must match the split logic used downstream in tune_lightgbm.py so that
    encodings learned here are used on the correct train subset at training.

    strategy:
      "time_based_issue_d" — issue_d < 2016-01-01
    """
    if strategy == SPLIT_STRATEGY:
        if "issue_d" not in df.columns:
            raise ValueError("time-based split requires issue_d column from data_cleaning")
        train_mask, _, _ = time_split_masks(df["issue_d"])
        return train_mask.to_numpy()

    raise ValueError(f"Unknown split strategy: {strategy!r}")


def main() -> None:
    if not DATA_IN.exists():
        raise FileNotFoundError(
            f"Input file not found: {DATA_IN}\n"
            "Run src/data_cleaning.py first."
        )

    print(f"Loading {DATA_IN.name} …")
    df = pd.read_parquet(DATA_IN)
    print(f"  Input  : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Default rate: {df[TARGET].mean():.3%}")
    print()

    # Build train mask so target encodings are leakage-free.
    # Default strategy matches downstream model training: train on loans before 2016.
    import os
    split_strategy = os.environ.get("SPLIT_STRATEGY", SPLIT_STRATEGY)
    print(f"Building train mask (strategy={split_strategy}) …")
    train_mask = build_train_mask(df, strategy=split_strategy)
    print(f"  Train rows: {train_mask.sum():,}  /  Held-out: {(~train_mask).sum():,}")
    print()

    print("Running feature engineering pipeline …")
    df_out, encoding_maps = engineer_features(df, fit_mask=train_mask)
    encoding_maps["fit_split_strategy"] = split_strategy

    # ── Summary ────────────────────────────────────────────────────────────
    print()
    print("=== FEATURE ENGINEERING SUMMARY ===")
    print(f"  Rows              : {len(df_out):,}")
    print(f"  Total columns     : {df_out.shape[1]}  (including target)")
    print(f"  Target column     : '{TARGET}'  (rate: {df_out[TARGET].mean():.3%})")
    print()

    # New engineered features
    new_features = [
        # Ordinal
        "grade_enc", "sub_grade_enc",
        # Low-cardinality encodings
        "verification_status_enc", "home_ownership_enc", "purpose_rate_enc",
        "term_60", "initial_list_status_enc", "application_type_joint",
        "disbursement_direct",
        # High-cardinality encodings
        "state_default_rate", "emp_title_log_freq",
        # Log transforms
        *[f"{c}_log" for c in LOG_TRANSFORM_COLS],
        # Ratios
        "loan_to_income", "installment_to_income", "credit_util_total",
        "bc_util_ratio", "int_rate_x_term", "fico_dti_score",
        "derog_ratio", "inq_per_acc", "revolving_debt_share",
    ]
    present = [c for c in new_features if c in df_out.columns]
    print(f"  Engineered features added : {len(present)}")
    print()

    # Correlation of new features with target
    corr = (
        df_out[present + [TARGET]]
        .corr()[TARGET]
        .drop(TARGET)
        .sort_values(key=abs, ascending=False)
    )
    print("  New features — |Pearson corr with default| (top 15):")
    for feat_name, c in corr.head(15).items():
        bar   = "█" * int(abs(c) * 40)
        sign  = "+" if c > 0 else "-"
        print(f"    {feat_name:<30}  {sign}{abs(c):.4f}  {bar}")

    # Null / inf check output
    null_total = df_out.isnull().sum().sum()
    print(f"\n  Null values   : {null_total}")
    print(f"  String cols   : {len(df_out.select_dtypes(include=['object', 'category']).columns)}")

    # Column inventory
    num_cols = df_out.select_dtypes(include="number").columns.tolist()
    print(f"  Numeric cols  : {len(num_cols)}")
    print()

    # Save encoding maps for API inference
    MAPS_OUT = BASE / "mlruns" / "artifacts" / "encoding_maps.json"
    MAPS_OUT.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(MAPS_OUT, "w") as f:
        json.dump(encoding_maps, f, indent=2)
    print(f"Encoding maps → {MAPS_OUT.relative_to(BASE)}")

    print(f"Saving to {DATA_OUT} …")
    df_out.to_parquet(DATA_OUT, index=False, engine="pyarrow", compression="snappy")

    size_mb = DATA_OUT.stat().st_size / 1e6
    print(f"  Done.  {DATA_OUT.name}  ({size_mb:.1f} MB)")
    print()
    print("All columns in output file:")
    for i, col in enumerate(df_out.columns, 1):
        print(f"  {i:3d}.  {col:<45}  {str(df_out[col].dtype)}")


if __name__ == "__main__":
    main()
