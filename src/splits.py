"""Time-based dataset splitting for LendingClub modeling."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


TRAIN_END = pd.Timestamp("2016-01-01")
VAL_END = pd.Timestamp("2017-01-01")
SPLIT_STRATEGY = "time_based_issue_d"


@dataclass(frozen=True)
class SplitSummary:
    name: str
    rows: int
    default_rate: float
    start_date: pd.Timestamp | None
    end_date: pd.Timestamp | None


def ensure_issue_datetime(issue_d: pd.Series) -> pd.Series:
    """Convert issue_d to datetime and fail fast if any dates cannot be parsed."""
    dates = pd.to_datetime(issue_d, errors="coerce")
    if dates.isna().any():
        raise ValueError(f"issue_d contains {int(dates.isna().sum()):,} unparseable values")
    return dates


def time_split_masks(issue_d: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return train/validation/test masks using fixed issue_d calendar cutoffs."""
    dates = ensure_issue_datetime(issue_d)
    train_mask = dates < TRAIN_END
    val_mask = (dates >= TRAIN_END) & (dates < VAL_END)
    test_mask = dates >= VAL_END

    for name, mask in (("train", train_mask), ("validation", val_mask), ("test", test_mask)):
        if not bool(mask.any()):
            raise ValueError(f"Time split produced an empty {name} set")

    return train_mask, val_mask, test_mask


def split_xy_by_issue_date(
    X: pd.DataFrame,
    y: pd.Series,
    issue_d: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split aligned X/y into train, validation, and test windows by issue_d."""
    train_mask, val_mask, test_mask = time_split_masks(issue_d)
    return (
        X.loc[train_mask].copy(),
        X.loc[val_mask].copy(),
        X.loc[test_mask].copy(),
        y.loc[train_mask].copy(),
        y.loc[val_mask].copy(),
        y.loc[test_mask].copy(),
    )


def split_summaries(issue_d: pd.Series, y: pd.Series) -> list[SplitSummary]:
    """Return row counts, default rates, and date ranges for each split."""
    dates = ensure_issue_datetime(issue_d)
    masks = dict(zip(("train", "validation", "test"), time_split_masks(dates)))
    summaries: list[SplitSummary] = []
    for name, mask in masks.items():
        split_dates = dates.loc[mask]
        summaries.append(
            SplitSummary(
                name=name,
                rows=int(mask.sum()),
                default_rate=float(y.loc[mask].mean()),
                start_date=split_dates.min(),
                end_date=split_dates.max(),
            )
        )
    return summaries


def print_split_summary(issue_d: pd.Series, y: pd.Series, indent: str = "      ") -> None:
    """Print counts and default rates for the fixed time split."""
    for summary in split_summaries(issue_d, y):
        print(
            f"{indent}{summary.name.title():10s}: {summary.rows:>9,} rows  |  "
            f"default rate: {summary.default_rate * 100:5.2f}%  |  "
            f"{summary.start_date.date()} to {summary.end_date.date()}"
        )
