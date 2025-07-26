"""Data loading, window slicing, and simulated drift injection.

The production-simulation idea: take 307k Home Credit applications, sort
by SK_ID_CURR (proxy arrival time), split into n_weeks roughly-equal
slices. Week 0 is the initial training set. Weeks 1..n are "fresh data
that arrives". From `drift_injection_start_week` onward, AMT_INCOME_TOTAL
is shifted up to simulate inflation; the fairness audit also sees a
slightly different gender mix.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from modelgate.config import settings
from modelgate.utils import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = {settings.target_col, settings.id_col}


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    path = Path(path) if path else settings.raw_dir / "application_train.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Download application_train.csv from Kaggle's "
            "home-credit-default-risk competition and place it in data/raw/."
        )
    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows x {df.shape[1]} cols")
    return df


def validate_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[settings.target_col].isna().any():
        raise ValueError("TARGET column contains NaNs")
    if not set(df[settings.target_col].unique()).issubset({0, 1}):
        raise ValueError("TARGET must be binary 0/1")


def add_age_bucket(df: pd.DataFrame) -> pd.DataFrame:
    if "DAYS_BIRTH" not in df.columns:
        return df
    age = (-df["DAYS_BIRTH"] / 365.25).clip(lower=0)
    df = df.copy()
    df["AGE_YEARS"] = age
    df["AGE_BUCKET"] = pd.cut(
        age,
        bins=[0, 25, 35, 45, 55, 65, 120],
        labels=["<25", "25-35", "35-45", "45-55", "55-65", "65+"],
        include_lowest=True,
    ).astype(str)
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace({"XNA": np.nan, "XAP": np.nan})
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("missing")
    return df


def split_into_weeks(df: pd.DataFrame, n_weeks: int | None = None) -> list[pd.DataFrame]:
    """Sort by SK_ID_CURR and chunk into n_weeks roughly-equal slices."""
    n_weeks = n_weeks if n_weeks is not None else settings.n_weeks
    sorted_df = df.sort_values(settings.id_col).reset_index(drop=True)
    chunks = np.array_split(sorted_df, n_weeks)
    weeks = [c.reset_index(drop=True) for c in chunks]
    logger.info(f"Sliced {len(df):,} rows into {n_weeks} weekly windows of ~{len(weeks[0]):,} rows each")
    return weeks


# FIXME: drift injection here is brutal -- 10% income shift is much
# bigger than what you'd see in real data. fine for demoing the gate
# triggers; not realistic for calibrating thresholds.
def inject_drift(week_df: pd.DataFrame, week_idx: int) -> pd.DataFrame:
    """If we're past the drift start, multiply income up and add noise.
    Returns a copy. No-op for early weeks."""
    if week_idx < settings.drift_injection_start_week:
        return week_df
    df = week_df.copy()
    rng = np.random.default_rng(settings.random_state + week_idx)
    if "AMT_INCOME_TOTAL" in df.columns:
        noise = rng.normal(0, 0.02, size=len(df))
        df["AMT_INCOME_TOTAL"] = df["AMT_INCOME_TOTAL"] * (settings.drift_income_multiplier + noise)
        logger.info(f"Week {week_idx}: applied drift to AMT_INCOME_TOTAL (multiplier ~{settings.drift_income_multiplier})")
    return df


def build_window(weeks: list[pd.DataFrame], end_week: int, rolling: int | None = None) -> pd.DataFrame:
    """Concatenate the last `rolling` weeks ending at end_week (inclusive).
    Each week's data is drift-injected first.
    """
    rolling = rolling if rolling is not None else settings.rolling_window_weeks
    start = max(0, end_week - rolling + 1)
    pieces = [inject_drift(weeks[i], i) for i in range(start, end_week + 1)]
    return pd.concat(pieces, ignore_index=True)


def prepare_window_for_training(
    weeks: list[pd.DataFrame], end_week: int, val_size: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a rolling window and carve out a within-window validation slice
    using a stratified split. Returns (train, val)."""
    window = build_window(weeks, end_week)
    window = add_age_bucket(window)
    window = basic_clean(window)
    train, val = train_test_split(
        window,
        test_size=val_size,
        random_state=settings.random_state,
        stratify=window[settings.target_col],
    )
    logger.info(f"Window ending week {end_week}: train {len(train):,}, val {len(val):,}")
    return train.reset_index(drop=True), val.reset_index(drop=True)


def prepare_holdout(weeks: list[pd.DataFrame], week_idx: int) -> pd.DataFrame:
    """The 'next week' holdout used to evaluate a challenger against the champion.
    Returns the (drift-injected) week_idx slice, fully preprocessed except features.
    """
    df = inject_drift(weeks[week_idx], week_idx)
    df = add_age_bucket(df)
    df = basic_clean(df)
    return df.reset_index(drop=True)
