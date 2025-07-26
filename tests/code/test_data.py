import numpy as np
import pandas as pd
import pytest

from modelgate.config import settings
from modelgate.data import add_age_bucket, basic_clean, inject_drift, split_into_weeks, validate_schema


def _make_df(n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        settings.id_col: np.arange(n),
        settings.target_col: rng.integers(0, 2, size=n),
        "DAYS_BIRTH": -rng.integers(20 * 365, 70 * 365, size=n),
        "AMT_INCOME_TOTAL": rng.uniform(20_000, 200_000, size=n),
        "AMT_CREDIT": rng.uniform(50_000, 600_000, size=n),
        "CODE_GENDER": rng.choice(["M", "F", "XNA"], size=n),
    })


def test_validate_schema_accepts_valid():
    validate_schema(_make_df())


def test_validate_schema_rejects_missing_target():
    df = _make_df().drop(columns=[settings.target_col])
    with pytest.raises(ValueError):
        validate_schema(df)


def test_age_bucket_added():
    df = add_age_bucket(_make_df())
    assert "AGE_BUCKET" in df.columns
    assert "AGE_YEARS" in df.columns


def test_basic_clean_replaces_sentinels():
    df = basic_clean(_make_df())
    assert (df["CODE_GENDER"] != "XNA").all()


def test_split_into_weeks_count_and_order():
    df = _make_df(n=800)
    weeks = split_into_weeks(df, n_weeks=8)
    assert len(weeks) == 8
    # weeks should be in SK_ID_CURR order
    last_id = -1
    for w in weeks:
        first_id = int(w[settings.id_col].iloc[0])
        assert first_id > last_id
        last_id = int(w[settings.id_col].iloc[-1])


def test_inject_drift_only_after_start_week():
    df = _make_df()
    original_mean = df["AMT_INCOME_TOTAL"].mean()
    before = inject_drift(df, week_idx=settings.drift_injection_start_week - 1)
    assert np.isclose(before["AMT_INCOME_TOTAL"].mean(), original_mean)
    after = inject_drift(df, week_idx=settings.drift_injection_start_week)
    assert after["AMT_INCOME_TOTAL"].mean() > original_mean
