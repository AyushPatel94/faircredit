"""Behavioural tests for the gate's behaviour over a sequence of weeks.

These don't run the real training loop (too slow). Instead they construct
metrics + fairness objects that mimic the shape of the 8-week sim and
assert that the gate makes the decisions we want.

The point: lock in the *contract* the README advertises -- promotions
happen on improvement, fairness regressions block, drift blocks.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from modelgate.evaluate import FairnessMetrics, Metrics
from modelgate.gate import decide


def _m(auc: float, pr: float = 0.25, brier: float = 0.07) -> Metrics:
    return Metrics(roc_auc=auc, pr_auc=pr, brier=brier, positive_rate=0.08, n=10_000)


def _f(dp: float = 0.10, eo: float = 0.10) -> FairnessMetrics:
    return FairnessMetrics(
        dp_ratio_by_attr={"CODE_GENDER": dp, "AGE_BUCKET": dp + 0.01},
        eo_ratio_by_attr={"CODE_GENDER": eo, "AGE_BUCKET": eo + 0.01},
        threshold=0.15,
    )


NOW = datetime.now(tz=timezone.utc)


def test_simulation_promotes_on_clear_improvement():
    """Week 1-style: challenger strictly improves on every metric."""
    d = decide(
        challenger=_m(auc=0.760),
        challenger_fair=_f(dp=0.12, eo=0.12),
        champion=_m(auc=0.745),
        champion_fair=_f(dp=0.11, eo=0.11),
        train_drift_psi=1.0,
        champion_trained_at=NOW - timedelta(days=3),
        now=NOW,
    )
    assert d.promote, d.reasons


def test_simulation_blocks_fairness_regression():
    """Week 2-style: AUC is fine but DP ratio collapses."""
    d = decide(
        challenger=_m(auc=0.755),
        challenger_fair=_f(dp=0.033, eo=0.030),  # both below 0.05 floor
        champion=_m(auc=0.748),
        champion_fair=_f(dp=0.079, eo=0.078),
        train_drift_psi=0.5,
        champion_trained_at=NOW - timedelta(days=3),
        now=NOW,
    )
    assert not d.promote
    assert any("fairness_dp_floor" in r for r in d.reasons)
    assert any("fairness_eo_floor" in r for r in d.reasons)


def test_simulation_blocks_when_drift_high():
    """Week 4-style: drift PSI exceeds the gate."""
    d = decide(
        challenger=_m(auc=0.760),
        challenger_fair=_f(dp=0.08, eo=0.08),
        champion=_m(auc=0.760),
        champion_fair=_f(dp=0.08, eo=0.08),
        train_drift_psi=2.65,
        champion_trained_at=NOW - timedelta(days=3),
        now=NOW,
    )
    assert not d.promote
    assert any("train_drift_psi" in r for r in d.reasons)


def test_simulation_high_auc_does_not_override_safety():
    """Week 7-style: challenger AUC is much higher but fairness EO is below
    floor AND drift is huge. The gate must still reject."""
    d = decide(
        challenger=_m(auc=0.865),  # huge gain
        challenger_fair=_f(dp=0.051, eo=0.038),  # EO below floor
        champion=_m(auc=0.752),
        champion_fair=_f(dp=0.046, eo=0.045),
        train_drift_psi=12.4,
        champion_trained_at=NOW - timedelta(days=3),
        now=NOW,
    )
    assert not d.promote
    # at minimum fairness_eo_floor and train_drift_psi must be in the failure list
    fail_names = " ".join(d.reasons)
    assert "fairness_eo_floor" in fail_names
    assert "train_drift_psi" in fail_names


@pytest.mark.parametrize("days_old,expected", [(3, False), (20, True)])
def test_strict_improvement_unless_champion_stale(days_old, expected):
    """Tie + fresh champion = skip; tie + stale champion = promote."""
    d = decide(
        challenger=_m(auc=0.780),
        challenger_fair=_f(),
        champion=_m(auc=0.780),
        champion_fair=_f(),
        train_drift_psi=0.5,
        champion_trained_at=NOW - timedelta(days=days_old),
        now=NOW,
    )
    assert d.promote == expected
