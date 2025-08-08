from datetime import datetime, timedelta, timezone

from modelgate.config import settings
from modelgate.evaluate import FairnessMetrics, Metrics
from modelgate.gate import decide


def _m(auc=0.78, pr=0.30, brier=0.06, pos=0.08, n=10_000) -> Metrics:
    return Metrics(roc_auc=auc, pr_auc=pr, brier=brier, positive_rate=pos, n=n)


def _f(dp: dict | None = None, eo: dict | None = None) -> FairnessMetrics:
    return FairnessMetrics(
        dp_ratio_by_attr=dp or {"CODE_GENDER": 0.4, "AGE_BUCKET": 0.3},
        eo_ratio_by_attr=eo or {"CODE_GENDER": 0.35, "AGE_BUCKET": 0.3},
        threshold=settings.audit_threshold,
    )


def test_bootstrap_promotes_when_above_floor():
    d = decide(challenger=_m(), challenger_fair=_f(), champion=None, champion_fair=None, train_drift_psi=0.0)
    assert d.promote


def test_bootstrap_rejects_below_auc_floor():
    d = decide(challenger=_m(auc=0.65), challenger_fair=_f(), champion=None, champion_fair=None, train_drift_psi=0.0)
    assert not d.promote
    assert any("auc_floor" in r for r in d.reasons)


def test_challenger_wins_strictly_better_promotes():
    chal = _m(auc=0.80)
    champ = _m(auc=0.78)
    d = decide(
        challenger=chal, challenger_fair=_f(),
        champion=champ, champion_fair=_f(),
        train_drift_psi=0.1,
        champion_trained_at=datetime.now(tz=timezone.utc) - timedelta(days=3),
    )
    assert d.promote


def test_challenger_tied_but_champion_fresh_skips():
    chal = _m(auc=0.78)
    champ = _m(auc=0.78)
    d = decide(
        challenger=chal, challenger_fair=_f(),
        champion=champ, champion_fair=_f(),
        train_drift_psi=0.1,
        champion_trained_at=datetime.now(tz=timezone.utc) - timedelta(days=3),
    )
    # tie + champion fresh -> strict_improvement_or_stale fails
    assert not d.promote
    assert any("strict_improvement" in r for r in d.reasons)


def test_challenger_tied_but_champion_stale_promotes():
    chal = _m(auc=0.78)
    champ = _m(auc=0.78)
    d = decide(
        challenger=chal, challenger_fair=_f(),
        champion=champ, champion_fair=_f(),
        train_drift_psi=0.1,
        champion_trained_at=datetime.now(tz=timezone.utc) - timedelta(days=20),
    )
    assert d.promote


def test_fairness_regression_blocks_even_with_better_auc():
    chal = _m(auc=0.82)  # better AUC
    champ = _m(auc=0.78)
    bad_fair = _f(dp={"CODE_GENDER": 0.02, "AGE_BUCKET": 0.01})  # below floor 0.05
    d = decide(
        challenger=chal, challenger_fair=bad_fair,
        champion=champ, champion_fair=_f(),
        train_drift_psi=0.1,
        champion_trained_at=datetime.now(tz=timezone.utc),
    )
    assert not d.promote
    assert any("fairness_dp_floor" in r for r in d.reasons)


def test_drift_too_large_blocks():
    chal = _m(auc=0.80)
    champ = _m(auc=0.78)
    d = decide(
        challenger=chal, challenger_fair=_f(),
        champion=champ, champion_fair=_f(),
        train_drift_psi=3.0,  # exceeds 2.0 max
        champion_trained_at=datetime.now(tz=timezone.utc),
    )
    assert not d.promote
    assert any("train_drift_psi" in r for r in d.reasons)


def test_pr_auc_regression_blocks():
    chal = _m(auc=0.80, pr=0.20)  # better AUC, worse PR-AUC
    champ = _m(auc=0.78, pr=0.30)
    d = decide(
        challenger=chal, challenger_fair=_f(),
        champion=champ, champion_fair=_f(),
        train_drift_psi=0.1,
        champion_trained_at=datetime.now(tz=timezone.utc),
    )
    assert not d.promote
    assert any("pr_auc_no_regression" in r for r in d.reasons)
