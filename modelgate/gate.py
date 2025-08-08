"""Promotion gate: decide whether the challenger replaces the champion.

Eight checks (see PLAN.md for the rationale of each). All must pass to
promote. Every failure is recorded with the actual values so the audit
log is informative.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from modelgate.config import settings
from modelgate.evaluate import FairnessMetrics, Metrics


@dataclass
class PromotionDecision:
    promote: bool
    reasons: list[str] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "promote": self.promote,
            "reasons": self.reasons,
            "checks": self.checks,
        }


def _check(name: str, passed: bool, detail: dict, reasons: list[str], checks: list[dict]) -> None:
    checks.append({"name": name, "passed": passed, **detail})
    if not passed:
        reasons.append(f"{name}: {detail}")


# NOTE: thresholds in config.py are tuned for the Home Credit
# distribution. moving to a different dataset (or production data)
# means re-fitting them against the observed champion baseline.
def decide(
    challenger: Metrics,
    challenger_fair: FairnessMetrics,
    champion: Metrics | None,
    champion_fair: FairnessMetrics | None,
    train_drift_psi: float,
    champion_trained_at: datetime | None = None,
    now: datetime | None = None,
) -> PromotionDecision:
    reasons: list[str] = []
    checks: list[dict] = []
    now = now or datetime.now(tz=timezone.utc)

    # cold start -- if there is no champion, anything that meets the floor wins
    if champion is None or champion_fair is None:
        _check(
            "auc_floor",
            challenger.roc_auc >= settings.auc_floor,
            {"challenger_auc": challenger.roc_auc, "floor": settings.auc_floor},
            reasons, checks,
        )
        return PromotionDecision(promote=len(reasons) == 0, reasons=reasons, checks=checks)

    # 1. AUC regression tolerance
    _check(
        "auc_no_regression",
        challenger.roc_auc >= champion.roc_auc - settings.auc_epsilon,
        {
            "challenger": challenger.roc_auc,
            "champion": champion.roc_auc,
            "epsilon": settings.auc_epsilon,
        },
        reasons, checks,
    )

    # 2. AUC absolute floor
    _check(
        "auc_floor",
        challenger.roc_auc >= settings.auc_floor,
        {"challenger_auc": challenger.roc_auc, "floor": settings.auc_floor},
        reasons, checks,
    )

    # 3. PR-AUC regression tolerance
    _check(
        "pr_auc_no_regression",
        challenger.pr_auc >= champion.pr_auc - settings.pr_auc_epsilon,
        {
            "challenger": challenger.pr_auc,
            "champion": champion.pr_auc,
            "epsilon": settings.pr_auc_epsilon,
        },
        reasons, checks,
    )

    # 4. Brier regression tolerance
    _check(
        "brier_no_regression",
        challenger.brier <= champion.brier + settings.brier_epsilon,
        {
            "challenger": challenger.brier,
            "champion": champion.brier,
            "epsilon": settings.brier_epsilon,
        },
        reasons, checks,
    )

    # 5. Fairness DP regression -- min DP ratio across attrs should not drop
    chal_dp = min(challenger_fair.dp_ratio_by_attr.values()) if challenger_fair.dp_ratio_by_attr else 0.0
    champ_dp = min(champion_fair.dp_ratio_by_attr.values()) if champion_fair.dp_ratio_by_attr else 0.0
    _check(
        "fairness_dp_floor",
        chal_dp >= settings.dp_ratio_min,
        {
            "challenger_min_dp_ratio": chal_dp,
            "champion_min_dp_ratio": champ_dp,
            "floor": settings.dp_ratio_min,
        },
        reasons, checks,
    )

    # 6. Fairness EO regression
    chal_eo = min(challenger_fair.eo_ratio_by_attr.values()) if challenger_fair.eo_ratio_by_attr else 0.0
    champ_eo = min(champion_fair.eo_ratio_by_attr.values()) if champion_fair.eo_ratio_by_attr else 0.0
    _check(
        "fairness_eo_floor",
        chal_eo >= settings.eo_ratio_min,
        {
            "challenger_min_eo_ratio": chal_eo,
            "champion_min_eo_ratio": champ_eo,
            "floor": settings.eo_ratio_min,
        },
        reasons, checks,
    )

    # 7. Training-data drift sanity
    _check(
        "train_drift_psi",
        train_drift_psi < settings.drift_psi_max,
        {"psi": train_drift_psi, "max": settings.drift_psi_max},
        reasons, checks,
    )

    # 8. Strict improvement, unless champion is stale
    champion_age_days = None
    strict_required = True
    if champion_trained_at is not None:
        champion_age_days = (now - champion_trained_at).days
        if champion_age_days >= settings.champion_max_age_days:
            strict_required = False
    strict_pass = (challenger.roc_auc > champion.roc_auc) if strict_required else True
    _check(
        "strict_improvement_or_stale",
        strict_pass,
        {
            "challenger_auc": challenger.roc_auc,
            "champion_auc": champion.roc_auc,
            "champion_age_days": champion_age_days,
            "strict_required": strict_required,
            "stale_threshold_days": settings.champion_max_age_days,
        },
        reasons, checks,
    )

    return PromotionDecision(promote=len(reasons) == 0, reasons=reasons, checks=checks)
