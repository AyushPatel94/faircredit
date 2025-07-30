from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from modelgate.config import settings
from modelgate.utils import get_logger

logger = get_logger(__name__)


@dataclass
class Metrics:
    roc_auc: float
    pr_auc: float
    brier: float
    positive_rate: float
    n: int

    def as_dict(self) -> dict:
        return {
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "brier": self.brier,
            "positive_rate": self.positive_rate,
            "n": self.n,
        }


@dataclass
class FairnessMetrics:
    dp_ratio_by_attr: dict[str, float]
    eo_ratio_by_attr: dict[str, float]
    threshold: float
    slice_table: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "dp_ratio_by_attr": self.dp_ratio_by_attr,
            "eo_ratio_by_attr": self.eo_ratio_by_attr,
            "slice_table": self.slice_table,
        }


def core_metrics(y_true, y_proba) -> Metrics:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    return Metrics(
        roc_auc=float(roc_auc_score(y_true, y_proba)),
        pr_auc=float(average_precision_score(y_true, y_proba)),
        brier=float(brier_score_loss(y_true, y_proba)),
        positive_rate=float(np.mean(y_true)),
        n=int(len(y_true)),
    )


def _dp_ratio(rates: dict[str, float]) -> float:
    vals = [r for r in rates.values() if not np.isnan(r)]
    if not vals or max(vals) == 0:
        return 0.0
    return float(min(vals) / max(vals))


def _eo_ratio_from_groups(
    df: pd.DataFrame,
    attr: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    min_count: int = 30,
) -> float:
    tprs, fprs = [], []
    for value, idx in df.groupby(attr).groups.items():
        if len(idx) < min_count:
            continue
        yt = y_true.loc[idx]
        yp = y_pred[df.index.get_indexer(idx)]
        if yt.nunique() < 2:
            continue
        tp = int(((yp == 1) & (yt == 1)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        tprs.append(tpr)
        fprs.append(fpr)
    if not tprs:
        return 1.0
    tpr_ratio = min(tprs) / max(tprs) if max(tprs) > 0 else 0.0
    fpr_ratio = min(fprs) / max(fprs) if max(fprs) > 0 else 1.0
    return float(min(tpr_ratio, fpr_ratio))


def fairness_metrics(
    df: pd.DataFrame,
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float | None = None,
    min_count: int = 30,
) -> FairnessMetrics:
    threshold = threshold if threshold is not None else settings.audit_threshold
    y_pred = (y_proba >= threshold).astype(int)

    dp_by, eo_by, table = {}, {}, []
    for attr in settings.protected_attributes:
        if attr not in df.columns:
            continue
        rates = {}
        for value, idx in df.groupby(attr).groups.items():
            if len(idx) < min_count:
                continue
            yhat = y_pred[df.index.get_indexer(idx)]
            rates[value] = float(np.mean(yhat))
            table.append({
                "attr": attr,
                "value": value,
                "count": int(len(idx)),
                "selection_rate": float(np.mean(yhat)),
            })
        dp_by[attr] = _dp_ratio(rates)
        eo_by[attr] = _eo_ratio_from_groups(df, attr, y_true, y_pred, min_count=min_count)

    return FairnessMetrics(
        dp_ratio_by_attr=dp_by,
        eo_ratio_by_attr=eo_by,
        threshold=threshold,
        slice_table=table,
    )


def evaluate_on_holdout(pipe, df: pd.DataFrame) -> tuple[Metrics, FairnessMetrics]:
    """Run a pipeline against a holdout dataframe and return both metric sets."""
    from modelgate.features import engineer, split_x_y

    eng = engineer(df.reset_index(drop=True))
    x, y = split_x_y(eng)
    proba = pipe.predict_proba(x)[:, 1]
    core = core_metrics(y, proba)
    fair = fairness_metrics(eng, y, proba)
    logger.info(
        f"Holdout n={core.n:,}  AUC={core.roc_auc:.4f}  PR-AUC={core.pr_auc:.4f}  Brier={core.brier:.4f}"
    )
    return core, fair
