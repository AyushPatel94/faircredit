from __future__ import annotations

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from modelgate.config import XGB_DEFAULT_PARAMS
from modelgate.features import build_preprocessor


def build_xgb_pipeline(x_sample, params: dict | None = None) -> Pipeline:
    params = params if params is not None else XGB_DEFAULT_PARAMS
    pre = build_preprocessor(x_sample)
    clf = XGBClassifier(**params)
    return Pipeline([("preprocess", pre), ("clf", clf)])
