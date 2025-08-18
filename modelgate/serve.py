"""FastAPI service that loads the current @champion from the MLflow Registry.

GET  /health     liveness + whether a champion is loaded
GET  /champion   metadata of the current champion
GET  /metrics    Prometheus exposition
POST /predict    single applicant -> probability + risk band
POST /reload     re-fetch the @champion alias and hot-swap the in-memory model
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field
from starlette.responses import Response

from modelgate.config import settings
from modelgate.data import add_age_bucket, basic_clean
from modelgate.features import engineer
from modelgate.registry import CHAMPION, Registry
from modelgate.utils import get_logger

logger = get_logger(__name__)


REQUEST_COUNTER = Counter(
    "modelgate_requests_total", "Total prediction requests", ["endpoint", "status"]
)
LATENCY_HIST = Histogram(
    "modelgate_request_latency_seconds",
    "Prediction latency in seconds",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
RISK_GAUGE = Gauge("modelgate_last_risk_proba", "Risk probability of last scored request")
CHAMPION_VERSION_GAUGE = Gauge("modelgate_champion_version", "Currently loaded champion version")


class Application(BaseModel):
    SK_ID_CURR: int = Field(..., description="Application id")
    payload: dict[str, Any] = Field(..., description="Applicant feature dict (sparse OK)")


class Prediction(BaseModel):
    sk_id_curr: int
    risk_proba: float
    risk_band: str
    champion_version: int


class ChampionInfo(BaseModel):
    model_name: str
    version: int | None
    aliases: list[str]
    tags: dict[str, str]
    created_at: str | None


def _risk_band(p: float) -> str:
    if p >= 0.5:
        return "high"
    if p >= 0.2:
        return "medium"
    return "low"


class ModelHolder:
    pipe: Any = None
    version: int | None = None
    expected_cols: list[str] = []
    created_at: str | None = None
    tags: dict[str, str] = {}

    @classmethod
    def load_champion(cls) -> ChampionInfo:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        reg = Registry()
        info = reg.get_alias_version(CHAMPION)
        if info is None:
            cls.pipe = None
            cls.version = None
            cls.expected_cols = []
            cls.created_at = None
            cls.tags = {}
            return ChampionInfo(model_name=reg.model_name, version=None, aliases=[], tags={}, created_at=None)

        uri = f"runs:/{info.run_id}/model"
        cls.pipe = mlflow.sklearn.load_model(uri)
        cls.version = info.version
        cls.created_at = info.created_at.isoformat()
        cls.tags = info.tags
        pre = cls.pipe.named_steps.get("preprocess") if hasattr(cls.pipe, "named_steps") else None
        cls.expected_cols = list(getattr(pre, "feature_names_in_", [])) if pre is not None else []
        CHAMPION_VERSION_GAUGE.set(info.version)
        logger.info(f"Loaded @champion v{info.version} ({len(cls.expected_cols)} features)")
        return ChampionInfo(
            model_name=reg.model_name,
            version=info.version,
            aliases=info.aliases,
            tags=info.tags,
            created_at=cls.created_at,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # TODO: warm up with a sample prediction here so the first user-facing
    # /predict doesn't pay the lazy-deserialise cost (~5s cold).
    try:
        ModelHolder.load_champion()
    except Exception as e:
        logger.warning(f"Champion not loaded at startup: {e}")
    yield


app = FastAPI(
    title="ModelGate API",
    description="Serves the current @champion model with hot-swap support.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if ModelHolder.pipe is not None else "no_champion",
        "champion_version": ModelHolder.version,
    }


@app.get("/champion", response_model=ChampionInfo)
def champion_info() -> ChampionInfo:
    reg = Registry()
    info = reg.get_alias_version(CHAMPION)
    if info is None:
        return ChampionInfo(model_name=reg.model_name, version=None, aliases=[], tags={}, created_at=None)
    return ChampionInfo(
        model_name=reg.model_name,
        version=info.version,
        aliases=info.aliases,
        tags=info.tags,
        created_at=info.created_at.isoformat(),
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=Prediction)
def predict(app_in: Application) -> Prediction:
    if ModelHolder.pipe is None:
        REQUEST_COUNTER.labels(endpoint="predict", status="error").inc()
        raise HTTPException(status_code=503, detail="champion not loaded")

    t0 = time.perf_counter()
    try:
        df = pd.DataFrame([{settings.id_col: app_in.SK_ID_CURR, **app_in.payload}])
        df = add_age_bucket(df)
        df = basic_clean(df)
        eng = engineer(df)
        drop_cols = [c for c in {settings.id_col, "AGE_BUCKET", settings.target_col} if c in eng.columns]
        x = eng.drop(columns=drop_cols)
        if ModelHolder.expected_cols:
            for col in ModelHolder.expected_cols:
                if col not in x.columns:
                    x[col] = np.nan
            x = x[ModelHolder.expected_cols].infer_objects(copy=False)
        proba = float(ModelHolder.pipe.predict_proba(x)[0, 1])

        LATENCY_HIST.labels(endpoint="predict").observe(time.perf_counter() - t0)
        REQUEST_COUNTER.labels(endpoint="predict", status="ok").inc()
        RISK_GAUGE.set(proba)
        return Prediction(
            sk_id_curr=app_in.SK_ID_CURR,
            risk_proba=proba,
            risk_band=_risk_band(proba),
            champion_version=ModelHolder.version or -1,
        )
    except Exception as e:
        REQUEST_COUNTER.labels(endpoint="predict", status="error").inc()
        logger.exception("predict failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/reload")
def reload_champion() -> ChampionInfo:
    return ModelHolder.load_champion()
