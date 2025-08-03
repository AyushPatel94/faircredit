"""Train a single XGBoost pipeline on one rolling window, log everything
to MLflow, register a new version, and tag it as @challenger.

Promotion to @champion is handled later by `gate.py` + `retrain.py`.
"""
from __future__ import annotations

import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import typer

from modelgate.config import settings
from modelgate.data import load_raw, prepare_window_for_training, split_into_weeks, validate_schema
from modelgate.evaluate import core_metrics, fairness_metrics
from modelgate.features import engineer, split_x_y
from modelgate.models import build_xgb_pipeline
from modelgate.registry import CHALLENGER, Registry
from modelgate.utils import ensure_dirs, get_logger

logger = get_logger(__name__)
app = typer.Typer(add_completion=False, help="Train a challenger for a given week")


def _setup_mlflow() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    name = "modelgate"
    existing = client.get_experiment_by_name(name)
    if existing is None:
        client.create_experiment(name=name, artifact_location=settings.mlflow_artifact_root)
    mlflow.set_experiment(name)


def train_week(week: int, set_alias: bool = True, raw_path: Path | str | None = None) -> dict:
    ensure_dirs()
    _setup_mlflow()

    df = load_raw(raw_path)
    validate_schema(df)
    weeks = split_into_weeks(df)
    train_df, val_df = prepare_window_for_training(weeks, week)
    train_df_eng = engineer(train_df)
    val_df_eng = engineer(val_df)
    x_train, y_train = split_x_y(train_df_eng)
    x_val, y_val = split_x_y(val_df_eng)

    pipe = build_xgb_pipeline(x_train)
    with mlflow.start_run(run_name=f"train-week-{week}-{int(time.time())}") as run:
        mlflow.log_params({"week": week, "rolling_weeks": settings.rolling_window_weeks,
                           "train_rows": len(x_train), "val_rows": len(x_val)})

        t0 = time.time()
        pipe.fit(x_train, y_train)
        fit_seconds = time.time() - t0
        mlflow.log_metric("fit_seconds", fit_seconds)

        val_proba = pipe.predict_proba(x_val)[:, 1]
        val = core_metrics(y_val, val_proba)
        val_fair = fairness_metrics(val_df_eng, y_val, val_proba)
        for k, v in val.as_dict().items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"val_{k}", v)
        for attr, ratio in val_fair.dp_ratio_by_attr.items():
            mlflow.log_metric(f"val_dp_ratio__{attr}", ratio)
        for attr, ratio in val_fair.eo_ratio_by_attr.items():
            mlflow.log_metric(f"val_eo_ratio__{attr}", ratio)

        mlflow.sklearn.log_model(pipe, name="model")
        model_uri = f"runs:/{run.info.run_id}/model"

        reg = Registry()
        version_info = reg.register_run(
            run_id=run.info.run_id,
            source=model_uri,
            tags={"week": str(week), "fit_seconds": f"{fit_seconds:.1f}"},
        )
        if set_alias:
            reg.stage_as_challenger(version_info.version)

        logger.info(
            f"Week {week} trained -> v{version_info.version}  "
            f"val_auc={val.roc_auc:.4f}  fit={fit_seconds:.1f}s"
        )
        return {
            "run_id": run.info.run_id,
            "version": version_info.version,
            "val_metrics": val.as_dict(),
            "val_fairness": val_fair.as_dict(),
            "fit_seconds": fit_seconds,
        }


@app.command()
def main(
    week: int = typer.Option(0, help="Week index (0-based)"),
    no_alias: bool = typer.Option(False, help="Don't apply the @challenger alias"),
) -> None:
    result = train_week(week=week, set_alias=not no_alias)
    print(f"\nversion v{result['version']}  val_auc {result['val_metrics']['roc_auc']:.4f}")


if __name__ == "__main__":
    app()
