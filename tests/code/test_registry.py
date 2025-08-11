"""Integration-ish tests for the MLflow Registry wrapper.

These use a temporary sqlite tracking URI so they don't pollute the
project's mlruns. Tests run the actual MLflow client, just against a
throwaway store.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import mlflow
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from modelgate.registry import CHALLENGER, CHAMPION, Registry


@pytest.fixture
def tmp_mlflow(tmp_path, monkeypatch):
    db = tmp_path / "mlflow.db"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    monkeypatch.setattr("modelgate.config.settings.mlflow_tracking_uri", f"sqlite:///{db.as_posix()}")
    monkeypatch.setattr("modelgate.config.settings.mlflow_artifact_root", f"file:{artifacts.as_posix()}")
    mlflow.set_tracking_uri(f"sqlite:///{db.as_posix()}")
    yield tmp_path


def _log_dummy_model() -> tuple[str, str]:
    """Train a tiny dummy classifier, log it, return (run_id, model_uri)."""
    pipe = Pipeline([("clf", DummyClassifier(strategy="constant", constant=0))])
    pipe.fit([[0], [1]], [0, 0])
    mlflow.set_experiment("test")
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(pipe, name="model")
        return run.info.run_id, f"runs:/{run.info.run_id}/model"


def test_create_and_get_model(tmp_mlflow):
    reg = Registry(model_name="test_model")
    # accessing twice should be idempotent
    reg2 = Registry(model_name="test_model")
    assert reg.model_name == reg2.model_name


def test_register_and_alias(tmp_mlflow):
    reg = Registry(model_name="test_model")
    run_id, uri = _log_dummy_model()
    v1 = reg.register_run(run_id=run_id, source=uri, tags={"flavor": "dummy"})
    assert v1.version == 1
    assert v1.tags["flavor"] == "dummy"

    reg.stage_as_challenger(v1.version)
    challenger = reg.get_alias_version(CHALLENGER)
    assert challenger is not None
    assert challenger.version == v1.version

    # no champion yet
    assert reg.get_alias_version(CHAMPION) is None

    reg.promote_to_champion(v1.version)
    champ = reg.get_alias_version(CHAMPION)
    assert champ is not None and champ.version == v1.version
    # promotion drops the challenger alias from the same version
    assert reg.get_alias_version(CHALLENGER) is None


def test_rollback_dry_run_makes_no_changes(tmp_mlflow):
    reg = Registry(model_name="test_model")
    run_id_1, uri_1 = _log_dummy_model()
    run_id_2, uri_2 = _log_dummy_model()
    v1 = reg.register_run(run_id=run_id_1, source=uri_1)
    v2 = reg.register_run(run_id=run_id_2, source=uri_2)
    reg.promote_to_champion(v2.version)
    assert reg.get_alias_version(CHAMPION).version == v2.version

    plan = reg.rollback_to(v1.version, dry_run=True)
    assert plan["dry_run"] is True
    assert plan["to_version"] == v1.version
    assert plan["from_version"] == v2.version
    # still pointing at v2
    assert reg.get_alias_version(CHAMPION).version == v2.version


def test_rollback_real_repoints_alias(tmp_mlflow):
    reg = Registry(model_name="test_model")
    run_id_1, uri_1 = _log_dummy_model()
    run_id_2, uri_2 = _log_dummy_model()
    v1 = reg.register_run(run_id=run_id_1, source=uri_1)
    v2 = reg.register_run(run_id=run_id_2, source=uri_2)
    reg.promote_to_champion(v2.version)

    reg.rollback_to(v1.version, dry_run=False)
    assert reg.get_alias_version(CHAMPION).version == v1.version


def test_rollback_unknown_version_raises(tmp_mlflow):
    reg = Registry(model_name="test_model")
    with pytest.raises(ValueError):
        reg.rollback_to(999, dry_run=True)


def test_delete_alias_is_idempotent(tmp_mlflow):
    reg = Registry(model_name="test_model")
    reg.delete_alias("nonexistent")  # should not raise
