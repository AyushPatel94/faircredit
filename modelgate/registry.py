"""Thin wrapper over MLflow Model Registry, using the modern Aliases API
(Stages are deprecated as of MLflow 2.9).

The interesting operations:
- promote(version) sets alias "champion" -> version, and demotes whatever
  was champion to nothing (the previous version still lives, can be
  re-pointed at via rollback).
- get_alias(alias) returns the current ModelVersion under that alias.
- archive(version) doesn't delete -- just removes any aliases pointing at
  it. The version stays accessible by number for rollback.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

from modelgate.config import settings
from modelgate.utils import get_logger

logger = get_logger(__name__)

CHAMPION = "champion"
CHALLENGER = "challenger"


@dataclass
class VersionInfo:
    version: int
    run_id: str
    aliases: list[str]
    tags: dict[str, str]
    created_at: datetime

    @property
    def is_champion(self) -> bool:
        return CHAMPION in self.aliases

    @property
    def is_challenger(self) -> bool:
        return CHALLENGER in self.aliases


class Registry:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.registered_model_name
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        self.client = MlflowClient()
        self._ensure_model_exists()

    def _ensure_model_exists(self) -> None:
        # NOTE: with the sqlalchemy backend the "not found" error is a
        # plain MlflowException, not a RestException. caught both by
        # matching on error_code and falling back to a string check.
        try:
            self.client.get_registered_model(self.model_name)
        except MlflowException as e:
            err_code = getattr(e, "error_code", "")
            if err_code == "RESOURCE_DOES_NOT_EXIST" or "not found" in str(e).lower():
                self.client.create_registered_model(self.model_name)
                logger.info(f"Created registered model: {self.model_name}")
            else:
                raise

    # ---- aliases --------------------------------------------------------

    def set_alias(self, alias: str, version: int) -> None:
        self.client.set_registered_model_alias(self.model_name, alias, str(version))
        logger.info(f"Aliased {self.model_name}@{alias} -> v{version}")

    def delete_alias(self, alias: str) -> None:
        try:
            self.client.delete_registered_model_alias(self.model_name, alias)
            logger.info(f"Cleared alias @{alias} on {self.model_name}")
        except MlflowException:
            pass

    def get_alias_version(self, alias: str) -> VersionInfo | None:
        try:
            mv = self.client.get_model_version_by_alias(self.model_name, alias)
        except MlflowException as e:
            err_code = getattr(e, "error_code", "")
            if err_code == "RESOURCE_DOES_NOT_EXIST" or "alias" in str(e).lower():
                return None
            raise
        return self._version_info(mv)

    # ---- versions -------------------------------------------------------

    def list_versions(self) -> list[VersionInfo]:
        infos = []
        for mv in self.client.search_model_versions(f"name='{self.model_name}'"):
            infos.append(self._version_info(mv))
        return sorted(infos, key=lambda v: v.version)

    def get_version(self, version: int) -> VersionInfo | None:
        try:
            mv = self.client.get_model_version(self.model_name, str(version))
        except MlflowException:
            return None
        return self._version_info(mv)

    def _version_info(self, mv) -> VersionInfo:
        aliases = list(getattr(mv, "aliases", []) or [])
        return VersionInfo(
            version=int(mv.version),
            run_id=mv.run_id,
            aliases=aliases,
            tags=dict(mv.tags) if mv.tags else {},
            created_at=datetime.fromtimestamp(mv.creation_timestamp / 1000, tz=timezone.utc),
        )

    # ---- high-level operations ------------------------------------------

    def register_run(self, run_id: str, source: str, tags: dict | None = None) -> VersionInfo:
        mv = self.client.create_model_version(
            name=self.model_name,
            source=source,
            run_id=run_id,
            tags=tags or {},
        )
        info = self.get_version(int(mv.version))
        assert info is not None
        return info

    def promote_to_champion(self, version: int) -> None:
        """Set @champion on `version` and remove any @challenger alias from it."""
        self.set_alias(CHAMPION, version)
        # if this version was the challenger, drop the challenger alias on it
        current_challenger = self.get_alias_version(CHALLENGER)
        if current_challenger is not None and current_challenger.version == version:
            self.delete_alias(CHALLENGER)

    def stage_as_challenger(self, version: int) -> None:
        self.set_alias(CHALLENGER, version)

    def rollback_to(self, version: int, dry_run: bool = False) -> dict:
        """Re-point @champion to a prior version."""
        target = self.get_version(version)
        if target is None:
            raise ValueError(f"Version {version} not found in {self.model_name}")
        current = self.get_alias_version(CHAMPION)
        plan = {
            "model": self.model_name,
            "dry_run": dry_run,
            "from_version": current.version if current else None,
            "to_version": version,
            "target_tags": target.tags,
        }
        if dry_run:
            logger.info(f"DRY-RUN rollback plan: {plan}")
            return plan
        self.set_alias(CHAMPION, version)
        logger.info(f"Rolled back @champion from v{plan['from_version']} -> v{version}")
        return plan
