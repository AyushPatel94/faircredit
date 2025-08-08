"""Append-only JSONL audit log of promotion decisions.

One decision per line. Never overwritten, never reordered. The whole
point is that you can `tail logs/decisions.jsonl` and see the full
history of what happened to your production model.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from modelgate.config import settings
from modelgate.utils import get_logger

logger = get_logger(__name__)

DEFAULT_LOG = settings.logs_dir / "decisions.jsonl"


def append(record: dict, path: Path | str | None = None) -> Path:
    path = Path(path) if path else DEFAULT_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = {"recorded_at": datetime.now(tz=timezone.utc).isoformat(), **record}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(enriched, default=str) + "\n")
    logger.info(f"Audit entry appended: {record.get('action')} @ {path}")
    return path


def read_all(path: Path | str | None = None) -> list[dict]:
    path = Path(path) if path else DEFAULT_LOG
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def latest(n: int = 5, path: Path | str | None = None) -> list[dict]:
    return read_all(path)[-n:]
