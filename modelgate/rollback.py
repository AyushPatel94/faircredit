"""Roll back the @champion alias to an earlier version.

Usage:
    modelgate rollback --to-version 5
    modelgate rollback --to-version 5 --dry-run
"""
from __future__ import annotations

import typer

from modelgate import audit
from modelgate.registry import CHAMPION, Registry
from modelgate.utils import get_logger

logger = get_logger(__name__)
app = typer.Typer(add_completion=False, help="Roll back the @champion alias")


@app.command()
def main(
    to_version: int = typer.Option(..., "--to-version", "-v", help="Version to roll back to"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the plan, make no changes"),
    reason: str = typer.Option("manual rollback", help="Reason for the rollback (audited)"),
) -> None:
    reg = Registry()
    plan = reg.rollback_to(to_version, dry_run=dry_run)
    print("\nrollback plan:")
    for k, v in plan.items():
        print(f"  {k}: {v}")
    if not dry_run:
        audit.append({"action": "rollback", "reason": reason, **plan})


if __name__ == "__main__":
    app()
