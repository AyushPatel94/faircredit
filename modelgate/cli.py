"""Top-level Typer CLI: groups the per-command apps."""
from __future__ import annotations

import typer

from modelgate import audit
from modelgate.registry import CHAMPION, Registry
from modelgate.retrain import app as retrain_app
from modelgate.rollback import app as rollback_app
from modelgate.train import app as train_app

app = typer.Typer(add_completion=False, help="ModelGate CLI")
app.add_typer(train_app, name="train")
app.add_typer(retrain_app, name="retrain")
app.add_typer(rollback_app, name="rollback")


@app.command()
def status() -> None:
    """Print the current @champion and the last few promotion decisions."""
    reg = Registry()
    champion = reg.get_alias_version(CHAMPION)
    print("\n== ModelGate status ==")
    if champion is None:
        print(f"  model: {reg.model_name}")
        print("  @champion: (none)")
    else:
        print(f"  model: {reg.model_name}")
        print(f"  @champion: v{champion.version}  run={champion.run_id[:8]}  created={champion.created_at}")
        if champion.tags:
            print(f"  tags: {champion.tags}")

    print("\n== Recent decisions ==")
    recent = audit.latest(5)
    if not recent:
        print("  (no entries)")
    for r in recent:
        action = r.get("action", "?")
        wk = r.get("week", "?")
        chal = r.get("challenger_version", "?")
        print(f"  [{r.get('recorded_at', '')[:19]}] week={wk} action={action} challenger=v{chal}")


@app.command()
def versions() -> None:
    """List all registered versions of the model."""
    reg = Registry()
    alias_map: dict[int, list[str]] = {}
    for alias in ("champion", "challenger"):
        v = reg.get_alias_version(alias)
        if v is not None:
            alias_map.setdefault(v.version, []).append(alias)
    for info in reg.list_versions():
        aliases = alias_map.get(info.version, [])
        alias_str = ",".join(aliases) if aliases else "-"
        print(f"v{info.version:<3}  aliases=[{alias_str}]  run={info.run_id[:8]}  created={info.created_at}")


if __name__ == "__main__":
    app()
