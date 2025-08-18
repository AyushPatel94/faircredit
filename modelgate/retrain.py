"""Full retrain orchestrator: train challenger, evaluate vs champion, gate, promote-or-skip, audit.

CLI: python -m modelgate.retrain --week N
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import mlflow
import typer

from modelgate import audit
from modelgate.config import settings
from modelgate.data import load_raw, prepare_holdout, split_into_weeks, validate_schema
from modelgate.drift import max_psi
from modelgate.evaluate import evaluate_on_holdout
from modelgate.gate import decide
from modelgate.registry import CHALLENGER, CHAMPION, Registry
from modelgate.train import train_week
from modelgate.utils import ensure_dirs, get_logger

logger = get_logger(__name__)
app = typer.Typer(add_completion=False, help="Retrain + evaluate vs champion + decide")


def _load_pipeline_from_version(version: int):
    reg = Registry()
    info = reg.get_version(version)
    if info is None:
        raise ValueError(f"No version {version} for {reg.model_name}")
    # use the run_id directly. the cleaner `models:/<name>@<alias>` URI
    # blows up on windows because mlflow's artifact resolver tries to
    # treat the windows drive letter as a URI scheme. burned an
    # afternoon on this. see NOTES.md.
    uri = f"runs:/{info.run_id}/model"
    return mlflow.sklearn.load_model(uri), info


def run_retrain(week: int, raw_path: Path | str | None = None) -> dict:
    ensure_dirs()
    reg = Registry()

    # If there's no champion yet, train the baseline and promote it directly.
    champion = reg.get_alias_version(CHAMPION)
    if champion is None:
        logger.info("No @champion -- training baseline and promoting it directly.")
        baseline = train_week(week=week, set_alias=True, raw_path=raw_path)
        reg.promote_to_champion(baseline["version"])
        record = {
            "action": "bootstrap_champion",
            "week": week,
            "version": baseline["version"],
            "val_metrics": baseline["val_metrics"],
        }
        audit.append(record)
        return record

    # Otherwise: train challenger on this week's window, evaluate on the *next* week as holdout.
    challenger = train_week(week=week, set_alias=True, raw_path=raw_path)

    df = load_raw(raw_path)
    validate_schema(df)
    weeks = split_into_weeks(df)
    holdout_week = min(week + 1, len(weeks) - 1)
    holdout = prepare_holdout(weeks, holdout_week)

    challenger_pipe, challenger_info = _load_pipeline_from_version(challenger["version"])
    champion_pipe, champion_info = _load_pipeline_from_version(champion.version)

    chal_core, chal_fair = evaluate_on_holdout(challenger_pipe, holdout)
    champ_core, champ_fair = evaluate_on_holdout(champion_pipe, holdout)

    # Training-data drift: compare numeric mean/dist of this week's training window vs champion training week
    champ_week = int(champion_info.tags.get("week", week - 1))
    chal_window = weeks[max(0, week - settings.rolling_window_weeks + 1) : week + 1]
    champ_window = weeks[max(0, champ_week - settings.rolling_window_weeks + 1) : champ_week + 1]
    chal_concat = _concat_numeric(chal_window)
    champ_concat = _concat_numeric(champ_window)
    psi = max_psi(champ_concat, chal_concat, features=list(chal_concat.columns)[:8])

    decision = decide(
        challenger=chal_core,
        challenger_fair=chal_fair,
        champion=champ_core,
        champion_fair=champ_fair,
        train_drift_psi=psi,
        champion_trained_at=champion.created_at,
    )

    record: dict[str, Any] = {
        "action": "promote" if decision.promote else "skip",
        "week": week,
        "holdout_week": holdout_week,
        "challenger_version": challenger["version"],
        "champion_version_before": champion.version,
        "challenger_metrics": chal_core.as_dict(),
        "champion_metrics": champ_core.as_dict(),
        "challenger_fairness": chal_fair.as_dict(),
        "champion_fairness": champ_fair.as_dict(),
        "train_drift_psi": psi,
        "decision": decision.as_dict(),
    }

    if decision.promote:
        reg.promote_to_champion(challenger["version"])
        reg.delete_alias(CHALLENGER)
        logger.info(f"PROMOTED v{challenger['version']} -> @champion")
    else:
        reg.delete_alias(CHALLENGER)
        logger.info(f"SKIPPED v{challenger['version']}: {decision.reasons}")

    audit.append(record)
    return record


def _concat_numeric(weeks_list):
    import pandas as pd

    df = pd.concat(weeks_list, ignore_index=True)
    return df.select_dtypes(include=["number"])


@app.command()
def main(week: int = typer.Option(..., help="Week index to retrain on")) -> None:
    out = run_retrain(week=week)
    print(f"\nDecision: {out['action']}")
    if "decision" in out:
        for r in out["decision"]["reasons"]:
            print(f"  - {r}")


if __name__ == "__main__":
    app()
