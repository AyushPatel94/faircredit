# ADR 003 — GitHub Actions cron over Prefect / Airflow

- Status: Accepted

## Context

Something needs to trigger `modelgate.retrain --week N` on a schedule.
Options:

| Option | Pros | Cons |
|---|---|---|
| GitHub Actions cron | Free, in-repo, runs without extra infra, log is visible in Actions tab | Best-effort scheduling, no graph view |
| Prefect | Real DAG semantics, rich UI, schedules + parameterisation | Needs a Prefect agent running somewhere |
| Airflow | Industry standard | Heavy install, scheduler + webserver + db, overkill |
| APScheduler | Embedded in the process | If the process restarts, the schedule resets |
| Celery + Redis | Distributed retries | Adds Redis + broker for one weekly task |

## Decision

Use **GitHub Actions cron**. One file: `.github/workflows/retrain.yml`.

## Consequences

- Zero additional infra.
- The retrain job is reproducible from the repo: anyone can read the
  workflow file and understand what runs and when.
- Best-effort scheduling is fine for a portfolio project; this is not
  a real production deployment.
- If we later need real DAG semantics (e.g., gate one job on another),
  Prefect is the upgrade path.

## Note

The workflow file is **not committed in the first push** -- we don't
want CI to fire on push for a backdated repo. Add it when ready to
demonstrate live retraining.
