# ADR 001 — Use MLflow Aliases, not Stages

- Status: Accepted

## Context

MLflow's classic model lifecycle uses **Stages** (None, Staging,
Production, Archived). A model version sits in exactly one stage at a
time, and you transition between them with
`transition_model_version_stage`. Tutorials and most blog posts still
show this API.

As of **MLflow 2.9.0**, Stages are deprecated and will be removed.
The replacement is **Aliases**: named pointers attached to a version.
A single version can hold multiple aliases (e.g., `@champion` and
`@canary`); a single alias can be re-pointed between versions.

## Decision

Use **Aliases** throughout. The two aliases we care about are
`@champion` (the version `serve.py` loads) and `@challenger` (whatever
the most recent retrain produced before it was evaluated).

## Consequences

- `models:/credit_risk@champion` is the canonical URI in `serve.py`.
- Promotion = `client.set_registered_model_alias("credit_risk", "champion", v)`.
  No transition needed.
- Rollback = same operation pointed at an older version. No "un-archive"
  step. Cleaner than Stages.
- Multiple aliases enables canary deployments later without API change.

## Rejected alternative

Sticking with Stages because tutorials use them. Deprecated APIs are
debt; better to write against the supported one from day 1.
