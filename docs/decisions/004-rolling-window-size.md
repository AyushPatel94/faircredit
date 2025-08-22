# ADR 004 — Rolling-window size of 4 weeks

- Status: Accepted

## Context

The retrain orchestrator uses a rolling window of past data to train
each challenger. Window size has to balance two failure modes:

- **Too short**: model overfits to short-term noise; every week's
  retrain looks dramatically different from the previous.
- **Too long**: drift takes too long to flow into the model; the
  retrain stops being meaningful.

## Decision

**4 weeks** rolling. Configurable via `settings.rolling_window_weeks`.

## Why 4 weeks

- The dataset is sliced into 8 weekly bins for the simulation. 4 is
  half the total -- enough to smooth single-week anomalies, but small
  enough that drift in weeks 4-7 can dominate the window by week 6.
- 4 weeks of ~38k rows per slice ≈ 150k training rows, which is plenty
  for XGBoost on this feature space.
- For a real production system the number would be tuned against
  business cycles. 4 here is illustrative.

## Consequences

- The first 3 weeks of the simulation are necessarily light on data;
  the model trained at week 0 only has the week-0 slice (since week
  -3 to -1 don't exist). The orchestrator clips the start to 0.
- Drift injection at week 4 enters the training window the next
  retrain cycle, then dominates by week 6.
