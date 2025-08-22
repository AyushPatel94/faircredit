# ADR 002 — Fairness as a hard promotion gate

- Status: Accepted

## Context

The retraining loop has to decide whether a challenger should replace
the champion. The obvious metric is AUC. Most production systems also
check PR-AUC and a calibration metric like Brier. Few systems gate on
**fairness** as a hard promotion requirement.

In a credit-risk context this matters: a model can improve overall AUC
while widening the gap between protected groups (or by leaning more on
a sensitive feature). If we promote on AUC alone, we silently introduce
adverse-impact risk.

## Decision

The gate has two fairness checks that **must** pass alongside the
accuracy checks:

- `min(DP ratio across protected attrs) ≥ 0.05` at threshold 0.15
- `min(EO ratio across protected attrs) ≥ 0.05` at threshold 0.15

Protected attributes default to `CODE_GENDER` and `AGE_BUCKET`.

## Why 0.05 and not 0.80?

The four-fifths rule (0.80 floor) is the US EEOC standard. Our
underlying Home Credit dataset is deeply imbalanced; observed DP/EO
ratios sit around 0.10-0.12 for both champion and challenger across the
8-week simulation. Setting the gate at 0.80 would cause the retrain
loop to **never promote anything**, which defeats the point.

The 0.05 floor is a **regression guard**: "don't make fairness
catastrophically worse." A challenger whose minimum DP/EO ratio drops
to 0.05 or below is failing materially worse than the population norm
for this dataset. This is the operating choice for the simulation, not
a claim of fairness compliance. Tightening the floor is a config change
in `config.py` (`dp_ratio_min`, `eo_ratio_min`).

## Consequences

- Every promotion decision in `logs/decisions.jsonl` carries DP/EO
  ratios so the operator can see fairness drift over time.
- A challenger with strictly better AUC can still be rejected if it
  regresses on fairness. This is the intended behaviour and the
  project's main differentiator.
- The 8-week simulation deliberately includes a week where this
  happens (drift week + retrain that improves AUC but lowers DP ratio).

## Rejected alternative

Soft warning instead of hard block. We tried it conceptually; it's
worse because nobody reads the warnings.
