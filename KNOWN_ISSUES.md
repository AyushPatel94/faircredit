# Known Issues & Limitations

Things that don't work cleanly, things that work but are sketchy, and
the surface where this project would need real engineering to ship to
production.

## Registry / MLflow

- **MLflow's `models:/<name>@alias` URI fails on Windows.** The
  artifact resolver hands a Windows path like `c:\Users\...\tmp_xxx`
  to the URI-scheme dispatcher and bombs. Workaround: load via
  `runs:/<run_id>/model` instead. Done in `retrain.py` and `serve.py`.
  Likely fixed in MLflow 3.x; not yet validated.

- **MLflow Stages API is deprecated.** Tutorials still use it. We use
  Aliases (the supported replacement) -- see ADR 001.

- **Filesystem MLflow tracking backend is deprecated.** We use a SQLite
  backend (`sqlite:///mlruns.db`). Default in `config.py`.

## The promotion gate

- **0.05 fairness floor is not a compliance claim.** Home Credit's
  observed DP/EO ratios sit around 0.10. The 0.05 floor is a
  *regression guard*. See ADR 002.

- **Drift PSI threshold is 2.0**, much higher than the textbook 0.25.
  Reason: early weeks have rolling windows of different sizes (the
  week-0 train has 1 week of data, the week-3 train has 4 weeks). PSI
  between unequal samples is naturally large. Real production with a
  stable backfill could tighten this back to 0.25.

- **The 8th check ("strict improvement, unless champion stale") biases
  toward stability.** A challenger that ties on AUC is rejected unless
  the champion is older than 14 days. This intentionally over-favours
  the incumbent because hot-swapping production models is risky.

## Simulation

- **Drift is simulated, not observed.** The 8-week sim multiplies
  `AMT_INCOME_TOTAL` by 1.10 from week 4. That's a heavy-handed
  intervention; real drift is subtler. The simulation proves the
  *mechanism* works, not that the gate is calibrated for any specific
  real-world drift signature.

- **Weeks are SK_ID_CURR-sorted, not time-sorted.** The dataset has no
  reliable timestamp on each application. SK_ID_CURR is monotonic in
  the dataset and used as a proxy.

## Serving

- **No request validation on individual fields.** The Pydantic
  `Application` schema accepts arbitrary `payload: dict`. Missing
  required columns are filled with NaN and imputed -- meaning a
  totally wrong payload still returns *some* score. Fine for prototype,
  needs a strict schema before any real deployment.

- **Cold start ~5s.** First request after server boot is slow because
  joblib + sklearn classifier deserialise lazily. Subsequent requests
  are fast. A warm-up request in `lifespan` would fix it.

- **No /predict/batch endpoint yet.** Single-request only. Adding a
  batch route is straightforward but wasn't part of the v1 scope.

## Testing

- **Coverage is 39% by line count.** Skewed by the CLI / serve / train
  modules (which exercise via integration, not unit tests). The hot
  paths -- `gate`, `audit`, `registry`, `drift` -- are 89-100% covered.
  See README test summary.

- **No tests run against an actual served FastAPI instance.** The
  `tests/code/test_registry.py` uses a temp MLflow store; `serve.py`
  is exercised only manually via curl.

## Build / Ops

- **Docker stack hasn't been built on a machine with Docker Desktop
  installed.** Dockerfile + docker-compose are written following
  standard patterns but aren't verified end-to-end.

- **No CI workflow live yet.** `.github/workflows/ci.yml` exists in
  `.gitignore` -- the file is ready locally but not committed, so no
  CI fires on push. Add it when ready.

## What I'd do given another week

1. Bureau.csv + previous_application.csv joins -- should push AUC to
   ~0.78 on this dataset.
2. Mitigation pass using `fairlearn.ExponentiatedGradient` to train
   challengers under a fairness constraint. Compare AUC trade-off.
3. Add a "canary" alias so a fraction of traffic can be routed to a
   challenger before promotion.
4. Replace the windows-uri workaround once MLflow 3.x is stable on
   Windows.
5. Add a Streamlit operator console for browsing decisions and
   triggering rollback through a UI.
