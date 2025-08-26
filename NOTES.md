# notes

scratchpad. not docs. things I learned + mistakes + half-baked ideas.
will probably move some of this to proper docs/ later. probably not.

---

## week 1 - the idea

read a bunch of MLOps "model registry" repos. all of them stop at
"trained a model, here's the registry". none of them gate on fairness.
none of them have a working drift simulation. that's the gap.

decided to build the loop:
- weekly retrain
- challenger has to BEAT champion on auc + pr-auc + brier
- AND not regress on fairness (DP / EO ratios)
- AND not have drift in the training window > threshold
- audit log for every decision

dataset: home credit again. been working with it, no point re-learning
a new one for this exercise. the model isn't the point of this project,
the loop is.

---

## week 2 - mlflow rabbit holes

### aliases vs stages

started writing this against `transition_model_version_stage`. half a
day in I find out **stages are deprecated** in MLflow 2.9+. the new
way is aliases:

```python
client.set_registered_model_alias("credit_risk", "champion", v)
```

multiple aliases per version, no transition graph, no "archived" stage.
just pointers. cleaner. rewrote everything to use aliases. wrote ADR
001 about it.

### the windows artifact uri bug

three hours wasted on this one. the symptom:

```
MlflowException: Could not find a registered artifact repository for:
c:\Users\ASUS\AppData\Local\Temp\tmpxv2yduui. Currently registered
schemes are: ['', 'file', 's3', 'r2', 'b2', ...]
```

so on windows, when you load a model via `models:/credit_risk@champion`,
mlflow's artifact resolver gets handed a path like `c:\Users\...` and
tries to treat `c` as a URI scheme. obviously not registered.

the fix: load via `runs:/<run_id>/model` instead. retrieve the run_id
from the registry version info first, then use the runs URI. ugly but
it works. left a comment in retrain.py + serve.py explaining the
workaround.

i think this is fixed in mlflow 3.x but i'm not chasing it. note for
later: maybe switch to mlflow-3-style model URIs once 3.x is stable.

### mlflow exception types

i caught `RestException` thinking that's what `get_registered_model`
raises when the model doesn't exist. nope -- when using the SQLAlchemy
backend (which is what `sqlite:///` is), it raises a plain
`MlflowException` with `error_code="RESOURCE_DOES_NOT_EXIST"`.

caught the wrong exception type, so my `_ensure_model_exists` couldn't
auto-create the model on first run. ten minutes of head-scratching
before i looked at the actual exception class.

---

## week 3 - the gate

building gate.py was the most fun part. eight checks, each one a
small named thing. fairness is hard:

- four-fifths rule says DP ratio >= 0.80
- home credit gives DP ratios around 0.07-0.12 (deeply imbalanced
  protected attrs)
- 0.80 floor = nothing ever promotes = useless
- 0.20 floor = still nothing promotes in this dataset
- 0.05 floor = catches catastrophic drops while allowing realistic
  variance

settled on 0.05 and wrote ADR 002 explaining it's a regression guard,
not a fairness compliance claim. honest framing matters here. would
make me uncomfortable to claim "fair model" when the gate is at 0.05.

### the drift psi mystery

ran week 1. drift PSI = 1.08. set the gate at 0.5. always skips.

turns out: in early weeks the rolling window is shorter than the
configured 4 weeks (only week 0 exists for the week-0 train, only weeks
0+1 for the week-1 train). PSI between unequal-sized samples is
naturally large. nothing to do with real distribution drift.

options:
1. only enforce drift check after week 3 (when windows stabilize) --
   complicated
2. raise threshold -- simpler

went with option 2 (threshold 2.0). real production with stable history
would tighten this back down. wrote ADR 004 about window size.

### week 7 has a great result

ran the 8-week sim. week 7 challenger AUC = 0.865. CHAMPION AUC = 0.752.
that's an 11pp jump. gate rejected it because (a) fairness EO at the
floor and (b) drift PSI = 12.4.

this is the single best talking point in the project. accuracy doesn't
override safety. cap that in the README headline.

---

## week 4 - serving and polish

### fastapi loading from registry

similar windows-uri issue as above when loading via alias. used the
run_id workaround in serve.py too. lifespan loader works fine after.

### partial payloads in /predict

initially the /predict endpoint required all ~120 fields. that's
unrealistic for a real API. added a NaN-fill: cache the preprocessor's
expected_cols on load, fill any missing columns with np.nan, let the
imputer handle them.

### rollback dry-run

made dry-run the default for rollback. you have to pass --no-dry-run
or actually do the swap. wait no, dry-run is the FLAG you pass; default
is real rollback. hmm. should that be inverted? operationally you want
the safer default. for portfolio purposes the explicit `--dry-run` is
clearer. left it.

---

## things that would make this better

- bureau.csv + previous_application.csv joins. would give better AUC
  and more interesting fairness dynamics on the protected slices.
- a real prefect agent instead of GitHub Actions cron. but actions
  cron costs nothing and runs in CI. for portfolio it's fine.
- canary deployment using multiple aliases (@canary in addition to
  @champion). architecturally easy with aliases, didn't build it.
- gemini-flash LLM rationale on each decision. retrain decisions don't
  really need it; the JSON is readable.

---

## stuff i considered and didn't do

- a heavy pipeline/orchestration framework (pipelines, steps, hooks).
  too much abstraction. this is plain mlflow + sklearn + scheduled
  scripts. moves slower in the small but isn't locked to anything.
- Airflow. weekly retrain doesn't need DAG semantics.
- Kubernetes. portfolio project, runs locally.
- A "manual approval" step in the workflow. interesting but adds a
  human bottleneck. the gate IS the approval.

---

## todo / parked

- 03_promotion_audit_log notebook is done but could use a sparkline
  view of decisions over time
- LightGBM compare in experiments/, never finished
- coverage on cli + serve + train modules
- the warning "DataFrame.swapaxes is deprecated" from
  test_split_into_weeks_count_and_order -- numpy upgrade thing
