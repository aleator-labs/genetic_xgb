# genetic_xgb Roadmap — From Library to Best-in-Class Prediction Platform

**Date:** 2026-07-02 (v2 — revised after multi-agent review)
**Status:** Approved for execution
**Review provenance:** v1 was reviewed by a 17-reviewer multi-agent workflow (11 per-item + 6
cross-cutting reviewers; 71 agents total), findings deduplicated and adversarially verified —
48 of 53 findings confirmed and incorporated below.
**Decisions made with user:** staged roadmap serving three audiences in order (data scientists →
ML engineers → AI agents); XGBoost-only, go deep; separate branch per feature; 100% branch
coverage; multi-agent review workflow + test gate + CI before every merge/release; model routing
(Fable orchestrates, Sonnet implements, Opus reviews/decides).
**Decisions made autonomously (flagged for revisit):** deployment target is a *train + serve*
compose stack; MLflow is the first tracker; the 1.1 callback contract is the **rich** variant;
the 2.3 trainer **service is kept** with an explicit execution model (not cut to a CLI image);
MLflow models are logged as **pyfunc wrapping the 1.2 safe format** (never estimator pickle);
immigration uses **age protection** (not catch-up training).

## Where the project stands

`genetic_xgb` is a ~1,350-LOC, 100%-branch-coverage, CI-gated library: GA-evolved XGBoost
hyperparameters with warm-start boosting, elitism/crossover/mutation, native early stopping,
genetic feature selection with overfit-aware fitness, and genuine sklearn compatibility (clone,
pipelines, CV). Multiclass already works (`multi:softprob`); `nthread=1` is already pinned per
member. What it lacks is everything *around* the core: observability, resumability, safe
persistence, integrations, deployment, machine-usable surfaces — and the repo lacks enforced
governance (0 required approvals, one stale-tolerant required check, no security jobs).

## Approaches considered

1. **Library-first, staged (chosen).** Harden the core and add a callback spine first; build MLOps
   on the callbacks; build AI surfaces on the MLOps layer. Each phase ships adoption value alone.
2. **Service-first.** Fast "wow", fragile foundation — every robustness gap becomes a production
   bug report. Rejected.
3. **Breadth-first.** Multi-backend + every tracker at once. Rejected by the scope decision.

## Architecture principle

The dependency structure is a **fan-out, not a line**:

```
Phase 0 governance/CI ─ gates everything
callback/event API (1.1) ──┬─→ MLflow callback (2.1)
                           ├─→ CLI + run reports (2.2) ─→ docker services (2.3) ─→ MCP HTTP transport (3.1)
                           └─→ JSONL event stream (ships IN 1.1) ─→ trainer live status (2.3), MCP get_run_status (3.1)
auto_fit / profiling (1.2) ─→ autopilot surfaces (3.2)
MCP stdio transport needs only 2.2 (local subprocess) — shippable before docker.
2.1 and 2.2 are parallel tracks.
```

Every run-observation surface (history_, MLflow metrics, report.json, trainer status, JSONL)
is a **view over one versioned generation-end event schema** — one schema to design, version,
and test, not five. Shipped reference callbacks (JSONL writer, progress bar) prove the API
carries real weight from day one.

---

## Phase 0 — Repo governance & CI/CD as code (gates Phase 1)

The "never skipped" guarantee is an enforcement workstream, not prose. **Exit criterion: 1.1 does
not start until the ruleset and `ci-required` are live on main.**

- **Branch protection as code:** ruleset JSON checked into the repo + a `gh api` apply script,
  reviewed like code — `strict=true` (up-to-date branches), required conversation resolution,
  dismiss stale reviews. Required check is a single aggregate **`ci-required`** job
  (`needs:` all lint/test/security/build jobs) so protection survives job renames and matrix
  growth. Do **not** require ≥1 human approval (solo maintainer cannot self-approve; it would
  deadlock merges).
- **CI hardening now:** top-level `permissions: contents: read`, SHA-pinned actions,
  `uv sync --locked`.
- **Security lane** (all folded into `ci-required`): `pip-audit` against `uv.lock` per PR + weekly
  scheduled; ruff S-rules (bandit-class checks); `actions/dependency-review-action`;
  `dependabot.yml` for pip + github-actions; GitHub secret-scanning push protection enabled.
- **Type checking:** `mypy --strict` (per-module `ignore_missing_imports` for
  xgboost/sklearn/joblib) as a required PR job; mirrored in `.githooks/pre-push` between ruff and
  pytest. `src/genetic_xgb/py.typed` created and asserted present in the wheel by a packaging test.
- **Agent review as CI:** an `agent-review` workflow on `pull_request` running headless
  code-review + security-review over the diff, posting inline findings, **failing on confirmed
  correctness/security issues**, added to required checks. The deep multi-agent drop-in review
  (existing `review-dropin` pattern) runs scheduled and before releases.
- **Standing rule (quality bar):** every gate this spec promises (CLI golden files, compose smoke,
  MCP contract tests, …) must land **in the same PR** as its addition to `ci-required` —
  otherwise it is advisory-only. `.githooks/pre-push` is developer convenience; CI required-checks
  are the sole enforcement boundary.

## Phase 1 — Core depth & robustness (win data scientists)

### 1.0 Release engineering first (before 1.1 lands)

Ship the pipeline, then develop against it — 1.1–1.3 each release as versioned 0.x.

- Fill pyproject metadata (classifiers, `project.urls`, keywords, authors); claim the `genetic-xgb`
  PyPI name immediately with 0.1.0 from current main (closes the squatting window; users can
  `pip install` from day one).
- Per-PR **packaging job** in `ci-required`: `uv build`, `twine check`, install wheel into clean
  venv, `import genetic_xgb`, smoke-fit a tiny model — broken sdists fail PRs, not releases.
- Tag-triggered `release.yml`: blocking gate (tag == pyproject version, CHANGELOG section exists,
  full matrix green, pip-audit clean) → PyPI **trusted publishing** (OIDC, PEP 740 attestations)
  behind a protected `release` environment → GitHub Release from the CHANGELOG entry.
- CHANGELOG with CI-enforced maintenance (release-please or towncrier); stated 0.x semver policy
  (breaking changes bump minor, flagged in changelog).
- **CI matrix** (gating deliverable of the PyPI milestone): Python 3.11/3.12/3.13 on ubuntu —
  100%-coverage gate only on the primary cell (ubuntu+3.13+uv.lock), plain pytest elsewhere; one
  lowest-bounds job (`uv sync --resolution lowest-direct`) validating the declared floors
  (xgboost>=2.0 etc.) or raise the floors; one macOS **or** Windows smoke leg no later than the
  1.2 thread work (xgboost OpenMP wheels and loky differ off Linux). Rationale: warm-start
  (`xgb.train(xgb_model=...)`) and booster save/load round-trips are the APIs that drift between
  xgboost majors.
- **OSS hygiene files** (precondition for first publish): CONTRIBUTING.md (documents the enforced
  workflow), SECURITY.md (supported versions + private vulnerability reporting enabled; revise
  when 2.3 ships network services), CODE_OF_CONDUCT.md, issue templates (bug w/ repro+env,
  feature), PR template whose checklist mirrors the CI gates.

### 1.1 Callback API (keystone) — rich contract

```python
class Callback:
    def on_fit_start(self, ctx: RunContext) -> None: ...
    def on_generation_end(self, ctx: RunContext, gen: GenerationResult) -> bool | None: ...
    def on_fit_end(self, ctx: RunContext, result: FitResult) -> None: ...
    # optional: on_checkpoint_save/on_checkpoint_load (see 1.2)
```

- **Rich contract (decision):** `RunContext` carries resolved estimator params (`get_params()`
  output, not just a hash), the resolved `MetricSpec` (name, greater_is_better, min_delta),
  `n_features_in_`, and the run manifest. `GenerationResult` carries the live
  `list[PopulationMember]` (genomes, booster bytes, masks, per-member scores/parents/rounds,
  **and a per-member `objectives: dict[str, float]` vector** — future-proofs Pareto), best-so-far
  (score/params/mask), an RNG state accessor, generation index, aggregate stats
  (best/mean/std fitness, wall-clock), and the **diversity metric**.
- **Diversity metric is a defined deliverable, not a field name:** mean per-gene normalized
  dispersion over numeric genes *in transformed space* (log for log-scale genes) + categorical
  entropy + mean pairwise Hamming distance over feature masks, reported as separate components;
  computed in the core generation-end path; recorded in `GenerationResult` and `History`.
- **Lifecycle contract:** the generation loop runs in `try/finally`; a terminal hook always fires.
  `FitResult` carries terminal status (`completed | stopped_early | interrupted | failed`). On
  mid-run exception/interrupt, best-so-far fitted attributes (`best_booster_`, `best_params_`,
  `history_`) are still set when ≥1 generation finished — the single mechanism behind 1.2's
  interrupt promise. One callback raising does not suppress others' terminal hooks (each invoked
  in its own try; re-raise after).
- **Multi-callback semantics:** every callback's `on_generation_end` runs every generation (no
  short-circuit); the run stops if ANY returned True (OR); documented order = internal first
  (history recording → patience/target stopping → user callbacks); a requested stop **skips
  `strategy.evolve`** so fitted state reflects the last evaluated population (preserves the
  current break-before-evolve invariant).
- Internals (history recording, patience/target stopping) are refactored to *be* callbacks.
- **Shipped reference callbacks:** `JsonlEventCallback` (surfaced as `events_path=`) — one
  `json.dumps` per generation-end + fit-start/fit-end records, with a **versioned event schema**
  (`schema_version`, published JSON Schema, CI contract test); and a progress-bar callback.
  The JSONL stream is the out-of-process observation surface 2.3 and 3.1 later tail.

### 1.2 Robustness & autonomy of a single run

- **Canonical safe persistence format (prerequisite, ships before/with checkpoints):**
  estimator-level save/load producing a directory: versioned JSON manifest (schema_version,
  config, seeds, versions), native booster bytes (`save_raw()`/UBJ — already how boosters are
  stored internally), masks as `.npz`, JSON sidecar for `feature_mask_` metadata / `classes_` /
  search-space + RNG state. **No pickle/cloudpickle anywhere in checkpoint or run-directory
  formats; deserialization must never execute code.** Fixes the current `save_model`
  NotImplementedError path; 2.1's MLflow artifact, 2.2's run directory, and 2.3's serving all
  wrap this one format. README's pickle/joblib advice is demoted to "trusted environments only".
- **Checkpoint/resume, full payload:** generation index; per-member state (incl. age/adaptive
  fields from 1.3); best-so-far (score/bytes/params/member); patience counters; accumulated
  History rows; RNG `bit_generator.state`; **internal train/val split row indices** (with
  `random_state=None` the split is otherwise irreproducible — resuming with a new split leaks old
  val rows into training); an extensible callback-state dict via `on_checkpoint_save/load` (lets
  MLflowCallback persist its run_id and reattach instead of duplicating runs). `classes_` is
  re-derived on resume (stated explicitly). Checkpoint schema carries a **version field and
  tolerates unknown/absent sections** so 1.3 state additions don't invalidate 1.2 checkpoints.
- **Resume safety & write strategy:** `resume=True` validates config hash, package versions, and
  data shape/hash from the checkpointed manifest (actionable error on mismatch); checkpoints are
  latest-only, temp-file + atomic rename, `checkpoint_every=N` knob (boosters grow every
  generation; snapshots are large).
- **Budgets & graceful stop:** `time_budget=` checked before each generation AND between member
  chunks (executor dispatches in ~n_jobs chunks) so runs can stop mid-generation;
  `refit_on_full` counts against the budget (skipped with actionable warning +
  `refit_full_=False` when remaining budget insufficient, estimated from observed per-round
  cost). Graceful stop is an **externally-triggerable cancellation signal** (flag/event checked
  at generation boundaries via the callback early-stop path) — KeyboardInterrupt is just one
  trigger; a cancelled run still returns best-so-far model, manifest, report. **Interrupt
  default: re-raise** (`on_interrupt="raise"|"return_best"`, default raise) after finalizing
  best-so-far state and writing a final checkpoint — swallowing Ctrl-C breaks CV loops and CI;
  the in-flight generation is discarded (all-or-nothing joblib map; stated limitation).
- **Preflight as a public module:** `profile(X, y) -> DatasetProfile` +
  `suggest_config(profile, budget) -> dict`; preflight warnings (constant/all-NaN columns, inf,
  train/val duplicate-row leakage, extreme imbalance, val-too-small-for-metric) are a consumer of
  `DatasetProfile` (`validate="warn"|"raise"|"off"`). This single module later backs
  `auto_fit` (1.4), the CLI `profile` subcommand (2.2), and MCP `profile_dataset` (3.1) — one
  profiler, three surfaces.
- **Thread allocation (corrected):** `nthread=1` per member is already the default — the work is
  allocation, not oversubscription-prevention: per-member
  `nthread = max(1, cores // effective_parallel_workers)` (sequential executor → all cores);
  lift the `nthread=1` pin in `refit_full` and single-model paths; set nthread on boosters loaded
  for prediction; warn/clamp when user `base_params` nthread × workers exceeds cores (the only
  real oversubscription path).
- **Native categorical support** (sequenced before 1.4 benchmarks — a data-path change, not a
  flag): preserve DataFrames with `category` dtypes through validation (today coerced to
  float32), masks over semantic columns, `enable_categorical=True` at every DMatrix site,
  categorical-aware save/load. **GPU verdict (decision, recorded in out-of-scope):** CPU-only
  through Phase 2; revisit after (population-parallel joblib × `device=cuda` needs deliberate
  scheduling); benchmark methodology, README, and the 2.3 image choice reference this.
- **Run manifest:** config, seeds, versions, data shape/hash → `run_manifest_`; reused by
  MLflow, reports, MCP.
- **Error-message convention (moved from Phase 3):** every raised exception follows
  what-happened / likely-cause / concrete-fix — codified as a documented convention + PR-checklist
  item now (the codebase already follows it), so new raise sites in callbacks/checkpoints/CLI are
  written to the bar, not retrofitted.

### 1.3 GA depth — evidence-gated

**Prerequisite/order:** the 1.4 benchmark harness lands **before or alongside the first 1.3
feature**. Acceptance criterion for every 1.3 feature: a pinned-seed **ablation** (feature
on vs off, equal time budget, OpenML suite) showing non-regression in median rank, run in the
scheduled benchmark workflow and attached to the PR.

- **Adaptive mutation (defined):** signal = **offspring displacement rate** (fraction of new
  offspring entering next generation's top_k — "acceptance rate" is undefined in elitist
  truncation). Adapted value is per-run internal state exposed as a fitted attribute — never a
  write to public `mutation_intensity` or the user-injected `strategy=` object (clone() shares it
  by reference; in-place adaptation breaks the get_params/clone contract). `Strategy` gains an
  optional `on_generation(stats)` adaptation hook that custom strategies may ignore.
  Hypermutation burst consumes the 1.1 diversity metric.
- **Immigration (decision: age protection):** immigrants are exempt from replacement for N
  generations (`age` field on `PopulationMember` — included in the 1.1 event shape and 1.2
  checkpoint from the start), because cold-started immigrants essentially never survive
  truncation selection against warm-started incumbents.
- **Diversity-mechanism composition policy (escalation ladder):** per-gene `resample_prob` always
  active → hypermutation on diversity collapse → immigration only if diversity hasn't recovered
  after N further generations. Triggers cannot fire simultaneously with undefined interaction.
- **Island model (corrected design):** logical subpopulations inside the existing synchronous
  loop — partition members into K groups, selection/crossover/mutation per group, ONE
  `executor.map` across all members, migration every M generations moves whole members (genome
  AND booster bytes so warm-start lineage survives). No "maps onto joblib parallelism" claim —
  islands add no parallelism. Validation: per-island size ≥ 2, `selection_top_k` enforced per
  island. Asynchronous per-island execution → out of scope.
- **Scalarized secondary objective (replaces Phase-1 Pareto):** `size_penalty` mirroring the
  existing `overfit_penalty` mechanism — `score = val_fitness − direction × size_penalty ×
  complexity`, complexity defined **round-count-neutrally** (per-tree average or at
  best_iteration) so old lineages aren't penalized for age. Keeps strategy/patience/History/
  checkpoints/sklearn semantics untouched. Full Pareto selection (`pareto_front_`) moves to a
  post-benchmark milestone gated on 1.4 evidence; the per-member `objectives` vector (1.1) makes
  it additive later.

### 1.4 Credibility artifacts

- **Benchmark protocol (stated, not vibes):** all methods scored on a held-out TEST split no
  search ever sees (never `best_score_` — README's own caveat); ≥5 seeds per (dataset, method)
  with mean±std/CIs; FLAML pinned to `estimator_list=['xgboost']` (search vs search, not
  XGBoost vs LightGBM); untuned-XGBoost-defaults baseline included; categorical-heavy OpenML
  datasets in the matrix; runs after 1.2 so equal budgets use native `time_budget=`.
- **Benchmark execution contract:** pinned larger runners (4+ cores — 2-core runners nullify the
  parallel-evaluation advantage the benchmark demonstrates); per-dataset matrix shards +
  OpenML download cache (never near the 6-h limit); ≥3 seeds minimum; every published number
  traceable — workflow commits `benchmarks/results/<date>-<version>.json` (versions, runner
  spec, seeds, budgets) and the docs page renders from those files, never hand-copied.
- **Plotting module:** `genetic_xgb.plotting` behind a `[plot]` extra (matplotlib moves from dev
  group): `plot_history`, `plot_feature_selection`, later `plot_pareto_front`. Notebook-facing
  Phase-1 surface; feeds the docs gallery; 2.2's report.html embeds these same functions (no
  second plot implementation).
- **Docs site (mkdocs-material):** quickstart, concepts, API reference, benchmark page, gallery,
  plus two conversion pages: **"Migrating from Optuna / RandomizedSearchCV / FLAML"**
  (side-by-side idiom mapping for exactly the benchmarked baselines) and **"When (not) to use
  genetic_xgb"** (honest positioning from the YAGNI list). `mkdocs build --strict` per-PR gate
  from the day the site lands; versioned deploy (mike) on release.
- **`auto_fit(X, y, time_budget=...)` lands here (end of Phase 1)** — it depends only on
  1.2 (profile/suggest_config, time_budget, manifest) + 1.3, and serves the Phase-1 audience.
  No `autopilot=True` constructor flag (would break get_params/clone truthfulness): auto_fit
  returns an estimator constructed with the decided params (get_params stays truthful),
  decisions surfaced as `autopilot_decisions_` + recorded in `run_manifest_`; includes a timed
  calibration probe (train 1–2 members briefly, size population/rounds/generations to the
  budget, record the measurement + decision); clone+CV estimator-checks test included.

---

## Phase 2 — MLOps & deployment (win ML engineers)

### 2.1 MLflow integration (`pip install genetic-xgb[mlflow]`)

- `MLflowCallback` on the 1.1 API: params from RunContext (resolved get_params + manifest) at
  start; per-generation best/mean fitness, diversity, n_features_selected as step metrics;
  it **logs — not defines —** the 1.1 diversity metric.
- **Model logging (decision):** `mlflow.pyfunc.log_model` with an artifacts directory in the
  **1.2 safe format** and a small loader module — never estimator pickle
  (registry write access must not equal arbitrary code execution). Includes (a) signature
  inferred from the **full pre-mask input schema** (`n_features_in_` columns — the model's true
  contract; the mask applies inside predict), (b) `input_example`, (c) `pip_requirements` pinned
  from `run_manifest_`. Acceptance test: log → `load_model` → predict on full-width X equals
  `estimator.predict`; mask survives the round trip.
- **Lifecycle:** run ends FAILED on exception, KILLED on interrupt, FINISHED otherwise (via the
  1.1 terminal hook) — no runs stuck RUNNING, no leaked active-run context.
- **Autolog suppression:** scoped `mlflow.autolog(disable=True)` around internal training
  (population_size × generations `xgb.train` calls + refit_full would otherwise spawn junk runs;
  autolog is on by default on Databricks); restored in on_fit_end. Test: fit under
  globally-enabled autolog produces exactly one run.
- Registry: in the compose flow, registration + alias promotion is the default (not "optional").
- Core stays tracker-agnostic — W&B later is just another callback.

### 2.2 CLI — the bridge to docker and agents

```
genetic-xgb train    --config train.yaml --data train.parquet --target y [--val val.parquet]
                     [--resume-from runs/<id>/]
genetic-xgb profile  --data train.parquet --target y
genetic-xgb evaluate --model model/ --data test.parquet [--metrics roc_auc,f1]
genetic-xgb predict  --model model/ --data new.parquet --out preds.parquet [--proba]
genetic-xgb report   --run-dir runs/<id>/
genetic-xgb agent-init            # writes AGENTS.md + train-model skill into consumer project
```

- **Config schema commitment (not "1:1 mirror" — several constructor args are Python objects):**
  YAML carries `task: classification|regression` and `target:` (neither is a constructor arg);
  metrics by registered name only; declarative search-space block (built-in space +
  extended/imbalance flags + optional per-param {low, high, log} overrides); scalar estimator
  kwargs. Callables/custom objects remain library-API-only. `yaml.safe_load` + schema validation
  (pydantic/jsonschema): unknown keys rejected, type/range-checked, errors in the house style.
  **The same schema object validates CLI, 2.3 POST /train, and 3.1 train_model**, with
  service-layer resource caps applied at the 2.3/3.1 boundary.
- **Data contract:** CSV + parquet by extension; parquet via a `[cli]` extra pinning pyarrow +
  CLI framework + report deps; non-numeric feature columns error actionably (encoding is out of
  scope pre-1.0; interacts with 1.2 preflight); `predict --proba` writes class-probability
  columns. 2.3 and 3.1 reuse this contract.
- **Run directory:** model (1.2 safe format), manifest, `events.jsonl` (always written —
  checkpoint_dir also defaults into the run dir, so every CLI run is resumable by construction),
  `report.json`, `report.html` (thin template over `genetic_xgb.plotting`).
- **report.json is a versioned contract:** `schema_version` shared with run_manifest_ and the
  JSONL event schema (one schema discipline — report.json is the materialized final view of the
  events log + manifest); additive-only within a major schema version, breaks bump it + CHANGELOG;
  JSON Schema committed to the repo; CI contract test validates real CLI output against it;
  AGENTS.md (3.3) and MCP get_run_report tests (3.1) reference the same schema file.

### 2.3 Docker compose — train + serve stack

`docker compose up` brings up **mlflow**, **trainer**, **server**:

- **mlflow:** `mlflow server --serve-artifacts` (proxied `mlflow-artifacts:/`) so containers never
  depend on sharing filesystem paths; sqlite + local volume default, postgres/minio documented as
  a storage-backend-only variant. A stated **volume contract** paragraph: where dataset input,
  run directories, checkpoints, and the backend store mount.
- **trainer (decision: keep the service, with an execution model):** FastAPI job service; each
  job is a CLI subprocess under a bounded pool, default `max_concurrent_jobs=1` (n_jobs=-1
  saturates all cores per job); **persistent job registry** on the run-dir volume so
  `GET /runs/{id}` answers after container restart (states: queued/running/succeeded/failed/
  interrupted); on restart, interrupted jobs re-invoke the CLI with `--resume-from`;
  `GET /runs/{id}` **tails the run's events.jsonl**; `POST /runs/{id}/stop` wired to the 1.2
  cancellation signal; `POST /datasets` (upload or register a mounted path) → `dataset_ref`
  consumed by `POST /train`; `GET /runs/{id}/artifacts/{name}` for model/report/predictions.
  This surface is under the 100%-coverage bar (Phase 2 is sized for it).
- **server:** starts model-less, returns 503 from /predict with an actionable error until a model
  exists; model selection via MLflow registry **alias (default `champion`)** from env; 2.1's
  registration+promotion is the compose default; `POST /model/reload` (or documented poll) so a
  new champion serves without stack restart; `GET /model/info` derives expected features from
  the logged signature + manifest. Loads models **only** through the registry (pyfunc over the
  1.2 safe format).
- **2.3.1 Security baseline (ship-blocker for the stack, in the sequencing table):**
  compose-internal network, only server + MLflow UI published and bound to 127.0.0.1 by default;
  shared API token generated into `.env`, required by trainer and server; dataset refs resolved
  strictly under a read-only `/data` root with realpath traversal rejection, URL refs off by
  default; body-size limits and timeouts everywhere; server-side caps on `time_budget` +
  one-job-at-a-time queue; non-root containers with memory/CPU limits; a short threat-model
  paragraph including: **registry write access equals code execution in the serving container**;
  default stack is for trusted/local networks. Compose smoke test asserts unauthenticated
  /train and /predict are rejected.
- **Images:** GHCR publishing added as jobs in the same 1.0 tag-triggered `release.yml` —
  build once, run the compose smoke test against the exact built images, trivy/grype scan gating
  the push (fail high/critical), syft SBOM + provenance attestations + cosign signing,
  semver + latest tags. CPU base image (per the 1.2 GPU verdict).

---

## Phase 3 — AI-ready (win agents as users)

### 3.1 MCP server (`genetic-xgb-mcp`)

Transports: **stdio ships first** (needs only 2.2 — local subprocess, local paths); HTTP fronts
the 2.3 stack. Tools:

- `profile_dataset(ref)` — thin wrapper over the 1.2 profiler
- `upload_dataset(...)` (HTTP transport) → `dataset_ref`; stdio accepts local paths
  (per-transport semantics in tool descriptions)
- `train_model(dataset_ref, config)` → run id (2.3 trainer or local subprocess; same config
  schema + caps as 2.2/2.3)
- `get_run_status(run_id)` — tails events.jsonl; `get_run_report(run_id)` — report.json
- `cancel_run(run_id)` — wired to the 1.2 cancellation signal via 2.3's stop endpoint
- `evaluate_model(model_ref, data, metrics=None)` — held-out scores through the metrics registry
  (closes the measurement step of the profile→train→**evaluate**→iterate loop); matching CLI verb
  exists in 2.2
- `predict(model_ref, data)` — returns a summary (row count, class distribution/stats) + an
  artifact ref for full predictions, never the inline array
- `list_models()` — each model's manifest summary (metric, best_score_, n_features, timestamp)
  so agents rank candidates without N calls

**Tool contract & security (designed with the tool list; enforcement lands in the 2.3 services):**
HTTP requires bearer-token auth per the MCP authorization spec (also covers the 2.3 FastAPI
endpoints); stdio may remain unauthenticated. Path-taking tools resolve under the data-root
sandbox with realpath containment. `predict`/`evaluate_model` accept only registry names /
run IDs enumerable via list_models — never filesystem paths (model loading is code-adjacent).
Server-side caps enforced independently of submitted configs (max time_budget, population,
n_jobs, concurrent jobs). Contract tests include traversal rejection, unauthenticated-request
rejection, out-of-registry model_ref rejection, cap enforcement.

### 3.2 Autopilot surfaces

`auto_fit` itself shipped in 1.4. Phase 3 exposes it to agents: `suggest_config` via MCP
(profile → recommended config with reasons), `autopilot_decisions_` in report.json, and the
upload→profile→train→evaluate→iterate loop documented end-to-end.

### 3.3 Agent-native ergonomics

- **Shipped skill + AGENTS.md with a delivery path:** packaged in the wheel;
  `genetic-xgb agent-init` writes AGENTS.md + the train-model skill into the consumer's project;
  CI contract test asserts the AGENTS.md-documented report.json schema matches actual CLI output.
- **Errors:** Phase 1 owns the prose convention; 3.3 is the machine-parseability pass — stable
  error codes / structured fields audited across surfaces.
- (JSONL event stream moved to 1.1; LLM-friendly-errors convention moved to 1.2.)

---

## Explicitly out of scope (YAGNI)

Multi-backend boosters; Kubernetes; distributed training (Ray/Dask); **asynchronous per-island
execution**; preprocessing/pipeline search (full AutoML); a custom web UI (MLflow is the UI);
learning-to-rank (revisit after Phase 1); **GPU training: CPU-only through Phase 2, revisit
after** (population-parallel joblib × device=cuda needs deliberate scheduling work); **full
Pareto multi-objective selection until 1.4 benchmarks show demand** (scalarized penalty ships
first; the objectives vector keeps it additive).

## Testing & quality bar

Baseline (already enforced, now extended): **100% branch coverage, ruff lint + format, mypy
strict, CI on PRs** — plus the Phase 0 standing rule: a promised gate lands in the same PR as
its addition to `ci-required`.

**Phase 1 gates** (acceptance gates for merging 1.2/1.3 — 100% coverage alone cannot catch
inverted triggers or resume divergence in stochastic control loops):
- Resume equivalence: a run interrupted at generation k and resumed reproduces the uninterrupted
  run's population state (genomes, masks, RNG state) and best score/params (fixed seed, n_jobs=1).
- Interrupt injection: KeyboardInterrupt raised from a callback mid-run → best-so-far fitted
  estimator finalized, re-raise honored; time_budget stops recorded in run_manifest_.
- Behavioral GA tests: hypermutation fires on a forced diversity collapse; immigration replaces
  members on a forced plateau (and age protection holds); island migration moves genomes at the
  configured interval — pinned-seed constructed scenarios, not just branch execution.
- Event/report schema contract tests; packaging test (py.typed in wheel, import + smoke-fit).

**Security gates** (phased with the attack surface):
- Phase 0/1 (before first PyPI publish): hardened ci.yml (permissions, SHA pins, --locked),
  dependabot, pip-audit per-PR + weekly, ruff S-rules, dependency review, secret-scanning push
  protection, PEP 740 attestations on publish.
- Phase 2 (release-blocking in the GHCR jobs): trivy/grype image scan, syft SBOM, cosign;
  input-hardening suite for trainer/server (safe_load only, traversal rejection,
  registry-only model loading); compose smoke asserts auth is enforced.
- Phase 3: MCP contract tests inherit the input-hardening cases; a one-page threat model of the
  MCP tool surface is an explicit deliverable.

**CI trigger design** (which gates run when):
- **Per-PR:** ruff, mypy, test matrix, packaging job, pip-audit, dependency review, agent-review
  workflow, schema contract tests, `mkdocs build --strict` (once docs exist), MCP stdio contract
  tests (fast) — all aggregated in `ci-required`.
- **Per-merge to main + path-filtered PRs (docker//cli changes):** compose smoke test
  (up → healthcheck → toy train → register/promote → /predict roundtrip, incl. auth-rejection
  and upload→train→fetch-artifact).
- **Scheduled:** weekly pip-audit; benchmark workflow (weekly cron + workflow_dispatch, pinned
  seeds, ≥3 seeds, publishes versioned results + docs page, alerts on regression beyond
  tolerance); deep multi-agent drop-in review.
- **Per-release (tag):** full gate (version/CHANGELOG/matrix/pip-audit) → PyPI publish →
  image build + smoke on built images + scan/SBOM/sign → GHCR → docs deploy (mike).

## Sequencing summary

| Order | Ships | Depends on |
|-------|-------|-----------|
| 0 | ruleset-as-code, `ci-required` aggregate, security lane, mypy+py.typed, agent-review workflow | — (gates 1.1) |
| 1.0 | pyproject metadata, PyPI 0.1.0 + trusted publishing, CHANGELOG, CI matrix + lowest-bounds + macOS/Windows leg, packaging job, OSS hygiene files | 0 |
| 1.1 | rich callback API + lifecycle, diversity metric, JSONL event stream + schema, progress callback, internals-as-callbacks | 0 |
| 1.2 | safe persistence format, checkpoints/resume, budgets + cancellation, profiler module, thread allocation, categorical support, error convention | 1.1 |
| 1.4a | benchmark harness + protocol (pulled ahead of 1.3) | 1.2 |
| 1.3 | adaptive mutation, age-protected immigration, escalation ladder, islands, size_penalty — each ablation-gated | 1.1, 1.2, 1.4a |
| 1.4b | docs site + migration/positioning pages, plotting module, auto_fit, benchmark publication | 1.2, 1.3 |
| 2.1 | MLflowCallback, pyfunc-over-safe-format + signature/reqs, autolog suppression | 1.1, 1.2 (parallel with 2.2) |
| 2.2 | CLI + config schema + data contract, versioned report.json/events, evaluate/profile/agent-init verbs | 1.1, 1.2 |
| 2.3 | compose stack + trainer execution model + serving lifecycle + **2.3.1 security baseline**, GHCR release jobs | 2.1, 2.2, 1.0 |
| 3.1a | MCP stdio server + tool contract tests | 2.2 |
| 3.1b | MCP HTTP transport + auth | 2.3 |
| 3.2/3.3 | autopilot surfaces, shipped skill/AGENTS.md + agent-init contract test, error-code audit | 1.4b, 3.1 |
