# genetic_xgb Roadmap — From Library to Best-in-Class Prediction Platform

**Date:** 2026-07-02
**Status:** Draft — awaiting user review
**Decisions made with user:** staged roadmap serving three audiences in order (data scientists → ML engineers → AI agents); XGBoost-only, go deep (no LightGBM/CatBoost backends).
**Decisions made autonomously (user AFK, revisit if wrong):** deployment target is a *train + serve* docker compose stack; MLflow is the first tracker integration, built on a tracker-agnostic callback API; the MCP server is the flagship AI surface.

## Where the project stands

`genetic_xgb` is a ~1,350-LOC, 100%-coverage, CI-gated library: GA-evolved XGBoost hyperparameters
with warm-start boosting, elitism/crossover/mutation, native early stopping, genetic feature
selection with overfit-aware fitness, and genuine sklearn compatibility (clone, pipelines, CV).
Multiclass already works (`multi:softprob`). What it lacks is everything *around* the core:
observability, resumability, integrations, deployment, and machine-usable surfaces.

## Approaches considered

1. **Library-first, staged (chosen).** Harden the core and add a callback spine first; build MLOps
   on the callbacks; build AI surfaces on the MLOps layer. Each phase ships adoption value alone,
   and each layer stands on a tested one below it.
2. **Service-first.** Build the MCP server + compose stack now for a fast "wow", harden later.
   Rejected: the service would wrap a core with no events, no checkpoints, no run manifests — every
   robustness gap becomes a production bug report.
3. **Breadth-first.** Multi-backend (LightGBM/CatBoost) + every tracker at once. Rejected by the
   scope decision: it dilutes the warm-start GA advantage (each backend warm-starts differently)
   and triples maintenance before there are users.

## Architecture principle

One dependency chain runs through the whole roadmap:

```
callback/event API  →  MLflow logging  →  CLI + run reports  →  docker services  →  MCP server / skills
```

Every downstream surface (progress bars, MLflow, checkpoints, JSONL event streams, MCP status
polling) consumes the same generation-end event. Build that spine once, in the core, first.

---

## Phase 1 — Core depth & robustness (win data scientists)

### 1.1 Callback API (keystone)

```python
class Callback:
    def on_fit_start(self, ctx: RunContext) -> None: ...
    def on_generation_end(self, ctx: RunContext, gen: GenerationResult) -> bool | None: ...
    def on_fit_end(self, ctx: RunContext, result: FitResult) -> None: ...
```

- `GenerationResult`: generation index, per-member scores, best/mean/std fitness, diversity metric,
  feature-mask stats, wall-clock. `RunContext`: config hash, seeds, library/XGBoost versions.
- Returning `True` from `on_generation_end` requests early stop (budget callbacks, agent aborts).
- Estimators accept `callbacks=[...]`. Internals (history recording, patience/target stopping)
  are refactored to *be* callbacks, proving the API carries real weight.

### 1.2 Robustness & autonomy of a single run

- **Checkpoint/resume:** `checkpoint_dir=` serializes population state (genomes, boosters, masks,
  RNG state) every generation; `fit(..., resume=True)` continues after crash/interrupt.
- **Budgets & graceful stop:** `time_budget=` (seconds) stops cleanly at a generation boundary;
  `KeyboardInterrupt` returns the best-so-far fitted model instead of losing the run.
- **Preflight data validation** (on by default, `validate="warn"|"raise"|"off"`): constant/all-NaN
  columns, inf values, duplicate rows shared between train and val (leakage), extreme class
  imbalance, val set too small for the metric. Warnings are actionable sentences.
- **Thread hygiene:** set per-member `nthread` so joblib workers × XGBoost threads never
  oversubscribe cores (today `n_jobs=-1` can explode to N×cores threads).
- **Run manifest:** every fit records config, seeds, package versions, data shape/hash into
  `run_manifest_` — the reproducibility receipt everything later (MLflow, reports, MCP) reuses.

### 1.3 GA depth — the "hot features" of the genetic side

- **Adaptive mutation:** self-tune `mutation_intensity` from acceptance rate; hypermutation burst
  when population diversity collapses (diversity tracked in `history_`).
- **Immigration:** replace stagnant members with fresh random genomes on plateau.
- **Island model:** `islands=K` evolves K subpopulations with periodic migration — maps perfectly
  onto the existing joblib parallelism and is a visible differentiator vs Optuna/FLAML.
- **Multi-objective fitness:** optimize `(validation metric, model size or predict latency)` with
  Pareto-based selection; `pareto_front_` on the fitted estimator. This targets people who *ship*
  models, not just score them.

### 1.4 Credibility artifacts

- **Benchmark suite** (`benchmarks/`): genetic_xgb vs Optuna(TPE) vs FLAML vs random search on
  ~10 OpenML datasets at equal time budgets; results published in docs and README.
- **Docs site:** mkdocs-material — quickstart, concepts (why GA + warm-start), API reference,
  benchmark page, gallery from the existing notebooks.
- **Release engineering:** CHANGELOG, semver policy, PyPI trusted publishing via GitHub Actions.

---

## Phase 2 — MLOps & deployment (win ML engineers)

### 2.1 MLflow integration (`pip install genetic-xgb[mlflow]`)

`MLflowCallback` built on 1.1: logs params + manifest at start; per-generation best/mean fitness,
diversity, n_features_selected as step metrics (evolution curves live in the MLflow UI); final
model as a **custom pyfunc flavor** that preserves the feature mask (native xgboost flavor can't);
optional model-registry registration. Core stays tracker-agnostic — W&B later is just another
callback.

### 2.2 CLI — the bridge to docker and agents

```
genetic-xgb train  --config train.yaml --data train.parquet [--val val.parquet]
genetic-xgb report --run-dir runs/2026-07-02T.../
genetic-xgb predict --model model/ --data new.parquet --out preds.parquet
```

YAML config mirrors constructor args 1:1. `train` emits a **run directory**: model, manifest,
`report.json` (machine-readable: scores, curves, selected features, budget usage) and
`report.html` (human-readable evolution plots).

### 2.3 Docker compose — train + serve stack

`docker compose up` brings up:

- **mlflow** — tracking server + registry (sqlite + local volume first; postgres/minio as a
  documented variant, not the default).
- **trainer** — FastAPI job service: `POST /train` (dataset ref + config) → run id;
  `GET /runs/{id}` → live status fed by the callback event stream; jobs execute the CLI path.
- **server** — FastAPI serving container: loads a registered model (pyfunc), `POST /predict`,
  `GET /model/info` (features expected, training manifest).

Images published to GHCR by the release workflow. The compose file is the "full package
deployment": submit data, watch evolution in MLflow, get a live predict endpoint.

---

## Phase 3 — AI-ready (win agents as users)

### 3.1 MCP server (`genetic-xgb-mcp`)

stdio (local) and HTTP (compose stack) transports. Tools:

- `profile_dataset(path)` → shape, types, imbalance, leakage risks, suggested config
- `train_model(data, config)` → run id (background job via 2.3 trainer or local subprocess)
- `get_run_status(run_id)` / `get_run_report(run_id)` → live generation events / final report.json
- `predict(model_ref, data)` and `list_models()`

An agent can autonomously run profile → train → inspect report → adjust config → retrain.

### 3.2 Autopilot mode — autonomy for humans *and* agents

`auto_fit(X, y, time_budget=600)` (also `autopilot=True` on estimators): profiles the dataset,
chooses metric, search space, population size, imbalance handling and stopping policy, runs
within budget, returns the fitted estimator + report. One call, defensible defaults, full
manifest of what it decided and why.

### 3.3 Agent-native ergonomics

- **JSONL event stream** (`events_path=`): one line per generation event — anything (an agent, a
  dashboard, `tail -f`) can watch a run without importing the library.
- **Repo ships an AI skill + AGENTS.md:** a `train-model` skill guiding an agent through the
  profile→train→evaluate→iterate loop, and AGENTS.md documenting the CLI, report.json schema, and
  MCP tools.
- **LLM-friendly errors:** every raised exception states what happened, the likely cause, and the
  concrete fix (e.g. "val set has 3 samples; roc_auc needs both classes — pass a larger X_val or
  drop to metric='accuracy'").

---

## Explicitly out of scope (YAGNI)

Multi-backend boosters; Kubernetes operator; distributed training (Ray/Dask); preprocessing/
pipeline search (full AutoML); a custom web UI (MLflow is the UI); learning-to-rank objectives
(revisit after Phase 1 ships).

## Testing & quality bar

Every phase keeps the existing bar: 100% branch coverage, ruff, CI on PRs. New surfaces get their
own gates — CLI golden-file tests, compose smoke test in CI (train a toy dataset end-to-end
through the stack), MCP tool contract tests, and the benchmark suite run pinned-seed in a
scheduled (not per-PR) workflow.

## Sequencing summary

| Phase | Ships | Depends on |
|-------|-------|-----------|
| 1 | callbacks, checkpoints/budgets/validation, adaptive GA + islands + multi-objective, benchmarks, docs, PyPI | — |
| 2 | MLflow callback, CLI + reports, compose train+serve stack | 1.1, 1.2 |
| 3 | MCP server, autopilot, event stream, shipped skill/AGENTS.md | 2.2, 2.3 |
