# genetic_xgb — Genetic-Algorithm Hyperparameter Optimization for XGBoost

A small library that optimizes **XGBoost classifier and regressor** hyperparameters by **evolving a
population of models with a genetic algorithm**, grounded in general evolutionary principles rather
than any single paper.

Each individual carries a **genome** (its hyperparameters) and a **phenotype** (its trained
booster). Every generation the population is grown a few boosting rounds (warm-start), scored on a
validation set (**fitness**), and evolved through the five canonical stages:

1. **Evaluate fitness** — train each member `step_rounds` more rounds; fitness = validation metric.
2. **Selection** — the top `selection_top_k` survive (elitism); the rest are replaced.
3. **Crossover** — offspring recombine genes from two parents; the fitter (**dominant**) parent
   contributes each gene with probability `dominance_prob`, and the child inherits the dominant
   parent's booster (warm-start continues the better model).
4. **Mutation** — a `mutation_fraction` of genes are perturbed with `mutation_intensity`.
5. **Stopping** — stop at `generations`, or early on `target_fitness` / `patience` plateau.

Models are trained **only** on the training set; fitness is computed **only** on the validation
set (enforced by tests). Population evaluation runs in parallel via joblib.

## Classification

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from genetic_xgb import GeneticXGBClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

clf = GeneticXGBClassifier(
    population_size=16,
    metric="roc_auc",          # logloss | accuracy | roc_auc | f1 | average_precision | callable
    selection_top_k=4,         # how many survive each generation
    dominance_prob=0.7,        # crossover: dominant-parent gene probability
    mutation_fraction=0.3,     # fraction of genes mutated
    mutation_intensity=0.2,    # mutation magnitude
    generations=20,            # max generations ("epochs")
    target_fitness=0.99,       # optional success criterion (early stop)
    patience=5,                # optional plateau early-stop
    step_rounds=10,            # boosting rounds added per generation
    n_jobs=-1, random_state=42,
)
clf.fit(X_train, y_train, X_val, y_val)

clf.best_score_              # best validation fitness
clf.best_params_            # winning hyperparameters
clf.predict_proba(X_val)    # (n_samples, n_classes)
clf.predict(X_val)          # class labels
clf.history_                # pandas DataFrame: full per-generation lineage
```

## Regression

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from genetic_xgb import GeneticXGBRegressor

X, y = load_diabetes(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)

reg = GeneticXGBRegressor(
    population_size=16,
    metric="rmse",             # rmse | mae | mse | r2 | callable
    generations=20,
    step_rounds=10,
    n_jobs=-1, random_state=42,
)
reg.fit(X_train, y_train, X_val, y_val)

reg.best_score_             # best validation RMSE
reg.predict(X_val)          # continuous predictions (no predict_proba)
```

Both estimators share the same genetic-algorithm core (`BaseGeneticXGB`); only the objective,
metric set, default search space, and prediction differ.

## Built-in early stopping

Set `early_stopping_rounds` to use XGBoost's native early stopping with the validation set as the
eval watchlist. Each member then stops adding trees once validation stops improving, so `step_rounds`
becomes a per-generation **upper bound** — set it comfortably larger than `early_stopping_rounds`.
The full (stopped) booster is kept; `history_['best_iteration']` records where each member stopped.

```python
reg = GeneticXGBRegressor(
    metric="rmse",
    step_rounds=60,            # upper bound per generation
    early_stopping_rounds=10,  # stop a member's growth on a 10-round plateau
    eval_metric="rmse",        # optional; defaults to the objective's native metric
)
```

Trees are always fit on the **training** gradients only; with early stopping the validation set
influences *when to stop adding trees* (model selection), never the gradients.

## Search spaces

```python
from genetic_xgb import default_classification_space, default_regression_space

# Classification: add tree-growth genes and a class-imbalance gene.
space = default_classification_space(extended=True, imbalance=True)

# Regression: same tree genes (no class-imbalance gene).
space = default_regression_space(extended=True)

clf = GeneticXGBClassifier(search_space=space, ...)
```

## scikit-learn compatibility

`GeneticXGBClassifier` and `GeneticXGBRegressor` are real scikit-learn estimators. They subclass
`sklearn.base.BaseEstimator` plus the matching mixin (`ClassifierMixin` for the classifier,
`RegressorMixin` for the regressor), so the standard estimator API works out of the box:

- **`get_params()` / `set_params()`** — all constructor arguments are plain stored attributes, so
  introspection and re-parameterization behave as sklearn expects.
- **`sklearn.base.clone()`** — produces an unfitted copy with the same hyperparameters.
- **`score(X, y, sample_weight=None)`** — inherited from the mixin: **accuracy** for the classifier,
  **R²** for the regressor. (Note this scores on whatever `(X, y)` you pass; it is independent of the
  GA fitness `metric`.)
- **`predict` / `predict_proba` shapes match XGBoost** — `predict` returns `(n_samples,)`;
  `predict_proba` (classifier only) returns `(n_samples, n_classes)` with columns ordered to match
  `classes_`.
- **Fitted attributes exposed** — `classes_` (classifier), `n_features_in_`, `feature_names_in_`
  (set when fitted from a DataFrame), and `feature_importances_` (read from `best_booster_`).
- **`sample_weight` is supported** in `fit`; it weights the **training** objective only (the
  validation fitness signal is left unweighted).
- **Input validation** — `predict` / `predict_proba` raise if the input feature count or feature
  names do not match what was seen at `fit` time; calling `predict` / `predict_proba` / `score`
  before fitting raises `sklearn.exceptions.NotFittedError`.

### The one deliberate deviation: `fit` requires an explicit validation set

Every other HPO estimator can hide the validation split inside `fit(X, y)`. This one cannot: the
genetic search needs a **held-out fitness signal** every generation, so the validation set is part of
the public `fit` signature:

```python
fit(X_train, y_train, X_val, y_val, sample_weight=None)
```

The direct consequence is that these estimators do **not** drop into the vanilla sklearn
meta-estimators that assume a two-argument `fit(X, y)`:

- `sklearn.pipeline.Pipeline`
- `sklearn.model_selection.GridSearchCV`
- `sklearn.model_selection.RandomizedSearchCV`
- `sklearn.model_selection.cross_val_score`

This is **by design**, not an oversight — those tools would have to invent or re-split a validation
set, which would defeat the point of an explicit, caller-controlled held-out signal. Everything that
operates on a *fitted* estimator (introspection, `clone`, `score`, `predict`) still works normally.

### `feature_importances_` + `sample_weight`

```python
import numpy as np
from genetic_xgb import GeneticXGBClassifier

# upweight a minority class without touching the search space
sample_weight = np.where(y_train == 1, 5.0, 1.0)

clf = GeneticXGBClassifier(metric="roc_auc", random_state=0)
clf.fit(X_train, y_train, X_val, y_val, sample_weight=sample_weight)

clf.feature_importances_     # (n_features_in_,), read from best_booster_
clf.n_features_in_           # int
clf.feature_names_in_        # present when X_train was a DataFrame
```

## Caveats / scope

- **`best_score_` is an in-search selection score, not a test estimate.** Like the `best_score_` of
  any HPO procedure (`GridSearchCV`, Optuna, ...), it is the best *validation* fitness reached
  during the search and is therefore optimistically biased — the search selected for it. For an
  unbiased estimate, evaluate the returned model on a **separate held-out test set** that was not
  used as `X_val`/`y_val`.
- **`best_params_` is the winning genome; it does not by itself reproduce `best_booster_`.** Because
  crossover warm-starts the dominant parent's booster, the trained model is the product of a whole
  warm-start lineage of hyperparameters, not a single `fit` with `best_params_`. To deploy the
  evolved model, call `predict` / `predict_proba` (which use `best_booster_`); treat `best_params_`
  as the final genome for inspection, not as a recipe to retrain `best_booster_` from scratch.
- **`early_stopping_rounds` without `eval_metric` stops on XGBoost's objective default metric.** That
  default may differ from the GA fitness `metric` (e.g. objective default `logloss` vs. fitness
  `roc_auc`), so members can stop on a metric you are not optimizing. Set `eval_metric` to align
  early stopping with the GA fitness metric.
- **`scale_pos_weight` only affects binary classification.** It is ignored for regression and has no
  defined meaning for multiclass objectives.
- **The full stopped booster is kept.** With early stopping, `best_iteration` is recorded in
  `history_` for insight only; the booster is **not** trimmed to `best_iteration`, so predictions
  use all trees grown up to the stopping point.

## Development

```bash
uv sync
git config core.hooksPath .githooks  # one-time: enable the pre-push quality gate
uv run ruff check . && uv run ruff format --check .
uv run pytest                       # 100% branch-coverage gate enforced
uv run jupyter lab examples/demo_classification.ipynb   # or examples/demo_regression.ipynb
# headless execution:
uv run jupyter nbconvert --to notebook --execute --inplace examples/demo_regression.ipynb
```

### Pre-push gate

`.githooks/pre-push` runs ruff (lint + format) and the test suite at 100% branch coverage, and
blocks the push if anything fails. Enable it once per clone with
`git config core.hooksPath .githooks` (shown above). Bypass in an emergency with
`git push --no-verify`.

### Multi-agent review (Claude Code)

`.claude/workflows/review-dropin.mjs` is a reusable review workflow: 5 independent reviewers →
aggregate/de-duplicate → one adversarial verifier per finding. Run `/review-dropin` in a Claude
Code session (or `Workflow({name: "review-dropin"})`).
