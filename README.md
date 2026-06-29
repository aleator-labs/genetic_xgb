# genetic_xgb — Genetic-Algorithm Hyperparameter Optimization for XGBoost

A library that optimizes **XGBoost classifier and regressor** hyperparameters by **evolving a
population of models with a genetic algorithm**.

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
clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)  # X_val / y_val are keyword-only

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
reg.fit(X_train, y_train, X_val=X_val, y_val=y_val)  # X_val / y_val are keyword-only

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
- **`sample_weight` is supported** in `fit` (keyword-only); it weights the **training** objective
  only (the validation fitness signal is left unweighted).
- **Input validation** — `predict` / `predict_proba` raise if the input feature count or feature
  names do not match what was seen at `fit` time; calling `predict` / `predict_proba` / `score`
  before fitting raises `sklearn.exceptions.NotFittedError`.
- **Missing values (`NaN`) are passed through to XGBoost**, preserving its native missing-value
  handling (inputs are not rejected for containing `NaN`).
- **`fit(X, y)` works** (an internal validation split is used), so the estimators run inside
  `Pipeline`, `cross_val_score`, and `GridSearchCV`; an explicit
  `fit(X_train, y_train, X_val=X_val, y_val=y_val)` is also supported (`X_val` / `y_val` /
  `sample_weight` are **keyword-only**), and `refit_full(X, y)` retrains the winner on all data (see
  below).

### Fitting: `fit(X, y)` or an explicit validation set

The genetic search needs a **held-out fitness signal** every generation. You can supply it two ways:

```python
# A. Plain sklearn-style fit — an internal stratified holdout (validation_fraction, default 0.2)
#    is carved from (X, y) for the GA fitness signal.
clf.fit(X, y)

# B. Explicit, caller-controlled validation set (use your own split):
#    X_val / y_val (and sample_weight) are keyword-only.
clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)
```

Because `fit(X, y)` works, these estimators **drop into the standard sklearn meta-estimators**:

- `sklearn.pipeline.Pipeline`
- `sklearn.model_selection.cross_val_score` (each outer fold runs the full GA on its training part
  with its own inner split — a genuinely unbiased estimate of the search procedure)
- `sklearn.model_selection.GridSearchCV` / `RandomizedSearchCV` (tunes the GA's meta-knobs, e.g.
  `mutation_fraction`, `population_size` — i.e. nested HPO; usually you only need this if you want to
  search the search itself)

Control the internal split size with `validation_fraction`. Provide `X_val`/`y_val` when you want a
fixed, caller-controlled split (e.g. a time-based holdout) instead of the random internal one.

### Using all the data for the final model: `refit_full`

After the search, train **one** XGBoost model on all of your data from the winning genome and deploy
it:

```python
clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)  # search
clf.refit_full(X_all, y_all)                          # retrain best_params_ on all data
clf.predict(X_test)                                   # now uses the refit, all-data model
```

`refit_full` trains a **single-configuration** model (the winning `best_params_`, for the winning
member's round count) on everything you pass, replaces `best_booster_`, and sets `refit_full_ =
True`. Note this is *not* the evolved warm-start lineage (which can't be replayed on new data) — it
is the conventional "refit the best configuration on all data" step, like `GridSearchCV`'s `refit`.

#### Automatic refit: `refit_on_full=True`

When you call the plain `fit(X, y)` form (which carves out an internal `validation_fraction`
holdout for the GA fitness signal), the deployed `best_booster_` is left trained on only
`1 - validation_fraction` of your data. Construct the estimator with `refit_on_full=True` to have
`fit` **automatically** call `refit_full(X, y)` on all of `(X, y)` once the search finishes (it sets
`refit_full_ = True`):

```python
clf = GeneticXGBClassifier(refit_on_full=True, random_state=0)
clf.fit(X, y)              # internal split for the GA, then auto-refit the winner on all of (X, y)
clf.predict(X_test)        # uses the all-data model
```

`refit_on_full` only triggers for the internal-split `fit(X, y)` path; if you supply an explicit
`X_val` / `y_val`, no automatic refit happens (call `refit_full` yourself if you want it). Any
`sample_weight` you pass to `fit` is forwarded to the automatic refit.

### Native XGBoost API: `get_booster`, `apply`, `save_model` / `load_model`

The fitted estimator exposes the underlying booster and the usual native-XGBoost hooks:

```python
booster = clf.get_booster()       # the deployed xgb.Booster (the model predict/predict_proba use)
leaves = clf.apply(X)             # per-tree leaf index each sample lands in (XGBoost pred_leaf)

clf.save_model("model.json")      # native XGBoost format (.json / .ubj by extension)
clf2 = GeneticXGBClassifier().load_model("model.json")  # restore for prediction
clf2.predict(X_test)
```

`save_model` / `load_model` use **XGBoost's native format**, so the file is portable to other
XGBoost tooling and languages — but it carries **only the booster**. It does **not** preserve the
search artifacts (`best_params_`, `history_`, `best_member_`) or the original class labels: a
classifier loaded this way exposes `classes_` as `0..k-1` (and `n_features_in_` from the booster),
not your original/string labels. For **full fidelity**, persist the whole estimator with `pickle`
or `joblib`, which round-trip every fitted attribute:

```python
import joblib
joblib.dump(clf, "estimator.joblib")          # whole estimator: booster + labels + search artifacts
clf = joblib.load("estimator.joblib")
```

### `feature_importances_` + `sample_weight`

```python
import numpy as np
from genetic_xgb import GeneticXGBClassifier

# upweight a minority class without touching the search space
sample_weight = np.where(y_train == 1, 5.0, 1.0)

clf = GeneticXGBClassifier(metric="roc_auc", random_state=0)
# X_val / y_val / sample_weight are all keyword-only.
clf.fit(X_train, y_train, X_val=X_val, y_val=y_val, sample_weight=sample_weight)

clf.feature_importances_     # (n_features_in_,), read from best_booster_
clf.n_features_in_           # int
clf.feature_names_in_        # present when X_train was a DataFrame
```

## Not a drop-in `XGBClassifier` / `XGBRegressor` rename

These estimators expose a native-XGBoost-flavored surface (`get_booster`, `apply`, `save_model` /
`load_model`, `feature_importances_`), but they are a **genetic-search wrapper**, not `XGBClassifier`
/ `XGBRegressor` under a new name. The intended, deliberate differences:

- **The constructor does not accept XGBoost hyperparameter names.** There is no `n_estimators`,
  `max_depth`, `learning_rate`, `subsample`, etc. on the constructor, because the genetic algorithm
  **searches** those genes. Pin a value you don't want searched via `base_params` (merged into every
  member's booster params), and control which genes are searched and over what ranges via
  `search_space`.
- **One `fit()` trains many boosters.** A single `fit` runs roughly `population_size × generations`
  booster trainings (the whole population, every generation), so it is far more compute than one
  `XGBoost` fit — budget accordingly.
- **XGBoost-specific `fit` / `predict` kwargs are not accepted.** `fit` takes only
  `X, y, *, X_val, y_val, sample_weight` (no `eval_set` — supply the validation set via
  `X_val` / `y_val`), and `predict` / `predict_proba` do not accept `output_margin`,
  `iteration_range`, or `base_margin`.

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
Code session (or `Workflow({scriptPath: ".claude/workflows/review-dropin.mjs"})`).
