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

## Development

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run pytest                       # 100% branch-coverage gate enforced
uv run jupyter lab examples/demo_classification.ipynb   # or examples/demo_regression.ipynb
# headless execution:
uv run jupyter nbconvert --to notebook --execute --inplace examples/demo_regression.ipynb
```
