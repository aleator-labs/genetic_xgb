# pbt_xgb — Population-Based Training for XGBoost

A small library that optimizes **XGBoost classifier** hyperparameters by **evolving a population
of models with a genetic algorithm**, grounded in general evolutionary principles rather than any
single paper.

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

## Usage

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from pbt_xgb import PopulationBasedTraining

X, y = load_breast_cancer(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

pbt = PopulationBasedTraining(
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
pbt.fit(X_train, y_train, X_val, y_val)

pbt.best_score_              # best validation fitness
pbt.best_params_            # winning hyperparameters
pbt.predict_proba(X_val)    # (n_samples, n_classes)
pbt.history_                # pandas DataFrame: full per-generation lineage
```

Custom / wider search space:

```python
from pbt_xgb import default_classification_space
space = default_classification_space(extended=True, imbalance=True)
pbt = PopulationBasedTraining(search_space=space, ...)
```

## Development

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run pytest                       # 100% branch-coverage gate enforced
uv run jupyter lab examples/demo_classification.ipynb   # interactive demo
# or run it headless:
uv run jupyter nbconvert --to notebook --execute --inplace examples/demo_classification.ipynb
```
