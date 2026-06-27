"""Genetic-algorithm estimators (classifier + regressor) for XGBoost.

:class:`BaseGeneticXGB` holds the task-agnostic genetic-algorithm engine; the two
subclasses supply only the task-specific hooks (objective, default metric and
registry, default search space, and prediction).
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import xgboost as xgb

from genetic_xgb.executor import make_executor
from genetic_xgb.history import History
from genetic_xgb.member import PopulationMember
from genetic_xgb.metrics import CLASSIFICATION_METRICS, REGRESSION_METRICS, resolve_metric
from genetic_xgb.search_space import default_classification_space, default_regression_space
from genetic_xgb.strategy import GeneticStrategy
from genetic_xgb.trainer import train_step

_SEED_MOD = 2**31 - 1


class BaseGeneticXGB:
    """Shared genetic-algorithm engine. Subclasses fill in the task-specific hooks."""

    _default_metric: str = "logloss"
    _metric_registry: dict = CLASSIFICATION_METRICS

    def __init__(
        self,
        population_size: int = 16,
        metric: Any = None,
        selection_top_k: int = 4,
        dominance_prob: float = 0.7,
        mutation_fraction: float = 0.3,
        mutation_intensity: float = 0.2,
        resample_prob: float = 0.1,
        generations: int = 20,
        target_fitness: float | None = None,
        patience: int | None = None,
        min_delta: float = 0.0,
        step_rounds: int = 10,
        early_stopping_rounds: int | None = None,
        eval_metric: str | None = None,
        search_space: Any = None,
        strategy: Any = None,
        n_jobs: int = -1,
        executor: str = "joblib",
        random_state: int | None = None,
        base_params: dict | None = None,
        greater_is_better: bool | None = None,
    ) -> None:
        self.population_size = population_size
        self.metric = metric
        self.selection_top_k = selection_top_k
        self.dominance_prob = dominance_prob
        self.mutation_fraction = mutation_fraction
        self.mutation_intensity = mutation_intensity
        self.resample_prob = resample_prob
        self.generations = generations
        self.target_fitness = target_fitness
        self.patience = patience
        self.min_delta = min_delta
        self.step_rounds = step_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.search_space = search_space
        self.strategy = strategy
        self.n_jobs = n_jobs
        self.executor = executor
        self.random_state = random_state
        self.base_params = base_params
        self.greater_is_better = greater_is_better

    def _member_seed(self, generation: int, member_id: int) -> int:
        base = 0 if self.random_state is None else int(self.random_state)
        return (base * 1_000_003 + generation * 1009 + member_id) % _SEED_MOD

    def _is_better(self, candidate: float, current: float, greater: bool) -> bool:
        return candidate > current if greater else candidate < current

    def _meets_target(self, best: float, greater: bool) -> bool:
        if self.target_fitness is None:
            return False
        return best >= self.target_fitness if greater else best <= self.target_fitness

    # --- task hooks (overridden by subclasses) ---
    def _make_base_params(self, y_train) -> dict:  # noqa: N803
        raise NotImplementedError

    def _default_space(self):
        raise NotImplementedError

    def fit(self, X_train, y_train, X_val, y_val):  # noqa: N803
        X_train = np.asarray(X_train, dtype=np.float32)  # noqa: N806
        X_val = np.asarray(X_val, dtype=np.float32)  # noqa: N806
        y_train = np.asarray(y_train)
        y_val = np.asarray(y_val)

        base = self._make_base_params(y_train)
        if self.base_params:
            base.update(self.base_params)

        metric_name = self._default_metric if self.metric is None else self.metric
        metric_spec = resolve_metric(metric_name, self.greater_is_better, self._metric_registry)
        greater = metric_spec.greater_is_better
        space = self.search_space or self._default_space()

        rng = np.random.default_rng(self.random_state)
        members = [
            PopulationMember(id=i, hyperparams=space.sample(rng))
            for i in range(self.population_size)
        ]

        strategy = self.strategy or GeneticStrategy(
            space=space,
            top_k=self.selection_top_k,
            dominance_prob=self.dominance_prob,
            mutation_fraction=self.mutation_fraction,
            mutation_intensity=self.mutation_intensity,
            resample_prob=self.resample_prob,
            greater_is_better=greater,
        )
        executor = make_executor(self.executor, self.n_jobs)

        history = History()
        best_score: float | None = None
        best_bytes: bytes | None = None
        best_params: dict | None = None
        best_member: PopulationMember | None = None

        patience_best: float | None = None
        no_improve = 0

        train = (X_train, y_train)
        val = (X_val, y_val)

        for generation in range(self.generations):
            args = [
                {
                    "booster_bytes": m.booster_bytes,
                    "hyperparams": m.hyperparams,
                    "train": train,
                    "val": val,
                    "step_rounds": self.step_rounds,
                    "metric": metric_spec,
                    "base_params": base,
                    "seed": self._member_seed(generation, m.id),
                    "early_stopping_rounds": self.early_stopping_rounds,
                    "eval_metric": self.eval_metric,
                }
                for m in members
            ]
            results = executor.map(train_step, args)
            for member, result in zip(members, results, strict=True):
                member.booster_bytes = result["booster_bytes"]
                member.score = result["fitness"]
                member.n_rounds = result["n_rounds"]
                member.best_iteration = result["best_iteration"]

            history.record(generation, members)

            gen_best = strategy.rank(members)[0]
            if best_score is None or self._is_better(gen_best.score, best_score, greater):
                best_score = gen_best.score
                best_bytes = gen_best.booster_bytes
                best_params = dict(gen_best.hyperparams)
                best_member = copy.deepcopy(gen_best)

            if patience_best is None:
                patience_best = gen_best.score
                no_improve = 0
            else:
                gain = gen_best.score - patience_best if greater else patience_best - gen_best.score
                if gain > self.min_delta:
                    patience_best = gen_best.score
                    no_improve = 0
                else:
                    no_improve += 1

            if self._meets_target(best_score, greater):
                break
            if self.patience is not None and no_improve >= self.patience:
                break
            if generation < self.generations - 1:
                members = strategy.evolve(members, rng)

        self.best_score_ = best_score
        self.best_params_ = best_params
        self.best_booster_ = best_bytes
        self.best_member_ = best_member
        self.history_ = history.to_frame()
        self.base_params_ = base
        return self

    def _raw_predict(self, X):  # noqa: N803
        X = np.asarray(X, dtype=np.float32)  # noqa: N806
        booster = xgb.Booster()
        booster.load_model(bytearray(self.best_booster_))
        return booster.predict(xgb.DMatrix(X))


class GeneticXGBClassifier(BaseGeneticXGB):
    """Evolve a population of XGBoost classifiers via a genetic algorithm."""

    _default_metric = "logloss"
    _metric_registry = CLASSIFICATION_METRICS

    def _make_base_params(self, y_train):  # noqa: N803
        n_classes = int(np.unique(y_train).size)
        self.n_classes_ = n_classes
        base = {"tree_method": "hist", "max_bin": 256, "verbosity": 0, "nthread": 1}
        if n_classes == 2:
            base["objective"] = "binary:logistic"
        else:
            base["objective"] = "multi:softprob"
            base["num_class"] = n_classes
        return base

    def _default_space(self):
        return default_classification_space()

    def predict_proba(self, X):  # noqa: N803
        proba = self._raw_predict(X)
        if proba.ndim == 1:
            return np.column_stack([1.0 - proba, proba])
        return proba

    def predict(self, X):  # noqa: N803
        return self.predict_proba(X).argmax(axis=1)


class GeneticXGBRegressor(BaseGeneticXGB):
    """Evolve a population of XGBoost regressors via a genetic algorithm."""

    _default_metric = "rmse"
    _metric_registry = REGRESSION_METRICS

    def _make_base_params(self, y_train):  # noqa: N803
        return {
            "tree_method": "hist",
            "max_bin": 256,
            "verbosity": 0,
            "nthread": 1,
            "objective": "reg:squarederror",
        }

    def _default_space(self):
        return default_regression_space()

    def predict(self, X):  # noqa: N803
        return self._raw_predict(X)
