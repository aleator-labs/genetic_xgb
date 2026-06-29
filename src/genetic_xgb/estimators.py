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
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import _check_sample_weight, check_is_fitted, validate_data

from genetic_xgb.executor import make_executor
from genetic_xgb.history import History
from genetic_xgb.member import PopulationMember
from genetic_xgb.metrics import CLASSIFICATION_METRICS, REGRESSION_METRICS, resolve_metric
from genetic_xgb.search_space import default_classification_space, default_regression_space
from genetic_xgb.strategy import GeneticStrategy
from genetic_xgb.trainer import train_step

_SEED_MOD = 2**31 - 1


class BaseGeneticXGB(BaseEstimator):
    """Shared genetic-algorithm engine. Subclasses fill in the task-specific hooks.

    Fitted attributes (set by :meth:`fit`):

    ``best_booster_``
        Raw bytes of the winning XGBoost booster. This is the only object that
        reproduces the search result; deploy the model via :meth:`predict` /
        ``predict_proba``, which load and run this booster.
    ``best_score_``
        The validation-selection score of the winning member. It is an
        *in-search* score used to pick the survivor each generation and is
        therefore **optimistically biased**: it is not a held-out estimate of
        generalization. Evaluate the fitted model on a fresh test set for an
        unbiased number.
    ``best_params_``
        The winning genome (hyperparameters of the best member). Because the
        engine warm-starts boosters and chains hyperparameters across a lineage,
        re-training a fresh booster from ``best_params_`` alone will **not**
        reproduce ``best_booster_``. Use ``best_booster_`` (via ``predict``) to
        deploy; treat ``best_params_`` as a report of the winning genome only.
    """

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

    def _member_seed(self, generation: int, member_id: int, rng: np.random.Generator) -> int:
        """Per-member booster seed.

        With ``random_state`` set, a fixed, reproducible formula is used. With
        ``random_state is None``, the seed is drawn from the (itself
        nondeterministic) run ``rng`` so repeated runs differ.
        """
        if self.random_state is None:
            return int(rng.integers(0, _SEED_MOD))
        return (int(self.random_state) * 1_000_003 + generation * 1009 + member_id) % _SEED_MOD

    def _validate_hyperparams(self) -> None:
        """Reject degenerate configurations with clear messages (see F5/F6)."""
        if self.population_size < 2:
            raise ValueError(f"population_size must be >= 2; got {self.population_size}.")
        if not 1 <= self.selection_top_k < self.population_size:
            raise ValueError(
                "selection_top_k must satisfy 1 <= selection_top_k < population_size "
                f"(={self.population_size}); got {self.selection_top_k}. A selection_top_k "
                ">= population_size selects the whole population, disabling all evolution."
            )
        if self.generations < 1:
            raise ValueError(f"generations must be >= 1; got {self.generations}.")
        if self.step_rounds < 1:
            raise ValueError(f"step_rounds must be >= 1; got {self.step_rounds}.")
        if not 0.0 <= self.mutation_fraction <= 1.0:
            raise ValueError(f"mutation_fraction must be in [0, 1]; got {self.mutation_fraction}.")
        if self.mutation_intensity < 0:
            raise ValueError(f"mutation_intensity must be >= 0; got {self.mutation_intensity}.")
        if not 0.0 <= self.dominance_prob <= 1.0:
            raise ValueError(f"dominance_prob must be in [0, 1]; got {self.dominance_prob}.")
        if not 0.0 <= self.resample_prob <= 1.0:
            raise ValueError(f"resample_prob must be in [0, 1]; got {self.resample_prob}.")

    def _validate_fit_inputs(self, X_train, y_train, X_val, y_val) -> None:  # noqa: N803
        """Check X/y shape agreement and that targets are 1-D (see F5)."""
        if y_train.ndim != 1:
            raise ValueError(f"y_train must be 1-D; got {y_train.ndim} dimensions.")
        if y_val.ndim != 1:
            raise ValueError(f"y_val must be 1-D; got {y_val.ndim} dimensions.")
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                "X_train and y_train must have the same number of rows; "
                f"got {X_train.shape[0]} and {y_train.shape[0]}."
            )
        if X_val.shape[0] != y_val.shape[0]:
            raise ValueError(
                "X_val and y_val must have the same number of rows; "
                f"got {X_val.shape[0]} and {y_val.shape[0]}."
            )

    def _validate_X(self, X, *, reset: bool):  # noqa: N802, N803
        """Validate ``X`` via sklearn, coercing to float32.

        With ``reset=True`` (training) this records ``n_features_in_`` and, for a
        pandas ``DataFrame``, ``feature_names_in_``. With ``reset=False``
        (validation / prediction) the feature count is checked against the fitted
        value and, when a ``DataFrame`` is supplied to a model fitted on named
        columns, the columns are reordered to the training order so that a
        column-permuted frame yields identical predictions.
        """
        if not reset and hasattr(self, "feature_names_in_") and hasattr(X, "columns"):
            missing = [c for c in self.feature_names_in_ if c not in X.columns]
            if missing:
                raise ValueError(
                    f"X is missing {len(missing)} feature(s) seen at fit time, e.g. {missing[:5]}"
                )
            X = X[list(self.feature_names_in_)]  # noqa: N806
        return validate_data(self, X, dtype=np.float32, reset=reset)

    def _encode_targets(self, y_train, y_val):  # noqa: N803
        """Task-specific target preparation. Base engine passes targets through."""
        return y_train, y_val

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

    def fit(self, X_train, y_train, X_val, y_val, sample_weight=None):  # noqa: N803
        self._validate_hyperparams()
        X_train = self._validate_X(X_train, reset=True)  # noqa: N806
        X_val = self._validate_X(X_val, reset=False)  # noqa: N806
        y_train = np.asarray(y_train)
        y_val = np.asarray(y_val)
        self._validate_fit_inputs(X_train, y_train, X_val, y_val)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X_train, dtype=np.float32)
        y_train, y_val = self._encode_targets(y_train, y_val)

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
                    "seed": self._member_seed(generation, m.id, rng),
                    "early_stopping_rounds": self.early_stopping_rounds,
                    "eval_metric": self.eval_metric,
                    "sample_weight": sample_weight,
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
            gen_score = gen_best.score
            if np.isfinite(gen_score) and (
                best_score is None or self._is_better(gen_score, best_score, greater)
            ):
                best_score = gen_score
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

        if best_bytes is None:
            raise ValueError(
                "No population member achieved a finite fitness value across any generation "
                "(all scores were NaN or infinite). Check the metric and the training data."
            )

        self.best_score_ = best_score
        self.best_params_ = best_params
        self.best_booster_ = best_bytes
        self.best_member_ = best_member
        self.history_ = history.to_frame()
        self.base_params_ = base
        return self

    def _load_booster(self) -> xgb.Booster:
        booster = xgb.Booster()
        booster.load_model(bytearray(self.best_booster_))
        return booster

    def _raw_predict(self, X):  # noqa: N803
        check_is_fitted(self, "best_booster_")
        X = self._validate_X(X, reset=False)  # noqa: N806
        return self._load_booster().predict(xgb.DMatrix(X))

    @property
    def feature_importances_(self) -> np.ndarray:
        """Gain-based importances, one per input feature, normalized to sum 1.

        Features the winning booster never split on are zero-filled. Raises
        :class:`~sklearn.exceptions.NotFittedError` before :meth:`fit`.
        """
        check_is_fitted(self, "best_booster_")
        scores = self._load_booster().get_score(importance_type="gain")
        importances = np.zeros(self.n_features_in_, dtype=np.float64)
        for key, gain in scores.items():
            importances[int(key[1:])] = gain
        total = importances.sum()
        if total > 0:
            importances /= total
        return importances


class GeneticXGBClassifier(ClassifierMixin, BaseGeneticXGB):
    """Evolve a population of XGBoost classifiers via a genetic algorithm.

    Class labels are label-encoded internally: arbitrary integer or string labels
    are supported. ``classes_`` holds the original labels in sorted order;
    :meth:`predict` returns those original labels and ``predict_proba`` returns
    one column per class, ordered to match ``classes_``.

    See :class:`BaseGeneticXGB` for caveats on ``best_score_`` (an optimistically
    biased in-search score) and ``best_params_`` (the winning genome, which does
    not by itself reproduce ``best_booster_``).
    """

    _default_metric = "logloss"
    _metric_registry = CLASSIFICATION_METRICS

    def _encode_targets(self, y_train, y_val):  # noqa: N803
        """Label-encode targets so arbitrary int/string labels train correctly.

        Stores ``self.classes_`` (original labels, sorted) and returns the encoded
        ``y_train`` / ``y_val``. Raises ``ValueError`` if fewer than two classes are
        present in training, or if a validation label was never seen in training.
        """
        encoder = LabelEncoder()
        y_train_enc = encoder.fit_transform(y_train)
        self.classes_ = encoder.classes_
        if self.classes_.size < 2:
            raise ValueError(
                f"Classification requires at least 2 classes in y_train; got {self.classes_.size}."
            )
        unseen = set(np.unique(y_val).tolist()) - set(self.classes_.tolist())
        if unseen:
            raise ValueError(
                f"Validation labels {sorted(unseen)} are not present in the training labels "
                f"{self.classes_.tolist()}."
            )
        y_val_enc = encoder.transform(y_val)
        return y_train_enc, y_val_enc

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
        encoded = self.predict_proba(X).argmax(axis=1)
        return self.classes_[encoded]


class GeneticXGBRegressor(RegressorMixin, BaseGeneticXGB):
    """Evolve a population of XGBoost regressors via a genetic algorithm.

    Targets are used as-is (no label encoding). See :class:`BaseGeneticXGB` for
    caveats on ``best_score_`` (an optimistically biased in-search score) and
    ``best_params_`` (the winning genome, which does not by itself reproduce
    ``best_booster_``; deploy via :meth:`predict`).
    """

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
