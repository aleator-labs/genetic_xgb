"""Genetic-algorithm estimators (classifier + regressor) for XGBoost.

:class:`BaseGeneticXGB` holds the task-agnostic genetic-algorithm engine; the two
subclasses supply only the task-specific hooks (objective, default metric and
registry, default search space, and prediction).
"""

from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import _check_sample_weight, check_is_fitted, validate_data

from genetic_xgb.executor import make_executor
from genetic_xgb.feature_selection import sample_mask
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
    _stratify_split: bool = True

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
        validation_fraction: float = 0.2,
        refit_on_full: bool = False,
        feature_selection: bool = False,
        feature_init_prob: float = 0.5,
        feature_mutation_rate: float = 0.1,
        min_features: int = 1,
        overfit_penalty: float = 0.0,
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
        self.validation_fraction = validation_fraction
        self.refit_on_full = refit_on_full
        self.feature_selection = feature_selection
        self.feature_init_prob = feature_init_prob
        self.feature_mutation_rate = feature_mutation_rate
        self.min_features = min_features
        self.overfit_penalty = overfit_penalty
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
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError(
                f"validation_fraction must be in (0, 1); got {self.validation_fraction}."
            )
        if not 0.0 < self.feature_init_prob <= 1.0:
            raise ValueError(f"feature_init_prob must be in (0, 1]; got {self.feature_init_prob}.")
        if not 0.0 <= self.feature_mutation_rate <= 1.0:
            raise ValueError(
                f"feature_mutation_rate must be in [0, 1]; got {self.feature_mutation_rate}."
            )
        if self.min_features < 1:
            raise ValueError(f"min_features must be >= 1; got {self.min_features}.")
        if self.overfit_penalty < 0:
            raise ValueError(f"overfit_penalty must be >= 0; got {self.overfit_penalty}.")

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
        # ensure_all_finite=False keeps XGBoost's native missing-value (NaN) handling.
        return validate_data(self, X, dtype=np.float32, ensure_all_finite=False, reset=reset)

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

    def _make_split(self, X, y, X_val, y_val, sample_weight):  # noqa: N803
        """Resolve the train/validation split used for the genetic fitness signal.

        When ``X_val``/``y_val`` are supplied they are used directly. When omitted,
        an internal holdout of size ``validation_fraction`` is carved from
        ``(X, y)`` (stratified for classification) so ``fit(X, y)`` works on its own.
        Returns ``(X_train, y_train, X_val, y_val, sample_weight_train)``.
        """
        if X_val is not None:
            return X, y, X_val, y_val, sample_weight
        y_arr = np.asarray(y)
        stratify = y_arr if self._stratify_split else None
        if sample_weight is None:
            X_tr, X_v, y_tr, y_v = train_test_split(  # noqa: N806
                X,
                y_arr,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=stratify,
            )
            return X_tr, y_tr, X_v, y_v, None
        X_tr, X_v, y_tr, y_v, sw_tr, _ = train_test_split(  # noqa: N806
            X,
            y_arr,
            np.asarray(sample_weight),
            test_size=self.validation_fraction,
            random_state=self.random_state,
            stratify=stratify,
        )
        return X_tr, y_tr, X_v, y_v, sw_tr

    def fit(self, X, y, *, X_val=None, y_val=None, sample_weight=None):  # noqa: N803
        self._validate_hyperparams()
        if (X_val is None) != (y_val is None):
            raise ValueError(
                "Provide both X_val and y_val, or neither (to use an internal validation split)."
            )
        used_internal_split = X_val is None
        full_sample_weight = sample_weight
        X_train, y_train, X_val, y_val, sample_weight = self._make_split(  # noqa: N806
            X, y, X_val, y_val, sample_weight
        )
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

        def _initial_mask():
            if not self.feature_selection:
                return None
            return sample_mask(self.n_features_in_, self.feature_init_prob, rng, self.min_features)

        members = [
            PopulationMember(id=i, hyperparams=space.sample(rng), feature_mask=_initial_mask())
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
            feature_selection=self.feature_selection,
            feature_mutation_rate=self.feature_mutation_rate,
            min_features=self.min_features,
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

        # Overfit-aware fitness (only when feature selection is on): penalize the train-vs-val
        # generalization gap so subsets that overfit are not selected.
        direction = 1.0 if greater else -1.0
        compute_train = self.feature_selection and self.overfit_penalty > 0

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
                    "feature_mask": m.feature_mask,
                    "compute_train_fitness": compute_train,
                }
                for m in members
            ]
            results = executor.map(train_step, args)
            for member, result in zip(members, results, strict=True):
                member.booster_bytes = result["booster_bytes"]
                member.n_rounds = result["n_rounds"]
                member.best_iteration = result["best_iteration"]
                val_fitness = result["fitness"]
                if compute_train:
                    train_fitness = result["train_fitness"]
                    member.val_score = val_fitness
                    member.train_score = train_fitness
                    # ``direction * (train - val) > 0`` means train scores better than val,
                    # i.e. overfitting; penalize it in the metric's native direction.
                    overfit = max(0.0, direction * (train_fitness - val_fitness))
                    member.score = val_fitness - direction * self.overfit_penalty * overfit
                else:
                    member.score = val_fitness

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
        self.feature_mask_ = best_member.feature_mask
        self.history_ = history.to_frame()
        self.base_params_ = base
        self.refit_full_ = False
        # When fit(X, y) used an internal holdout, optionally retrain the winner on ALL of
        # (X, y) so the deployed model is not left trained on only (1 - validation_fraction).
        if self.refit_on_full and used_internal_split:
            self.refit_full(X, y, sample_weight=full_sample_weight)
        return self

    def _load_booster(self) -> xgb.Booster:
        booster = xgb.Booster()
        booster.load_model(bytearray(self.best_booster_))
        return booster

    def _select_features(self, X):  # noqa: N803
        """Restrict validated ``X`` to the selected feature columns (no-op if disabled)."""
        mask = getattr(self, "feature_mask_", None)
        return X if mask is None else X[:, mask]

    def _raw_predict(self, X):  # noqa: N803
        check_is_fitted(self, "best_booster_")
        X = self._validate_X(X, reset=False)  # noqa: N806
        return self._load_booster().predict(xgb.DMatrix(self._select_features(X)))

    def get_support(self, indices: bool = False):
        """Return the selected-feature mask (or selected indices if ``indices=True``).

        Without feature selection, all features are selected.
        """
        check_is_fitted(self, "best_booster_")
        mask = self.feature_mask_
        if mask is None:
            mask = np.ones(self.n_features_in_, dtype=bool)
        return np.flatnonzero(mask) if indices else mask

    @property
    def feature_importances_(self) -> np.ndarray:
        """Gain-based importances, one per input feature, normalized to sum 1.

        Features the winning booster never split on are zero-filled. Raises
        :class:`~sklearn.exceptions.NotFittedError` before :meth:`fit`.
        """
        check_is_fitted(self, "best_booster_")
        scores = self._load_booster().get_score(importance_type="gain")
        importances = np.zeros(self.n_features_in_, dtype=np.float64)
        # Booster feature indices ``f{j}`` refer to the SELECTED columns; map them back
        # to original feature positions through the mask (excluded features stay 0).
        support = self.get_support(indices=True)
        for key, gain in scores.items():
            importances[support[int(key[1:])]] = gain
        total = importances.sum()
        if total > 0:
            importances /= total
        return importances

    def _encode_y_for_refit(self, y):
        """Prepare targets for :meth:`refit_full`. Base passes through (regression)."""
        return np.asarray(y)

    def refit_full(self, X, y, sample_weight=None):  # noqa: N803
        """Train one XGBoost model on all of ``(X, y)`` from ``best_params_`` and deploy it.

        This is the conventional "refit the winning configuration on all data" step
        (e.g. train + validation combined, after the search picked a winner). The
        result is a SINGLE-configuration model trained for the winning member's round
        count -- it is NOT the evolved warm-start lineage -- and it REPLACES
        ``best_booster_`` so ``predict`` / ``predict_proba`` use it. Sets
        ``refit_full_ = True``. ``best_score_`` / ``best_params_`` still describe the
        search and are unchanged.
        """
        check_is_fitted(self, "best_booster_")
        X = self._validate_X(X, reset=False)  # noqa: N806
        y_enc = self._encode_y_for_refit(y)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float32)
        params = {**self.base_params_, **self.best_params_}
        dtrain = xgb.DMatrix(self._select_features(X), label=y_enc, weight=sample_weight)
        booster = xgb.train(params, dtrain, num_boost_round=self.best_member_.n_rounds)
        self.best_booster_ = bytes(booster.save_raw())
        self.refit_full_ = True
        return self

    def get_booster(self) -> xgb.Booster:
        """Return the fitted XGBoost :class:`~xgboost.Booster` (the deployed model)."""
        check_is_fitted(self, "best_booster_")
        return self._load_booster()

    def apply(self, X):  # noqa: N803
        """Return the leaf index each sample falls into, per tree (XGBoost ``pred_leaf``)."""
        x_sel = self._select_features(self._validate_X(X, reset=False))
        return self._load_booster().predict(xgb.DMatrix(x_sel), pred_leaf=True)

    def save_model(self, fname) -> None:
        """Save the fitted booster in XGBoost's native format (JSON/UBJ by extension).

        This writes only the booster (portable to other XGBoost tooling/languages); the
        search artifacts (``best_params_``, ``history_``) and original class labels are not
        included. Use :mod:`pickle` / :func:`joblib.dump` to persist the whole estimator.

        Not supported when feature selection was used: the native booster carries only the
        selected columns and no feature mask, so a reload could not restrict inputs and would
        mispredict. Use pickle / joblib (which preserve ``feature_mask_``) instead.
        """
        check_is_fitted(self, "best_booster_")
        if getattr(self, "feature_mask_", None) is not None:
            raise NotImplementedError(
                "save_model is unsupported when feature_selection was used (the native format "
                "cannot carry the feature mask). Use pickle / joblib.dump to persist the estimator."
            )
        self.get_booster().save_model(fname)

    def _restore_after_load(self, booster: xgb.Booster) -> None:
        """Hook to restore task-specific fitted state after :meth:`load_model`."""

    def load_model(self, fname):
        """Load a native XGBoost booster into this estimator for prediction.

        Restores ``best_booster_`` and ``n_features_in_`` (and, for the classifier,
        ``classes_`` as ``0..k-1``). Native models do not carry original (string /
        non-0-based) labels; use :mod:`pickle` / :func:`joblib.load` for full fidelity.
        """
        booster = xgb.Booster()
        booster.load_model(fname)
        self.best_booster_ = bytes(booster.save_raw())
        self.n_features_in_ = booster.num_features()
        self._restore_after_load(booster)
        return self


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

    def _encode_y_for_refit(self, y):
        """Encode refit targets with the classes learned during the search."""
        encoder = LabelEncoder()
        encoder.classes_ = self.classes_
        return encoder.transform(np.asarray(y))

    def _restore_after_load(self, booster):
        """Recover class metadata from a natively loaded booster (labels become 0..k-1)."""
        config = json.loads(booster.save_config())
        num_class = int(config["learner"]["learner_model_param"]["num_class"])
        self.n_classes_ = num_class if num_class > 0 else 2
        self.classes_ = np.arange(self.n_classes_)

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
    _stratify_split = False

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
