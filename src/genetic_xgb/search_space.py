"""Gene/search-space layer for the genetic-algorithm XGBoost estimators."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np


@dataclass(frozen=True)
class Hyperparameter:
    """A single tunable gene (float / int / categorical)."""

    name: str
    kind: str  # "float" | "int" | "categorical"
    low: float | None = None
    high: float | None = None
    log: bool = False
    choices: tuple[Any, ...] | None = None

    @property
    def _choices(self) -> tuple[Any, ...]:
        """``choices``, narrowed — always set for categorical genes."""
        return cast("tuple[Any, ...]", self.choices)

    @property
    def _bounds(self) -> tuple[float, float]:
        """``(low, high)``, narrowed — always set for float/int genes."""
        return cast(float, self.low), cast(float, self.high)

    def sample(self, rng: np.random.Generator) -> Any:
        if self.kind == "categorical":
            choices = self._choices
            return choices[int(rng.integers(len(choices)))]
        low, high = self._bounds
        if self.log:
            value = math.exp(rng.uniform(math.log(low), math.log(high)))
        else:
            value = rng.uniform(low, high)
        if self.kind == "int":
            return self.clip(value)
        return value

    def mutate(self, value: Any, rng: np.random.Generator, intensity: float) -> Any:
        if self.kind == "categorical":
            choices = self._choices
            return choices[int(rng.integers(len(choices)))]
        low, high = self._bounds
        if self.log:
            value = value * math.exp(rng.normal(0, intensity))
        else:
            value = value + rng.normal(0, intensity * (high - low))
        return self.clip(value)

    def clip(self, value: Any) -> Any:
        if self.kind == "categorical":
            choices = self._choices
            return value if value in choices else choices[0]
        low, high = self._bounds
        clipped = min(max(value, low), high)
        if self.kind == "int":
            return int(round(clipped))
        return clipped


class SearchSpace:
    """An ordered collection of :class:`Hyperparameter` genes."""

    def __init__(self, params: list[Hyperparameter]) -> None:
        self.params = tuple(params)
        self._by_name = {p.name: p for p in self.params}

    def names(self) -> list[str]:
        return [p.name for p in self.params]

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        return {p.name: p.sample(rng) for p in self.params}

    def clip(self, params: dict[str, Any]) -> dict[str, Any]:
        return {name: self._by_name[name].clip(value) for name, value in params.items()}

    def mutate(
        self,
        params: dict[str, Any],
        rng: np.random.Generator,
        fraction: float,
        intensity: float,
        resample_prob: float,
    ) -> dict[str, Any]:
        names = self.names()
        k = round(fraction * len(params))
        # A positive fraction should always mutate at least one gene, even when
        # rounding would otherwise floor it to zero.
        if fraction > 0 and k == 0:
            k = 1
        chosen = set(rng.choice(names, size=k, replace=False))
        new = dict(params)
        for name in names:
            if name in chosen:
                hp = self._by_name[name]
                if rng.random() < resample_prob:
                    new[name] = hp.sample(rng)
                else:
                    new[name] = hp.mutate(params[name], rng, intensity)
        return self.clip(new)


def _core_genes() -> list[Hyperparameter]:
    """The 11 tree genes shared by classification and regression."""
    return [
        Hyperparameter("learning_rate", "float", low=1e-3, high=0.3, log=True),
        Hyperparameter("max_depth", "int", low=3, high=10),
        Hyperparameter("min_child_weight", "float", low=1, high=10, log=True),
        Hyperparameter("gamma", "float", low=0, high=5),
        Hyperparameter("subsample", "float", low=0.5, high=1),
        Hyperparameter("colsample_bytree", "float", low=0.5, high=1),
        Hyperparameter("colsample_bylevel", "float", low=0.5, high=1),
        Hyperparameter("colsample_bynode", "float", low=0.5, high=1),
        Hyperparameter("max_delta_step", "float", low=0, high=10),
        Hyperparameter("reg_alpha", "float", low=1e-8, high=1, log=True),
        Hyperparameter("reg_lambda", "float", low=1e-8, high=10, log=True),
    ]


def _extended_genes() -> list[Hyperparameter]:
    """Tree-growth shape genes added when ``extended=True``."""
    return [
        Hyperparameter("grow_policy", "categorical", choices=("depthwise", "lossguide")),
        Hyperparameter("max_leaves", "int", low=0, high=256),
        Hyperparameter("num_parallel_tree", "int", low=1, high=4),
    ]


def default_classification_space(extended: bool = False, imbalance: bool = False) -> SearchSpace:
    """Build the default XGBoost classification search space.

    When ``imbalance`` is True a ``scale_pos_weight`` gene is added. Note that
    ``scale_pos_weight`` is a *binary*-classification knob: it is effectively a
    no-op for multiclass (``multi:softprob`` / ``multi:softmax``) objectives, so
    enabling ``imbalance`` for a multiclass problem will tune a gene XGBoost ignores.
    """
    params = _core_genes()
    if extended:
        params += _extended_genes()
    if imbalance:
        # scale_pos_weight only affects binary objectives; ignored for multiclass.
        params.append(Hyperparameter("scale_pos_weight", "float", low=0.1, high=10, log=True))
    return SearchSpace(params)


def default_regression_space(extended: bool = False) -> SearchSpace:
    """Build the default XGBoost regression search space (no class-imbalance gene)."""
    params = _core_genes()
    if extended:
        params += _extended_genes()
    return SearchSpace(params)
