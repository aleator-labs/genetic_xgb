"""Gene/search-space layer for population-based training of XGBoost classifiers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Hyperparameter:
    """A single tunable gene (float / int / categorical)."""

    name: str
    kind: str  # "float" | "int" | "categorical"
    low: float | None = None
    high: float | None = None
    log: bool = False
    choices: tuple | None = None

    def sample(self, rng) -> Any:
        if self.kind == "categorical":
            return self.choices[int(rng.integers(len(self.choices)))]
        if self.log:
            value = math.exp(rng.uniform(math.log(self.low), math.log(self.high)))
        else:
            value = rng.uniform(self.low, self.high)
        if self.kind == "int":
            return self.clip(value)
        return value

    def mutate(self, value, rng, intensity) -> Any:
        if self.kind == "categorical":
            return self.choices[int(rng.integers(len(self.choices)))]
        if self.log:
            value = value * math.exp(rng.normal(0, intensity))
        else:
            value = value + rng.normal(0, intensity * (self.high - self.low))
        return self.clip(value)

    def clip(self, value) -> Any:
        if self.kind == "categorical":
            return value if value in self.choices else self.choices[0]
        clipped = min(max(value, self.low), self.high)
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

    def sample(self, rng) -> dict[str, Any]:
        return {p.name: p.sample(rng) for p in self.params}

    def clip(self, params: dict) -> dict:
        return {name: self._by_name[name].clip(value) for name, value in params.items()}

    def mutate(self, params, rng, fraction, intensity, resample_prob) -> dict:
        names = self.names()
        k = round(fraction * len(params))
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


def default_classification_space(extended: bool = False, imbalance: bool = False) -> SearchSpace:
    """Build the default XGBoost classification search space."""
    params: list[Hyperparameter] = [
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
    if extended:
        params += [
            Hyperparameter("grow_policy", "categorical", choices=("depthwise", "lossguide")),
            Hyperparameter("max_leaves", "int", low=0, high=256),
            Hyperparameter("num_parallel_tree", "int", low=1, high=4),
        ]
    if imbalance:
        params.append(Hyperparameter("scale_pos_weight", "float", low=0.1, high=10, log=True))
    return SearchSpace(params)
