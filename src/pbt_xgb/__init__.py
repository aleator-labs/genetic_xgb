"""Population-based training (genetic algorithm) for XGBoost classifiers."""

from __future__ import annotations

from pbt_xgb.history import History
from pbt_xgb.member import PopulationMember
from pbt_xgb.metrics import MetricSpec, resolve_metric
from pbt_xgb.pbt import PopulationBasedTraining
from pbt_xgb.search_space import (
    Hyperparameter,
    SearchSpace,
    default_classification_space,
)
from pbt_xgb.strategy import GeneticStrategy
from pbt_xgb.trainer import train_step

__all__ = [
    "GeneticStrategy",
    "History",
    "Hyperparameter",
    "MetricSpec",
    "PopulationBasedTraining",
    "PopulationMember",
    "SearchSpace",
    "default_classification_space",
    "resolve_metric",
    "train_step",
]
