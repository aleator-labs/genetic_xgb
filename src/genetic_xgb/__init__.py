"""Genetic-algorithm hyperparameter optimization for XGBoost classifiers and regressors."""

from __future__ import annotations

from genetic_xgb.estimators import (
    BaseGeneticXGB,
    GeneticXGBClassifier,
    GeneticXGBRegressor,
)
from genetic_xgb.history import History
from genetic_xgb.member import PopulationMember
from genetic_xgb.metrics import MetricSpec, resolve_metric
from genetic_xgb.search_space import (
    Hyperparameter,
    SearchSpace,
    default_classification_space,
    default_regression_space,
)
from genetic_xgb.strategy import GeneticStrategy
from genetic_xgb.trainer import train_step

__all__ = [
    "BaseGeneticXGB",
    "GeneticStrategy",
    "GeneticXGBClassifier",
    "GeneticXGBRegressor",
    "History",
    "Hyperparameter",
    "MetricSpec",
    "PopulationMember",
    "SearchSpace",
    "default_classification_space",
    "default_regression_space",
    "resolve_metric",
    "train_step",
]
