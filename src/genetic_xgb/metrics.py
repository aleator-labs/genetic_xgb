"""Metric registry and resolver for the genetic-algorithm estimators.

This module OWNS :class:`MetricSpec` plus the classification and regression metric
registries and the resolver. All scorer functions are module-level so they remain
picklable for joblib.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


@dataclass(frozen=True)
class MetricSpec:
    """Describes a metric: its name, direction, proba needs, and scorer fn."""

    name: str
    greater_is_better: bool
    needs_proba: bool
    fn: Callable

    def score(self, y_true, proba) -> float:
        return float(self.fn(y_true, np.asarray(proba)))


def _to_labels(proba):
    if proba.ndim == 1:
        return (proba >= 0.5).astype(int)
    return proba.argmax(axis=1)


def _logloss(y_true, proba):
    return log_loss(y_true, proba)


def _accuracy(y_true, proba):
    return accuracy_score(y_true, _to_labels(proba))


def _roc_auc(y_true, proba):
    if proba.ndim == 1:
        return roc_auc_score(y_true, proba)
    return roc_auc_score(y_true, proba, multi_class="ovr")


def _f1(y_true, proba):
    average = "binary" if proba.ndim == 1 else "macro"
    return f1_score(y_true, _to_labels(proba), average=average)


def _average_precision(y_true, proba):
    if proba.ndim == 1:
        return average_precision_score(y_true, proba)
    classes = np.arange(proba.shape[1])
    y_bin = label_binarize(y_true, classes=classes)
    return average_precision_score(y_bin, proba, average="macro")


def _rmse(y_true, pred):
    return np.sqrt(mean_squared_error(y_true, pred))


def _mse(y_true, pred):
    return mean_squared_error(y_true, pred)


def _mae(y_true, pred):
    return mean_absolute_error(y_true, pred)


def _r2(y_true, pred):
    return r2_score(y_true, pred)


METRICS: dict[str, MetricSpec] = {
    "logloss": MetricSpec("logloss", greater_is_better=False, needs_proba=True, fn=_logloss),
    "accuracy": MetricSpec("accuracy", greater_is_better=True, needs_proba=False, fn=_accuracy),
    "roc_auc": MetricSpec("roc_auc", greater_is_better=True, needs_proba=True, fn=_roc_auc),
    "f1": MetricSpec("f1", greater_is_better=True, needs_proba=False, fn=_f1),
    "average_precision": MetricSpec(
        "average_precision", greater_is_better=True, needs_proba=True, fn=_average_precision
    ),
}

REGRESSION_METRICS: dict[str, MetricSpec] = {
    "rmse": MetricSpec("rmse", greater_is_better=False, needs_proba=False, fn=_rmse),
    "mse": MetricSpec("mse", greater_is_better=False, needs_proba=False, fn=_mse),
    "mae": MetricSpec("mae", greater_is_better=False, needs_proba=False, fn=_mae),
    "r2": MetricSpec("r2", greater_is_better=True, needs_proba=False, fn=_r2),
}

# Readable alias; ``METRICS`` is kept as the default so existing call sites keep working.
CLASSIFICATION_METRICS = METRICS


def resolve_metric(metric, greater_is_better=None, registry=None) -> MetricSpec:
    """Resolve ``metric`` (a key in ``registry`` or a callable) to a :class:`MetricSpec`.

    ``registry`` defaults to the classification :data:`METRICS`; the regressor passes
    :data:`REGRESSION_METRICS`.
    """
    registry = METRICS if registry is None else registry
    if isinstance(metric, str):
        try:
            return registry[metric]
        except KeyError:
            available = ", ".join(registry)
            raise ValueError(f"Unknown metric {metric!r}; available metrics: {available}") from None
    if greater_is_better is None:
        raise ValueError("greater_is_better must be provided when metric is a callable.")
    return MetricSpec(
        name=getattr(metric, "__name__", "custom"),
        greater_is_better=greater_is_better,
        needs_proba=False,
        fn=metric,
    )
