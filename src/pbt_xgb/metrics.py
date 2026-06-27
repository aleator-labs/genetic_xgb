"""Metric registry and resolver for population-based training.

This module OWNS :class:`MetricSpec` plus the metric registry and resolver.
All scorer functions are module-level so they remain picklable for joblib.
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


METRICS: dict[str, MetricSpec] = {
    "logloss": MetricSpec("logloss", greater_is_better=False, needs_proba=True, fn=_logloss),
    "accuracy": MetricSpec("accuracy", greater_is_better=True, needs_proba=False, fn=_accuracy),
    "roc_auc": MetricSpec("roc_auc", greater_is_better=True, needs_proba=True, fn=_roc_auc),
    "f1": MetricSpec("f1", greater_is_better=True, needs_proba=False, fn=_f1),
    "average_precision": MetricSpec(
        "average_precision", greater_is_better=True, needs_proba=True, fn=_average_precision
    ),
}


def resolve_metric(metric, greater_is_better=None) -> MetricSpec:
    """Resolve ``metric`` (a registry key or callable) to a :class:`MetricSpec`."""
    if isinstance(metric, str):
        try:
            return METRICS[metric]
        except KeyError:
            available = ", ".join(METRICS)
            raise ValueError(f"Unknown metric {metric!r}; available metrics: {available}") from None
    if greater_is_better is None:
        raise ValueError("greater_is_better must be provided when metric is a callable.")
    return MetricSpec(
        name=getattr(metric, "__name__", "custom"),
        greater_is_better=greater_is_better,
        needs_proba=True,
        fn=metric,
    )
