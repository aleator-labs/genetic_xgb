"""Metric registry and resolver for the genetic-algorithm estimators.

This module OWNS :class:`MetricSpec` plus the classification and regression metric
registries and the resolver. All scorer functions are module-level so they remain
picklable for joblib.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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
    """Describes a metric: its name, direction, proba needs, and scorer fn.

    ``needs_proba`` is *advisory* metadata only. It records whether the built-in
    metric is naturally probability-based (e.g. ``logloss``/``roc_auc``) versus
    label-based (e.g. ``accuracy``); it is **not** consulted by :meth:`score` and
    does not transform the inputs. Callers that branch on probability vs. label
    behaviour may read it, but nothing in this module relies on it.

    Custom callable metrics registered via :func:`resolve_metric` receive the raw
    model output unchanged: ``score()`` simply forwards the prediction array. For
    classification that is the probability array (1-D ``P(class=1)`` for binary or
    a 2-D ``(n, n_classes)`` matrix for multiclass); for regression it is the
    vector of continuous predictions. The callable is responsible for any label
    conversion it needs.
    """

    name: str
    greater_is_better: bool
    needs_proba: bool
    fn: Callable[..., Any]

    def score(self, y_true: np.ndarray, proba: np.ndarray) -> float:
        # ``proba`` is forwarded to ``fn`` unchanged (only coerced to ndarray);
        # see the class docstring for the contract on custom callables.
        return float(self.fn(y_true, np.asarray(proba)))


def _to_labels(proba: np.ndarray) -> np.ndarray:
    if proba.ndim == 1:
        return (proba >= 0.5).astype(int)
    return proba.argmax(axis=1)


def _logloss(y_true: np.ndarray, proba: np.ndarray) -> Any:
    # Derive the label set from the proba shape so a validation split that is
    # missing a class present in training (fewer classes in ``y_true`` than
    # columns in ``proba``) does not crash. sklearn would otherwise infer the
    # labels from ``y_true`` and reject the wider probability matrix.
    labels = range(proba.shape[1]) if proba.ndim == 2 else [0, 1]
    return log_loss(y_true, proba, labels=labels)


def _accuracy(y_true: np.ndarray, proba: np.ndarray) -> Any:
    return accuracy_score(y_true, _to_labels(proba))


def _roc_auc(y_true: np.ndarray, proba: np.ndarray) -> Any:
    if proba.ndim == 1:
        return roc_auc_score(y_true, proba)
    return roc_auc_score(y_true, proba, multi_class="ovr")


def _f1(y_true: np.ndarray, proba: np.ndarray) -> Any:
    average = "binary" if proba.ndim == 1 else "macro"
    return f1_score(y_true, _to_labels(proba), average=average)


def _average_precision(y_true: np.ndarray, proba: np.ndarray) -> Any:
    if proba.ndim == 1:
        return average_precision_score(y_true, proba)
    classes = np.arange(proba.shape[1])
    y_bin = label_binarize(y_true, classes=classes)
    return average_precision_score(y_bin, proba, average="macro")


def _rmse(y_true: np.ndarray, pred: np.ndarray) -> Any:
    return np.sqrt(mean_squared_error(y_true, pred))


def _mse(y_true: np.ndarray, pred: np.ndarray) -> Any:
    return mean_squared_error(y_true, pred)


def _mae(y_true: np.ndarray, pred: np.ndarray) -> Any:
    return mean_absolute_error(y_true, pred)


def _r2(y_true: np.ndarray, pred: np.ndarray) -> Any:
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


def resolve_metric(
    metric: str | Callable[..., Any],
    greater_is_better: bool | None = None,
    registry: dict[str, MetricSpec] | None = None,
) -> MetricSpec:
    """Resolve ``metric`` (a key in ``registry`` or a callable) to a :class:`MetricSpec`.

    ``registry`` defaults to the classification :data:`METRICS`; the regressor passes
    :data:`REGRESSION_METRICS`.

    A custom callable is wrapped in a :class:`MetricSpec` with ``needs_proba=False``
    (advisory only). Regardless of that flag, the callable is invoked with the raw
    model output: probabilities for classification, continuous predictions for
    regression. See :class:`MetricSpec` for the full contract.
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
