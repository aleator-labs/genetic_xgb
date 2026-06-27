from __future__ import annotations

import numpy as np
import pytest
import xgboost as xgb

from genetic_xgb.metrics import (
    METRICS,
    REGRESSION_METRICS,
    MetricSpec,
    _accuracy,
    _average_precision,
    _f1,
    _logloss,
    _mae,
    _mse,
    _r2,
    _rmse,
    _roc_auc,
    _to_labels,
    resolve_metric,
)


def _binary_proba(binary_data):
    """Train a tiny real xgboost model and return canonical 1D P(class=1)."""
    dtrain = xgb.DMatrix(binary_data.X_train, label=binary_data.y_train)
    booster = xgb.train(
        {"objective": "binary:logistic", "tree_method": "hist", "verbosity": 0},
        dtrain,
        num_boost_round=5,
    )
    proba = booster.predict(xgb.DMatrix(binary_data.X_val))
    assert proba.ndim == 1
    return binary_data.y_val, proba


def _multiclass_proba(multiclass_data):
    """Train a tiny real xgboost model and return canonical 2D (n, k) proba."""
    dtrain = xgb.DMatrix(multiclass_data.X_train, label=multiclass_data.y_train)
    booster = xgb.train(
        {
            "objective": "multi:softprob",
            "num_class": multiclass_data.n_classes,
            "tree_method": "hist",
            "verbosity": 0,
        },
        dtrain,
        num_boost_round=5,
    )
    proba = booster.predict(xgb.DMatrix(multiclass_data.X_val))
    assert proba.ndim == 2
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
    return multiclass_data.y_val, proba


# ----------------------- helper / scorer-level tests -----------------------


def test_to_labels_binary():
    proba = np.array([0.1, 0.6, 0.5, 0.49], dtype=np.float32)
    np.testing.assert_array_equal(_to_labels(proba), np.array([0, 1, 1, 0]))


def test_to_labels_multiclass():
    proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]], dtype=np.float32)
    np.testing.assert_array_equal(_to_labels(proba), np.array([0, 2]))


def test_scorers_binary(binary_data):
    y, p = _binary_proba(binary_data)
    assert _logloss(y, p) > 0
    assert 0.0 <= _accuracy(y, p) <= 1.0
    assert 0.0 <= _roc_auc(y, p) <= 1.0
    assert 0.0 <= _f1(y, p) <= 1.0
    assert 0.0 <= _average_precision(y, p) <= 1.0


def test_scorers_multiclass(multiclass_data):
    y, p = _multiclass_proba(multiclass_data)
    assert _logloss(y, p) > 0
    assert 0.0 <= _accuracy(y, p) <= 1.0
    assert 0.0 <= _roc_auc(y, p) <= 1.0
    assert 0.0 <= _f1(y, p) <= 1.0
    assert 0.0 <= _average_precision(y, p) <= 1.0


# ----------------------- registry tests -----------------------


def test_registry_keys():
    assert set(METRICS) == {
        "logloss",
        "accuracy",
        "roc_auc",
        "f1",
        "average_precision",
    }


@pytest.mark.parametrize(
    ("name", "gib", "needs_proba"),
    [
        ("logloss", False, True),
        ("accuracy", True, False),
        ("roc_auc", True, True),
        ("f1", True, False),
        ("average_precision", True, True),
    ],
)
def test_registry_specs(name, gib, needs_proba):
    spec = METRICS[name]
    assert isinstance(spec, MetricSpec)
    assert spec.name == name
    assert spec.greater_is_better is gib
    assert spec.needs_proba is needs_proba


def test_metricspec_score_returns_float(binary_data):
    y, p = _binary_proba(binary_data)
    val = METRICS["accuracy"].score(y, p)
    assert isinstance(val, float)


def test_metricspec_score_accepts_list_proba():
    # score() should np.asarray the proba arg, so a python list works.
    spec = METRICS["accuracy"]
    val = spec.score(np.array([0, 1, 1]), [0.2, 0.9, 0.8])
    assert val == 1.0


# ----------------------- resolve_metric tests -----------------------


def test_resolve_metric_str():
    spec = resolve_metric("roc_auc")
    assert spec is METRICS["roc_auc"]


def test_resolve_metric_unknown_key():
    with pytest.raises(ValueError) as exc:
        resolve_metric("nope")
    msg = str(exc.value)
    # error should list available keys
    for key in METRICS:
        assert key in msg


def test_resolve_metric_callable():
    def my_metric(y_true, proba):
        return 0.5

    spec = resolve_metric(my_metric, greater_is_better=True)
    assert isinstance(spec, MetricSpec)
    assert spec.name == "my_metric"
    assert spec.greater_is_better is True
    assert spec.needs_proba is False
    assert spec.fn is my_metric
    assert spec.score(np.array([0, 1]), np.array([0.1, 0.9])) == 0.5


def test_resolve_metric_callable_no_name():
    spec = resolve_metric(lambda y, p: 1.0, greater_is_better=False)
    # lambda __name__ is "<lambda>" which exists; use a nameless-ish object instead
    assert spec.greater_is_better is False
    assert spec.score(np.array([0]), np.array([0.0])) == 1.0


def test_resolve_metric_callable_object_without_name():
    class Callable:
        def __call__(self, y_true, proba):
            return 0.25

    obj = Callable()
    assert not hasattr(obj, "__name__")
    spec = resolve_metric(obj, greater_is_better=True)
    assert spec.name == "custom"
    assert spec.score(np.array([1]), np.array([0.9])) == 0.25


def test_resolve_metric_callable_missing_gib():
    with pytest.raises(ValueError):
        resolve_metric(lambda y, p: 0.0)


# ----------------------- regression metric tests -----------------------


def _regression_pred(regression_data):
    """Train a tiny real xgboost regressor and return continuous predictions."""
    dtrain = xgb.DMatrix(regression_data.X_train, label=regression_data.y_train)
    booster = xgb.train(
        {"objective": "reg:squarederror", "tree_method": "hist", "verbosity": 0},
        dtrain,
        num_boost_round=5,
    )
    pred = booster.predict(xgb.DMatrix(regression_data.X_val))
    return regression_data.y_val, pred


def test_regression_scorers(regression_data):
    y, pred = _regression_pred(regression_data)
    assert _mse(y, pred) > 0
    # rmse is the square root of mse.
    np.testing.assert_allclose(_rmse(y, pred), np.sqrt(_mse(y, pred)))
    assert _mae(y, pred) > 0
    assert _r2(y, pred) <= 1.0


def test_regression_registry_keys():
    assert set(REGRESSION_METRICS) == {"rmse", "mse", "mae", "r2"}
    assert REGRESSION_METRICS["rmse"].greater_is_better is False
    assert REGRESSION_METRICS["r2"].greater_is_better is True
    assert all(spec.needs_proba is False for spec in REGRESSION_METRICS.values())


def test_resolve_metric_with_regression_registry():
    spec = resolve_metric("rmse", registry=REGRESSION_METRICS)
    assert spec is REGRESSION_METRICS["rmse"]


def test_resolve_metric_unknown_key_in_regression_registry():
    with pytest.raises(ValueError) as exc:
        resolve_metric("logloss", registry=REGRESSION_METRICS)
    msg = str(exc.value)
    for key in REGRESSION_METRICS:
        assert key in msg
