"""Tests for the pure training step (real XGBoost, no mocks)."""

from __future__ import annotations

import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification

from genetic_xgb.metrics import resolve_metric
from genetic_xgb.trainer import train_step

BINARY_BASE = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "max_bin": 256,
    "verbosity": 0,
    "nthread": 1,
}
MULTI_BASE = {
    "objective": "multi:softprob",
    "num_class": 3,
    "tree_method": "hist",
    "max_bin": 256,
    "verbosity": 0,
    "nthread": 1,
}
HP = {"learning_rate": 0.1, "max_depth": 3}


def test_cold_start_produces_valid_booster(binary_data):
    out = train_step(
        booster_bytes=None,
        hyperparams=HP,
        train=(binary_data.X_train, binary_data.y_train),
        val=(binary_data.X_val, binary_data.y_val),
        step_rounds=10,
        metric=resolve_metric("logloss"),
        base_params=BINARY_BASE,
        seed=0,
    )
    assert set(out) == {"booster_bytes", "fitness", "n_rounds", "best_iteration"}
    assert isinstance(out["booster_bytes"], bytes) and len(out["booster_bytes"]) > 0
    assert isinstance(out["fitness"], float)
    assert out["n_rounds"] == 10
    # Early stopping off -> no best_iteration recorded.
    assert out["best_iteration"] is None
    # The returned bytes load into a usable booster.
    booster = xgb.Booster()
    booster.load_model(bytearray(out["booster_bytes"]))
    proba = booster.predict(xgb.DMatrix(binary_data.X_val))
    assert proba.shape == (binary_data.X_val.shape[0],)


def test_warm_start_increases_rounds_by_exactly_step_rounds(binary_data):
    metric = resolve_metric("logloss")
    train = (binary_data.X_train, binary_data.y_train)
    val = (binary_data.X_val, binary_data.y_val)
    first = train_step(
        booster_bytes=None,
        hyperparams=HP,
        train=train,
        val=val,
        step_rounds=7,
        metric=metric,
        base_params=BINARY_BASE,
        seed=0,
    )
    second = train_step(
        booster_bytes=first["booster_bytes"],
        hyperparams=HP,
        train=train,
        val=val,
        step_rounds=7,
        metric=metric,
        base_params=BINARY_BASE,
        seed=0,
    )
    assert first["n_rounds"] == 7
    assert second["n_rounds"] == first["n_rounds"] + 7


def test_fitness_equals_independent_recompute_and_differs_from_train(binary_data):
    metric = resolve_metric("logloss")
    train = (binary_data.X_train, binary_data.y_train)
    val = (binary_data.X_val, binary_data.y_val)
    out = train_step(
        booster_bytes=None,
        hyperparams=HP,
        train=train,
        val=val,
        step_rounds=15,
        metric=metric,
        base_params=BINARY_BASE,
        seed=0,
    )
    booster = xgb.Booster()
    booster.load_model(bytearray(out["booster_bytes"]))
    val_proba = booster.predict(xgb.DMatrix(binary_data.X_val))
    expected = metric.score(binary_data.y_val, val_proba)
    assert out["fitness"] == expected

    train_proba = booster.predict(xgb.DMatrix(binary_data.X_train))
    train_score = metric.score(binary_data.y_train, train_proba)
    # The val fitness is computed on val, which differs from the train score.
    assert out["fitness"] != train_score


def test_multiclass_returns_2d_proba_and_scores(multiclass_data):
    metric = resolve_metric("logloss")
    out = train_step(
        booster_bytes=None,
        hyperparams=HP,
        train=(multiclass_data.X_train, multiclass_data.y_train),
        val=(multiclass_data.X_val, multiclass_data.y_val),
        step_rounds=10,
        metric=metric,
        base_params=MULTI_BASE,
        seed=0,
    )
    booster = xgb.Booster()
    booster.load_model(bytearray(out["booster_bytes"]))
    proba = booster.predict(xgb.DMatrix(multiclass_data.X_val))
    assert proba.ndim == 2 and proba.shape[1] == 3
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
    assert out["fitness"] == metric.score(multiclass_data.y_val, proba)


def test_no_leakage_train_only_model_at_or_below_chance_on_adversarial_val():
    # Train signal and validation labels are deliberately anti-correlated.
    features, y = make_classification(
        n_samples=400,
        n_features=10,
        n_informative=6,
        n_redundant=0,
        random_state=7,
    )
    X = features.astype(np.float32)  # noqa: N806
    n = X.shape[0]
    half = n // 2
    X_train, y_train = X[:half], y[:half]  # noqa: N806
    # Validation features same distribution, but labels FLIPPED vs the signal.
    X_val = X[half:]  # noqa: N806
    y_val = 1 - y[half:]

    metric = resolve_metric("accuracy")
    out = train_step(
        booster_bytes=None,
        hyperparams={"learning_rate": 0.3, "max_depth": 4},
        train=(X_train, y_train),
        val=(X_val, y_val),
        step_rounds=50,
        metric=metric,
        base_params=BINARY_BASE,
        seed=0,
    )
    # A train-only model cannot exploit the reversed val labels: at/below chance.
    assert out["fitness"] <= 0.55


def test_early_stopping_caps_rounds_and_records_best_iteration(binary_data):
    # step_rounds is a large upper bound; validation plateaus so it stops well before it.
    out = train_step(
        booster_bytes=None,
        hyperparams={"learning_rate": 0.3, "max_depth": 3},
        train=(binary_data.X_train, binary_data.y_train),
        val=(binary_data.X_val, binary_data.y_val),
        step_rounds=200,
        metric=resolve_metric("logloss"),
        base_params=BINARY_BASE,
        seed=0,
        early_stopping_rounds=10,
    )
    assert out["n_rounds"] < 200  # early stopping capped the round count
    assert isinstance(out["best_iteration"], int)


def test_early_stopping_with_explicit_eval_metric(binary_data):
    # Passing eval_metric routes it into params; the run still produces a valid model.
    out = train_step(
        booster_bytes=None,
        hyperparams=HP,
        train=(binary_data.X_train, binary_data.y_train),
        val=(binary_data.X_val, binary_data.y_val),
        step_rounds=100,
        metric=resolve_metric("logloss"),
        base_params=BINARY_BASE,
        seed=0,
        early_stopping_rounds=10,
        eval_metric="auc",
    )
    assert out["best_iteration"] is not None
    assert out["n_rounds"] >= 1


def test_early_stopping_no_leakage_on_adversarial_val():
    # Same reversed-signal probe, now WITH early stopping: still at/below chance.
    features, y = make_classification(
        n_samples=400, n_features=10, n_informative=6, n_redundant=0, random_state=7
    )
    X = features.astype(np.float32)  # noqa: N806
    half = X.shape[0] // 2
    out = train_step(
        booster_bytes=None,
        hyperparams={"learning_rate": 0.3, "max_depth": 4},
        train=(X[:half], y[:half]),
        val=(X[half:], 1 - y[half:]),
        step_rounds=100,
        metric=resolve_metric("accuracy"),
        base_params=BINARY_BASE,
        seed=0,
        early_stopping_rounds=10,
    )
    assert out["fitness"] <= 0.55
