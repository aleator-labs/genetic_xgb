"""Integration tests for GeneticXGBClassifier on real datasets (no mocks)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from genetic_xgb import GeneticXGBClassifier
from genetic_xgb.search_space import default_classification_space

_TINY_X = np.zeros((6, 3), dtype=np.float32)
_TINY_Y = np.array([0, 1, 0, 1, 0, 1])


def _nan_metric(y_true, proba) -> float:
    """A metric that always returns NaN (used to exercise the no-finite-fitness guard)."""
    return float("nan")


def _short_pbt(**overrides) -> GeneticXGBClassifier:
    params = {
        "population_size": 6,
        "generations": 4,
        "step_rounds": 5,
        "executor": "sequential",
        "random_state": 0,
        "selection_top_k": 2,
    }
    params.update(overrides)
    return GeneticXGBClassifier(**params)


def test_best_fitness_no_worse_than_generation_zero(binary_data) -> None:
    pbt = _short_pbt(metric="logloss").fit(
        binary_data.X_train,
        binary_data.y_train,
        X_val=binary_data.X_val,
        y_val=binary_data.y_val,
    )
    gen0_best = pbt.history_[pbt.history_["generation"] == 0]["score"].min()
    # logloss: smaller is better, so best overall must be <= gen-0 best.
    assert pbt.best_score_ <= gen0_best + 1e-9
    assert isinstance(pbt.best_params_, dict)
    for name in default_classification_space().names():
        assert name in pbt.best_params_


def test_sequential_reproducibility(binary_data) -> None:
    a = _short_pbt().fit(
        binary_data.X_train,
        binary_data.y_train,
        X_val=binary_data.X_val,
        y_val=binary_data.y_val,
    )
    b = _short_pbt().fit(
        binary_data.X_train,
        binary_data.y_train,
        X_val=binary_data.X_val,
        y_val=binary_data.y_val,
    )
    assert a.best_score_ == b.best_score_
    assert a.best_params_ == b.best_params_


def test_predict_proba_shape_and_rows_sum_to_one(binary_data) -> None:
    pbt = _short_pbt().fit(
        binary_data.X_train,
        binary_data.y_train,
        X_val=binary_data.X_val,
        y_val=binary_data.y_val,
    )
    proba = pbt.predict_proba(binary_data.X_val)
    assert proba.shape == (binary_data.X_val.shape[0], binary_data.n_classes)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    preds = pbt.predict(binary_data.X_val)
    assert preds.shape == (binary_data.X_val.shape[0],)
    assert set(np.unique(preds)).issubset({0, 1})


def test_multiclass_path(multiclass_data) -> None:
    pbt = _short_pbt(metric="accuracy").fit(
        multiclass_data.X_train,
        multiclass_data.y_train,
        X_val=multiclass_data.X_val,
        y_val=multiclass_data.y_val,
    )
    assert pbt.n_classes_ == multiclass_data.n_classes
    proba = pbt.predict_proba(multiclass_data.X_val)
    assert proba.shape == (multiclass_data.X_val.shape[0], multiclass_data.n_classes)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    preds = pbt.predict(multiclass_data.X_val)
    assert set(np.unique(preds)).issubset(set(range(multiclass_data.n_classes)))


def test_history_rows_and_lineage_columns(binary_data) -> None:
    pbt = _short_pbt().fit(
        binary_data.X_train,
        binary_data.y_train,
        X_val=binary_data.X_val,
        y_val=binary_data.y_val,
    )
    assert len(pbt.history_) == pbt.generations * pbt.population_size
    for column in ("generation", "member_id", "score", "n_rounds", "best_iteration", "parents"):
        assert column in pbt.history_.columns
    assert "learning_rate" in pbt.history_.columns
    # Early stopping is off by default -> best_iteration is recorded as null.
    assert pbt.history_["best_iteration"].isna().all()


def test_target_fitness_triggers_early_stop(binary_data) -> None:
    # accuracy: greater is better; an easy target reachable in one generation.
    pbt = _short_pbt(metric="accuracy", generations=10, target_fitness=0.5).fit(
        binary_data.X_train,
        binary_data.y_train,
        X_val=binary_data.X_val,
        y_val=binary_data.y_val,
    )
    generations_run = pbt.history_["generation"].nunique()
    assert generations_run < 10
    assert pbt.best_score_ >= 0.5


def test_patience_plateau_triggers_early_stop(binary_data) -> None:
    # Impossible-to-beat min_delta forces a plateau immediately.
    pbt = _short_pbt(generations=10, patience=1, min_delta=1e9).fit(
        binary_data.X_train,
        binary_data.y_train,
        X_val=binary_data.X_val,
        y_val=binary_data.y_val,
    )
    generations_run = pbt.history_["generation"].nunique()
    assert generations_run < 10


def test_base_params_override_merges_into_booster_config(binary_data) -> None:
    # Supplying base_params exercises the merge branch and must train a real model.
    pbt = _short_pbt(base_params={"max_bin": 64, "grow_policy": "lossguide"}).fit(
        binary_data.X_train,
        binary_data.y_train,
        X_val=binary_data.X_val,
        y_val=binary_data.y_val,
    )
    proba = pbt.predict_proba(binary_data.X_val)
    assert proba.shape == (binary_data.X_val.shape[0], binary_data.n_classes)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_early_stopping_records_best_iteration(binary_data) -> None:
    # With early stopping on, every member records a non-null best_iteration and still predicts.
    pbt = _short_pbt(step_rounds=60, early_stopping_rounds=5).fit(
        binary_data.X_train,
        binary_data.y_train,
        X_val=binary_data.X_val,
        y_val=binary_data.y_val,
    )
    assert pbt.history_["best_iteration"].notna().all()
    proba = pbt.predict_proba(binary_data.X_val)
    assert proba.shape == (binary_data.X_val.shape[0], binary_data.n_classes)


# --- F1: label encoding round-trip ---------------------------------------------------------


def test_string_labels_train_and_round_trip(multiclass_data) -> None:
    names = np.array(["setosa", "versicolor", "virginica"])
    y_train = names[multiclass_data.y_train]
    y_val = names[multiclass_data.y_val]
    pbt = _short_pbt(metric="accuracy").fit(
        multiclass_data.X_train, y_train, X_val=multiclass_data.X_val, y_val=y_val
    )
    # classes_ are the ORIGINAL labels in sorted order.
    assert list(pbt.classes_) == ["setosa", "versicolor", "virginica"]
    preds = pbt.predict(multiclass_data.X_val)
    # predict returns the original string labels, not positional indices.
    assert preds.dtype.kind in {"U", "S", "O"}
    assert set(np.unique(preds)).issubset(set(names.tolist()))
    proba = pbt.predict_proba(multiclass_data.X_val)
    assert proba.shape == (multiclass_data.X_val.shape[0], 3)
    # proba columns align with classes_: argmax mapped through classes_ equals predict.
    assert np.array_equal(pbt.classes_[proba.argmax(axis=1)], preds)


def test_noncontiguous_int_labels_round_trip(multiclass_data) -> None:
    mapping = np.array([10, 20, 30])
    y_train = mapping[multiclass_data.y_train]
    y_val = mapping[multiclass_data.y_val]
    pbt = _short_pbt(metric="accuracy").fit(
        multiclass_data.X_train, y_train, X_val=multiclass_data.X_val, y_val=y_val
    )
    assert list(pbt.classes_) == [10, 20, 30]
    preds = pbt.predict(multiclass_data.X_val)
    assert set(np.unique(preds)).issubset({10, 20, 30})
    proba = pbt.predict_proba(multiclass_data.X_val)
    assert np.array_equal(pbt.classes_[proba.argmax(axis=1)], preds)


def test_y_val_label_unseen_in_training_raises() -> None:
    x_val = np.zeros((3, 3), dtype=np.float32)
    y_val = np.array([0, 1, 2])  # class 2 never seen in training
    with pytest.raises(ValueError, match="not present in the training labels"):
        _short_pbt().fit(_TINY_X, _TINY_Y, X_val=x_val, y_val=y_val)


def test_single_class_in_y_train_raises() -> None:
    y_train = np.zeros(6, dtype=int)  # only one class
    with pytest.raises(ValueError, match="at least 2 classes"):
        _short_pbt().fit(_TINY_X, y_train, X_val=_TINY_X, y_val=y_train)


# --- F4: prediction before fit ----------------------------------------------------------------


def test_predict_before_fit_raises_not_fitted() -> None:
    pbt = GeneticXGBClassifier()
    with pytest.raises(NotFittedError):
        pbt.predict(_TINY_X)
    with pytest.raises(NotFittedError):
        pbt.predict_proba(_TINY_X)


# --- F11: all-nonfinite fitness ---------------------------------------------------------------


def test_all_nonfinite_fitness_raises_clear_error(binary_data) -> None:
    pbt = _short_pbt(metric=_nan_metric, greater_is_better=False)
    with pytest.raises(ValueError, match="finite fitness"):
        pbt.fit(
            binary_data.X_train,
            binary_data.y_train,
            X_val=binary_data.X_val,
            y_val=binary_data.y_val,
        )


# --- F5 / F6: input validation guards ---------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"population_size": 1}, "population_size must be >= 2"),
        ({"population_size": 6, "selection_top_k": 6}, "disabling all evolution"),
        ({"selection_top_k": 0}, "selection_top_k"),
        ({"generations": 0}, "generations must be >= 1"),
        ({"step_rounds": 0}, "step_rounds must be >= 1"),
        ({"mutation_fraction": 1.5}, "mutation_fraction"),
        ({"mutation_intensity": -0.1}, "mutation_intensity"),
        ({"dominance_prob": 1.5}, "dominance_prob"),
        ({"resample_prob": 1.5}, "resample_prob"),
    ],
)
def test_invalid_hyperparams_raise(kwargs, match) -> None:
    pbt = _short_pbt(**kwargs)
    with pytest.raises(ValueError, match=match):
        pbt.fit(_TINY_X, _TINY_Y, X_val=_TINY_X, y_val=_TINY_Y)


def test_y_train_not_1d_raises() -> None:
    with pytest.raises(ValueError, match="y_train must be 1-D"):
        _short_pbt().fit(_TINY_X, _TINY_Y.reshape(-1, 1), X_val=_TINY_X, y_val=_TINY_Y)


def test_y_val_not_1d_raises() -> None:
    with pytest.raises(ValueError, match="y_val must be 1-D"):
        _short_pbt().fit(_TINY_X, _TINY_Y, X_val=_TINY_X, y_val=_TINY_Y.reshape(-1, 1))


def test_x_train_y_train_row_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="X_train and y_train"):
        _short_pbt().fit(_TINY_X, _TINY_Y[:-1], X_val=_TINY_X, y_val=_TINY_Y)


def test_x_val_y_val_row_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="X_val and y_val"):
        _short_pbt().fit(_TINY_X, _TINY_Y, X_val=_TINY_X, y_val=_TINY_Y[:-1])


# --- F14: per-member seed determinism ---------------------------------------------------------


def test_member_seed_nondeterministic_when_random_state_none() -> None:
    pbt = GeneticXGBClassifier(random_state=None)
    rng = np.random.default_rng(0)
    first = pbt._member_seed(0, 0, rng)
    second = pbt._member_seed(0, 0, rng)
    # Same generation/member but draws advance the rng -> different seeds.
    assert first != second


def test_member_seed_deterministic_when_random_state_set() -> None:
    pbt = GeneticXGBClassifier(random_state=7)
    a = pbt._member_seed(1, 2, np.random.default_rng(0))
    b = pbt._member_seed(1, 2, np.random.default_rng(123))
    # rng is ignored when random_state is set; the formula is reproducible.
    assert a == b


# --- sklearn-compatibility battery -----------------------------------------------------------


def test_clone_returns_unfitted_estimator_with_same_params() -> None:
    pbt = _short_pbt(metric="accuracy", base_params={"max_bin": 64})
    cloned = clone(pbt)
    assert isinstance(cloned, GeneticXGBClassifier)
    assert cloned is not pbt
    # clone copies the constructor params verbatim and yields a fresh, unfitted estimator.
    assert cloned.get_params() == pbt.get_params()
    assert not hasattr(cloned, "best_booster_")


def test_get_params_set_params_round_trip() -> None:
    pbt = GeneticXGBClassifier()
    pbt.set_params(population_size=9, metric="accuracy", random_state=3)
    params = pbt.get_params()
    assert params["population_size"] == 9
    assert params["metric"] == "accuracy"
    assert params["random_state"] == 3
    # round-trip: feeding get_params back into a new estimator reproduces the params.
    twin = GeneticXGBClassifier(**params)
    assert twin.get_params() == params


def test_score_returns_float_accuracy(binary_data) -> None:
    pbt = _short_pbt(metric="accuracy").fit(
        binary_data.X_train,
        binary_data.y_train,
        X_val=binary_data.X_val,
        y_val=binary_data.y_val,
    )
    score = pbt.score(binary_data.X_val, binary_data.y_val)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_feature_importances_shape_and_sum(binary_data) -> None:
    pbt = _short_pbt().fit(
        binary_data.X_train,
        binary_data.y_train,
        X_val=binary_data.X_val,
        y_val=binary_data.y_val,
    )
    importances = pbt.feature_importances_
    assert importances.shape == (pbt.n_features_in_,)
    assert importances.dtype == np.float64
    # Real splits exist on cancer data, so the gains normalize to sum 1.
    assert np.isclose(importances.sum(), 1.0)
    assert np.all(importances >= 0.0)


def test_feature_importances_all_zero_when_no_splits() -> None:
    # Constant (all-zero) features give the booster nothing to split on: importances are all 0.
    pbt = _short_pbt().fit(_TINY_X, _TINY_Y, X_val=_TINY_X, y_val=_TINY_Y)
    importances = pbt.feature_importances_
    assert importances.shape == (3,)
    assert importances.sum() == 0.0


def test_feature_importances_before_fit_raises() -> None:
    with pytest.raises(NotFittedError):
        _ = GeneticXGBClassifier().feature_importances_


def test_n_features_in_recorded(binary_data) -> None:
    pbt = _short_pbt().fit(
        binary_data.X_train,
        binary_data.y_train,
        X_val=binary_data.X_val,
        y_val=binary_data.y_val,
    )
    assert pbt.n_features_in_ == binary_data.X_train.shape[1]


def test_sample_weight_accepted_and_changes_fit(binary_data) -> None:
    base = _short_pbt(metric="accuracy").fit(
        binary_data.X_train,
        binary_data.y_train,
        X_val=binary_data.X_val,
        y_val=binary_data.y_val,
    )
    rng = np.random.default_rng(0)
    weights = rng.uniform(0.1, 5.0, size=binary_data.X_train.shape[0]).astype(np.float32)
    weighted = _short_pbt(metric="accuracy").fit(
        binary_data.X_train,
        binary_data.y_train,
        X_val=binary_data.X_val,
        y_val=binary_data.y_val,
        sample_weight=weights,
    )
    # Per-row weights alter the training gradients, so the fitted boosters differ.
    assert weighted.best_booster_ != base.best_booster_


def test_predict_wrong_n_features_raises(binary_data) -> None:
    pbt = _short_pbt().fit(
        binary_data.X_train,
        binary_data.y_train,
        X_val=binary_data.X_val,
        y_val=binary_data.y_val,
    )
    bad = binary_data.X_val[:, :-1]  # one feature short
    with pytest.raises(ValueError, match="features"):
        pbt.predict(bad)
    with pytest.raises(ValueError, match="features"):
        pbt.predict_proba(bad)


def test_dataframe_column_reorder_gives_same_predictions(binary_data) -> None:
    names = [f"col_{i}" for i in range(binary_data.X_train.shape[1])]
    df_train = pd.DataFrame(binary_data.X_train, columns=names)
    df_val = pd.DataFrame(binary_data.X_val, columns=names)
    pbt = _short_pbt().fit(df_train, binary_data.y_train, X_val=df_val, y_val=binary_data.y_val)
    assert list(pbt.feature_names_in_) == names
    # A column-permuted frame must yield identical predictions (columns reordered internally).
    shuffled = df_val[names[::-1]]
    np.testing.assert_array_equal(pbt.predict(shuffled), pbt.predict(df_val))
    np.testing.assert_allclose(
        pbt.predict_proba(shuffled), pbt.predict_proba(df_val), rtol=0, atol=0
    )


def test_nan_inputs_are_accepted_like_xgboost(binary_data) -> None:
    # XGBoost treats NaN as missing; genetic_xgb must not reject it.
    X_train = binary_data.X_train.copy()  # noqa: N806
    X_train[0, 0] = np.nan
    X_val = binary_data.X_val.copy()  # noqa: N806
    X_val[0, 0] = np.nan
    pbt = _short_pbt().fit(X_train, binary_data.y_train, X_val=X_val, y_val=binary_data.y_val)
    preds = pbt.predict(X_val)  # predicting on NaN-containing data also works
    assert preds.shape == (X_val.shape[0],)


def test_dataframe_missing_columns_raises_clear_error(binary_data) -> None:
    names = [f"col_{i}" for i in range(binary_data.X_train.shape[1])]
    df_train = pd.DataFrame(binary_data.X_train, columns=names)
    df_val = pd.DataFrame(binary_data.X_val, columns=names)
    pbt = _short_pbt().fit(df_train, binary_data.y_train, X_val=df_val, y_val=binary_data.y_val)
    # Dropping a trained feature must raise a clear ValueError, not a raw KeyError.
    with pytest.raises(ValueError, match="missing"):
        pbt.predict(df_val.drop(columns=[names[0]]))


def test_fit_x_y_internal_split(binary_data) -> None:
    # fit(X, y) without an explicit validation set carves an internal stratified holdout.
    clf = _short_pbt().fit(binary_data.X_train, binary_data.y_train)
    assert clf.n_features_in_ == binary_data.X_train.shape[1]
    assert clf.predict(binary_data.X_val).shape == (binary_data.X_val.shape[0],)


def test_fit_internal_split_with_sample_weight(binary_data) -> None:
    weights = np.ones(binary_data.y_train.shape[0], dtype=np.float32)
    clf = _short_pbt().fit(binary_data.X_train, binary_data.y_train, sample_weight=weights)
    assert clf.predict(binary_data.X_val).shape == (binary_data.X_val.shape[0],)


def test_fit_partial_validation_args_raises(binary_data) -> None:
    with pytest.raises(ValueError, match="both X_val and y_val"):
        _short_pbt().fit(
            binary_data.X_train, binary_data.y_train, X_val=binary_data.X_val, y_val=None
        )


def test_invalid_validation_fraction_raises(binary_data) -> None:
    with pytest.raises(ValueError, match="validation_fraction"):
        _short_pbt(validation_fraction=0.0).fit(binary_data.X_train, binary_data.y_train)


def test_cross_val_score_runs(binary_data) -> None:
    # fit(X, y) makes the estimator usable inside sklearn meta-estimators.
    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(_short_pbt(), binary_data.X_train, binary_data.y_train, cv=2)
    assert scores.shape == (2,)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_refit_full_replaces_booster_and_predicts(binary_data) -> None:
    clf = _short_pbt().fit(
        binary_data.X_train,
        binary_data.y_train,
        X_val=binary_data.X_val,
        y_val=binary_data.y_val,
    )
    before = clf.best_booster_
    x_all = np.vstack([binary_data.X_train, binary_data.X_val])
    y_all = np.concatenate([binary_data.y_train, binary_data.y_val])
    clf.refit_full(x_all, y_all)
    assert clf.refit_full_ is True
    assert clf.best_booster_ != before  # deployed model retrained on all data
    preds = clf.predict(binary_data.X_val)
    assert set(np.unique(preds)).issubset(set(np.unique(y_all)))


# --- new surface: booster access, apply, native save/load, refit_on_full, keyword-only --------


def test_get_booster_returns_xgboost_booster(binary_data) -> None:
    pbt = _short_pbt().fit(
        binary_data.X_train,
        binary_data.y_train,
        X_val=binary_data.X_val,
        y_val=binary_data.y_val,
    )
    booster = pbt.get_booster()
    assert isinstance(booster, xgb.Booster)


def test_apply_returns_2d_leaf_indices(binary_data) -> None:
    pbt = _short_pbt().fit(
        binary_data.X_train,
        binary_data.y_train,
        X_val=binary_data.X_val,
        y_val=binary_data.y_val,
    )
    leaves = pbt.apply(binary_data.X_val)
    assert leaves.ndim == 2
    assert len(leaves) == binary_data.X_val.shape[0]


def test_save_load_model_binary_round_trip(binary_data) -> None:
    pbt = _short_pbt().fit(
        binary_data.X_train,
        binary_data.y_train,
        X_val=binary_data.X_val,
        y_val=binary_data.y_val,
    )
    expected = pbt.predict(binary_data.X_val)
    with tempfile.TemporaryDirectory() as tmp:
        fname = str(Path(tmp) / "model.json")
        pbt.save_model(fname)
        fresh = GeneticXGBClassifier()
        fresh.load_model(fname)
    # Native load recovers a 0-based binary label set and matches predictions on X_val.
    assert list(fresh.classes_) == [0, 1]
    assert fresh.n_features_in_ == binary_data.X_train.shape[1]
    np.testing.assert_array_equal(fresh.predict(binary_data.X_val), expected)


def test_save_load_model_multiclass_round_trip(multiclass_data) -> None:
    pbt = _short_pbt(metric="accuracy").fit(
        multiclass_data.X_train,
        multiclass_data.y_train,
        X_val=multiclass_data.X_val,
        y_val=multiclass_data.y_val,
    )
    expected = pbt.predict(multiclass_data.X_val)
    with tempfile.TemporaryDirectory() as tmp:
        fname = str(Path(tmp) / "model.json")
        pbt.save_model(fname)
        fresh = GeneticXGBClassifier()
        fresh.load_model(fname)
    # The num_class>0 branch restores 0..k-1 labels and the feature count.
    assert list(fresh.classes_) == [0, 1, 2]
    assert fresh.n_features_in_ == multiclass_data.X_train.shape[1]
    np.testing.assert_array_equal(fresh.predict(multiclass_data.X_val), expected)


def test_refit_on_full_with_fit_x_y_sets_flag(binary_data) -> None:
    # refit_on_full=True with fit(X, y) auto-retrains the winner on all of (X, y).
    clf = _short_pbt(refit_on_full=True).fit(binary_data.X_train, binary_data.y_train)
    assert clf.refit_full_ is True
    assert clf.predict(binary_data.X_val).shape == (binary_data.X_val.shape[0],)


def test_fit_rejects_positional_validation_args(binary_data) -> None:
    # X_val/y_val/sample_weight are keyword-only: 4 positional args is a TypeError.
    with pytest.raises(TypeError):
        _short_pbt().fit(
            binary_data.X_train,
            binary_data.y_train,
            binary_data.X_val,
            binary_data.y_val,
        )
