"""Integration tests for PopulationBasedTraining on real datasets (no mocks)."""

from __future__ import annotations

import numpy as np

from pbt_xgb import PopulationBasedTraining
from pbt_xgb.search_space import default_classification_space


def _short_pbt(**overrides) -> PopulationBasedTraining:
    params = {
        "population_size": 6,
        "generations": 4,
        "step_rounds": 5,
        "executor": "sequential",
        "random_state": 0,
        "selection_top_k": 2,
    }
    params.update(overrides)
    return PopulationBasedTraining(**params)


def test_best_fitness_no_worse_than_generation_zero(binary_data) -> None:
    pbt = _short_pbt(metric="logloss").fit(
        binary_data.X_train,
        binary_data.y_train,
        binary_data.X_val,
        binary_data.y_val,
    )
    gen0_best = pbt.history_[pbt.history_["generation"] == 0]["score"].min()
    # logloss: smaller is better, so best overall must be <= gen-0 best.
    assert pbt.best_score_ <= gen0_best + 1e-9
    assert isinstance(pbt.best_params_, dict)
    for name in default_classification_space().names():
        assert name in pbt.best_params_


def test_sequential_reproducibility(binary_data) -> None:
    a = _short_pbt().fit(
        binary_data.X_train, binary_data.y_train, binary_data.X_val, binary_data.y_val
    )
    b = _short_pbt().fit(
        binary_data.X_train, binary_data.y_train, binary_data.X_val, binary_data.y_val
    )
    assert a.best_score_ == b.best_score_
    assert a.best_params_ == b.best_params_


def test_predict_proba_shape_and_rows_sum_to_one(binary_data) -> None:
    pbt = _short_pbt().fit(
        binary_data.X_train, binary_data.y_train, binary_data.X_val, binary_data.y_val
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
        multiclass_data.X_val,
        multiclass_data.y_val,
    )
    assert pbt.n_classes_ == multiclass_data.n_classes
    proba = pbt.predict_proba(multiclass_data.X_val)
    assert proba.shape == (multiclass_data.X_val.shape[0], multiclass_data.n_classes)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    preds = pbt.predict(multiclass_data.X_val)
    assert set(np.unique(preds)).issubset(set(range(multiclass_data.n_classes)))


def test_history_rows_and_lineage_columns(binary_data) -> None:
    pbt = _short_pbt().fit(
        binary_data.X_train, binary_data.y_train, binary_data.X_val, binary_data.y_val
    )
    assert len(pbt.history_) == pbt.generations * pbt.population_size
    for column in ("generation", "member_id", "score", "n_rounds", "parents"):
        assert column in pbt.history_.columns
    assert "learning_rate" in pbt.history_.columns


def test_target_fitness_triggers_early_stop(binary_data) -> None:
    # accuracy: greater is better; an easy target reachable in one generation.
    pbt = _short_pbt(metric="accuracy", generations=10, target_fitness=0.5).fit(
        binary_data.X_train, binary_data.y_train, binary_data.X_val, binary_data.y_val
    )
    generations_run = pbt.history_["generation"].nunique()
    assert generations_run < 10
    assert pbt.best_score_ >= 0.5


def test_patience_plateau_triggers_early_stop(binary_data) -> None:
    # Impossible-to-beat min_delta forces a plateau immediately.
    pbt = _short_pbt(generations=10, patience=1, min_delta=1e9).fit(
        binary_data.X_train, binary_data.y_train, binary_data.X_val, binary_data.y_val
    )
    generations_run = pbt.history_["generation"].nunique()
    assert generations_run < 10


def test_base_params_override_merges_into_booster_config(binary_data) -> None:
    # Supplying base_params exercises the merge branch and must train a real model.
    pbt = _short_pbt(base_params={"max_bin": 64, "grow_policy": "lossguide"}).fit(
        binary_data.X_train, binary_data.y_train, binary_data.X_val, binary_data.y_val
    )
    proba = pbt.predict_proba(binary_data.X_val)
    assert proba.shape == (binary_data.X_val.shape[0], binary_data.n_classes)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
