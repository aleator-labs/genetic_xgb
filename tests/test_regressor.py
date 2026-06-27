"""Integration tests for GeneticXGBRegressor on real data (load_diabetes, no mocks)."""

from __future__ import annotations

import numpy as np

from genetic_xgb import GeneticXGBRegressor
from genetic_xgb.search_space import default_regression_space


def _short_reg(**overrides) -> GeneticXGBRegressor:
    params = {
        "population_size": 6,
        "generations": 4,
        "step_rounds": 5,
        "executor": "sequential",
        "random_state": 0,
        "selection_top_k": 2,
    }
    params.update(overrides)
    return GeneticXGBRegressor(**params)


def test_default_objective_and_metric(regression_data) -> None:
    reg = _short_reg().fit(
        regression_data.X_train,
        regression_data.y_train,
        regression_data.X_val,
        regression_data.y_val,
    )
    # Default regression objective and rmse fitness (smaller is better).
    assert reg.base_params_["objective"] == "reg:squarederror"
    gen0_best = reg.history_[reg.history_["generation"] == 0]["score"].min()
    assert reg.best_score_ <= gen0_best + 1e-9
    for name in default_regression_space().names():
        assert name in reg.best_params_


def test_predict_returns_continuous_vector(regression_data) -> None:
    reg = _short_reg().fit(
        regression_data.X_train,
        regression_data.y_train,
        regression_data.X_val,
        regression_data.y_val,
    )
    preds = reg.predict(regression_data.X_val)
    assert preds.shape == (regression_data.X_val.shape[0],)
    assert preds.dtype.kind == "f"
    # Continuous regression output, not class labels.
    assert np.unique(preds).size > 2
    assert not hasattr(reg, "predict_proba")


def test_metric_override_mae_and_r2(regression_data) -> None:
    mae = _short_reg(metric="mae").fit(
        regression_data.X_train,
        regression_data.y_train,
        regression_data.X_val,
        regression_data.y_val,
    )
    assert mae.best_score_ >= 0.0
    # r2 exercises the greater-is-better direction in the regression registry.
    r2 = _short_reg(metric="r2").fit(
        regression_data.X_train,
        regression_data.y_train,
        regression_data.X_val,
        regression_data.y_val,
    )
    gen0_best_r2 = r2.history_[r2.history_["generation"] == 0]["score"].max()
    assert r2.best_score_ >= gen0_best_r2 - 1e-9


def test_sequential_reproducibility(regression_data) -> None:
    a = _short_reg().fit(
        regression_data.X_train,
        regression_data.y_train,
        regression_data.X_val,
        regression_data.y_val,
    )
    b = _short_reg().fit(
        regression_data.X_train,
        regression_data.y_train,
        regression_data.X_val,
        regression_data.y_val,
    )
    assert a.best_score_ == b.best_score_
    assert a.best_params_ == b.best_params_


def test_early_stopping_records_best_iteration(regression_data) -> None:
    reg = _short_reg(step_rounds=60, early_stopping_rounds=5).fit(
        regression_data.X_train,
        regression_data.y_train,
        regression_data.X_val,
        regression_data.y_val,
    )
    assert reg.history_["best_iteration"].notna().all()
    assert reg.predict(regression_data.X_val).shape == (regression_data.X_val.shape[0],)
