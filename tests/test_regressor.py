"""Integration tests for GeneticXGBRegressor on real data (load_diabetes, no mocks)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

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
        X_val=regression_data.X_val,
        y_val=regression_data.y_val,
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
        X_val=regression_data.X_val,
        y_val=regression_data.y_val,
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
        X_val=regression_data.X_val,
        y_val=regression_data.y_val,
    )
    assert mae.best_score_ >= 0.0
    # r2 exercises the greater-is-better direction in the regression registry.
    r2 = _short_reg(metric="r2").fit(
        regression_data.X_train,
        regression_data.y_train,
        X_val=regression_data.X_val,
        y_val=regression_data.y_val,
    )
    gen0_best_r2 = r2.history_[r2.history_["generation"] == 0]["score"].max()
    assert r2.best_score_ >= gen0_best_r2 - 1e-9


def test_sequential_reproducibility(regression_data) -> None:
    a = _short_reg().fit(
        regression_data.X_train,
        regression_data.y_train,
        X_val=regression_data.X_val,
        y_val=regression_data.y_val,
    )
    b = _short_reg().fit(
        regression_data.X_train,
        regression_data.y_train,
        X_val=regression_data.X_val,
        y_val=regression_data.y_val,
    )
    assert a.best_score_ == b.best_score_
    assert a.best_params_ == b.best_params_


def test_early_stopping_records_best_iteration(regression_data) -> None:
    reg = _short_reg(step_rounds=60, early_stopping_rounds=5).fit(
        regression_data.X_train,
        regression_data.y_train,
        X_val=regression_data.X_val,
        y_val=regression_data.y_val,
    )
    assert reg.history_["best_iteration"].notna().all()
    assert reg.predict(regression_data.X_val).shape == (regression_data.X_val.shape[0],)


def test_regressor_does_not_label_encode(regression_data) -> None:
    # The regressor must NOT label-encode targets (no classes_ attribute is set).
    reg = _short_reg().fit(
        regression_data.X_train,
        regression_data.y_train,
        X_val=regression_data.X_val,
        y_val=regression_data.y_val,
    )
    assert not hasattr(reg, "classes_")


def test_predict_before_fit_raises_not_fitted() -> None:
    reg = GeneticXGBRegressor()
    with pytest.raises(NotFittedError):
        reg.predict(np.zeros((3, 4), dtype=np.float32))


def test_invalid_hyperparams_raise_for_regressor(regression_data) -> None:
    # The shared validation engine also guards the regressor (F5/F6).
    reg = _short_reg(selection_top_k=6, population_size=6)
    with pytest.raises(ValueError, match="disabling all evolution"):
        reg.fit(
            regression_data.X_train,
            regression_data.y_train,
            X_val=regression_data.X_val,
            y_val=regression_data.y_val,
        )


# --- sklearn-compatibility battery -----------------------------------------------------------


def test_clone_returns_unfitted_estimator_with_same_params() -> None:
    reg = _short_reg(metric="mae")
    cloned = clone(reg)
    assert isinstance(cloned, GeneticXGBRegressor)
    assert cloned is not reg
    assert cloned.get_params() == reg.get_params()
    assert not hasattr(cloned, "best_booster_")


def test_get_params_set_params_round_trip() -> None:
    reg = GeneticXGBRegressor()
    reg.set_params(population_size=8, metric="r2", random_state=5)
    params = reg.get_params()
    assert params["population_size"] == 8
    assert params["metric"] == "r2"
    twin = GeneticXGBRegressor(**params)
    assert twin.get_params() == params


def test_score_returns_float_r2(regression_data) -> None:
    reg = _short_reg().fit(
        regression_data.X_train,
        regression_data.y_train,
        X_val=regression_data.X_val,
        y_val=regression_data.y_val,
    )
    score = reg.score(regression_data.X_val, regression_data.y_val)
    # RegressorMixin.score returns the R^2 coefficient of determination as a float.
    assert isinstance(score, float)


def test_feature_importances_shape_and_sum(regression_data) -> None:
    reg = _short_reg().fit(
        regression_data.X_train,
        regression_data.y_train,
        X_val=regression_data.X_val,
        y_val=regression_data.y_val,
    )
    importances = reg.feature_importances_
    assert importances.shape == (reg.n_features_in_,)
    assert importances.dtype == np.float64
    assert np.isclose(importances.sum(), 1.0)
    assert np.all(importances >= 0.0)


def test_feature_importances_before_fit_raises() -> None:
    with pytest.raises(NotFittedError):
        _ = GeneticXGBRegressor().feature_importances_


def test_n_features_in_recorded(regression_data) -> None:
    reg = _short_reg().fit(
        regression_data.X_train,
        regression_data.y_train,
        X_val=regression_data.X_val,
        y_val=regression_data.y_val,
    )
    assert reg.n_features_in_ == regression_data.X_train.shape[1]


def test_sample_weight_accepted_and_changes_fit(regression_data) -> None:
    base = _short_reg().fit(
        regression_data.X_train,
        regression_data.y_train,
        X_val=regression_data.X_val,
        y_val=regression_data.y_val,
    )
    rng = np.random.default_rng(0)
    weights = rng.uniform(0.1, 5.0, size=regression_data.X_train.shape[0]).astype(np.float32)
    weighted = _short_reg().fit(
        regression_data.X_train,
        regression_data.y_train,
        X_val=regression_data.X_val,
        y_val=regression_data.y_val,
        sample_weight=weights,
    )
    assert weighted.best_booster_ != base.best_booster_


def test_predict_wrong_n_features_raises(regression_data) -> None:
    reg = _short_reg().fit(
        regression_data.X_train,
        regression_data.y_train,
        X_val=regression_data.X_val,
        y_val=regression_data.y_val,
    )
    with pytest.raises(ValueError, match="features"):
        reg.predict(regression_data.X_val[:, :-1])


def test_dataframe_column_reorder_gives_same_predictions(regression_data) -> None:
    names = [f"col_{i}" for i in range(regression_data.X_train.shape[1])]
    df_train = pd.DataFrame(regression_data.X_train, columns=names)
    df_val = pd.DataFrame(regression_data.X_val, columns=names)
    reg = _short_reg().fit(
        df_train, regression_data.y_train, X_val=df_val, y_val=regression_data.y_val
    )
    assert list(reg.feature_names_in_) == names
    shuffled = df_val[names[::-1]]
    np.testing.assert_allclose(reg.predict(shuffled), reg.predict(df_val), rtol=0, atol=0)


def test_fit_x_y_internal_split(regression_data) -> None:
    # Regression uses an unstratified internal holdout when no validation set is given.
    reg = _short_reg().fit(regression_data.X_train, regression_data.y_train)
    assert reg.n_features_in_ == regression_data.X_train.shape[1]
    assert reg.predict(regression_data.X_val).shape == (regression_data.X_val.shape[0],)


def test_refit_full_with_sample_weight(regression_data) -> None:
    reg = _short_reg().fit(
        regression_data.X_train,
        regression_data.y_train,
        X_val=regression_data.X_val,
        y_val=regression_data.y_val,
    )
    x_all = np.vstack([regression_data.X_train, regression_data.X_val])
    y_all = np.concatenate([regression_data.y_train, regression_data.y_val])
    weights = np.ones(len(y_all), dtype=np.float32)
    reg.refit_full(x_all, y_all, sample_weight=weights)
    assert reg.refit_full_ is True
    assert reg.predict(regression_data.X_val).shape == (regression_data.X_val.shape[0],)


# --- new surface: booster access, apply, native save/load, keyword-only -----------------------


def test_get_booster_returns_xgboost_booster(regression_data) -> None:
    reg = _short_reg().fit(
        regression_data.X_train,
        regression_data.y_train,
        X_val=regression_data.X_val,
        y_val=regression_data.y_val,
    )
    assert isinstance(reg.get_booster(), xgb.Booster)


def test_apply_returns_2d_leaf_indices(regression_data) -> None:
    reg = _short_reg().fit(
        regression_data.X_train,
        regression_data.y_train,
        X_val=regression_data.X_val,
        y_val=regression_data.y_val,
    )
    leaves = reg.apply(regression_data.X_val)
    assert leaves.ndim == 2
    assert len(leaves) == regression_data.X_val.shape[0]


def test_save_load_model_round_trip(regression_data) -> None:
    reg = _short_reg().fit(
        regression_data.X_train,
        regression_data.y_train,
        X_val=regression_data.X_val,
        y_val=regression_data.y_val,
    )
    expected = reg.predict(regression_data.X_val)
    with tempfile.TemporaryDirectory() as tmp:
        fname = str(Path(tmp) / "model.json")
        reg.save_model(fname)
        fresh = GeneticXGBRegressor()
        fresh.load_model(fname)
    assert fresh.n_features_in_ == regression_data.X_train.shape[1]
    np.testing.assert_allclose(fresh.predict(regression_data.X_val), expected, rtol=0, atol=0)


def test_fit_rejects_positional_validation_args(regression_data) -> None:
    # X_val/y_val/sample_weight are keyword-only: 4 positional args is a TypeError.
    with pytest.raises(TypeError):
        _short_reg().fit(
            regression_data.X_train,
            regression_data.y_train,
            regression_data.X_val,
            regression_data.y_val,
        )
