"""Tests for genetic feature selection (real data, no mocks)."""

from __future__ import annotations

import numpy as np
import pytest
import xgboost as xgb
from sklearn.datasets import make_classification, make_regression

from genetic_xgb import GeneticXGBClassifier, GeneticXGBRegressor
from genetic_xgb.feature_selection import (
    crossover_masks,
    mutate_mask,
    sample_mask,
)
from genetic_xgb.member import PopulationMember
from genetic_xgb.metrics import resolve_metric
from genetic_xgb.search_space import Hyperparameter, SearchSpace
from genetic_xgb.strategy import GeneticStrategy
from genetic_xgb.trainer import train_step

BINARY_BASE = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "max_bin": 256,
    "verbosity": 0,
    "nthread": 1,
}


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# --------------------------- feature_selection.py helpers ---------------------------


def test_sample_mask_within_bounds_and_min_features() -> None:
    mask = sample_mask(10, init_prob=0.5, rng=_rng(1), min_features=1)
    assert mask.shape == (10,) and mask.dtype == bool
    assert 1 <= int(mask.sum()) <= 10


def test_sample_mask_enforces_min_features_via_floor() -> None:
    # min_features == n forces the _ensure_min "turn on off-bits" path.
    mask = sample_mask(8, init_prob=0.5, rng=_rng(2), min_features=8)
    assert int(mask.sum()) == 8


def test_crossover_masks_dominance_extremes() -> None:
    dom = np.array([True, True, True, True])
    rec = np.array([False, False, False, False])
    assert np.array_equal(crossover_masks(dom, rec, 1.0, _rng(0)), dom)
    assert np.array_equal(crossover_masks(dom, rec, 0.0, _rng(0)), rec)


def test_mutate_mask_flip_rate_zero_is_noop() -> None:
    mask = np.array([True, False, True, False, True])
    out = mutate_mask(mask, flip_rate=0.0, rng=_rng(0), min_features=1)
    assert np.array_equal(out, mask)


def test_mutate_mask_flip_all_then_min_features_floor() -> None:
    mask = np.array([True, True, True])
    # flip_rate 1.0 flips all to False, then _ensure_min turns >= min_features back on.
    out = mutate_mask(mask, flip_rate=1.0, rng=_rng(3), min_features=2)
    assert int(out.sum()) >= 2


# --------------------------- PopulationMember.inherit_from ---------------------------


def _tiny_booster_bytes() -> bytes:
    x = np.random.default_rng(0).normal(size=(40, 4)).astype(np.float32)
    y = (x[:, 0] > 0).astype(int)
    booster = xgb.train(BINARY_BASE, xgb.DMatrix(x, label=y), num_boost_round=3)
    return bytes(booster.save_raw())


def test_inherit_from_changed_mask_resets_warm_start() -> None:
    dom = PopulationMember(
        id=0,
        hyperparams={"a": 1},
        booster_bytes=_tiny_booster_bytes(),
        n_rounds=3,
        best_iteration=2,
        feature_mask=np.array([True, True, False, False]),
    )
    rec = PopulationMember(
        id=1, hyperparams={"a": 2}, feature_mask=np.array([True, True, True, True])
    )
    child = PopulationMember(id=2, hyperparams={})
    child.inherit_from(dom, rec, {"a": 1}, feature_mask=np.array([False, True, True, False]))
    # Different column set -> warm-start dropped, cold start next time.
    assert child.booster_bytes is None
    assert child.n_rounds == 0
    assert child.best_iteration is None
    assert np.array_equal(child.feature_mask, [False, True, True, False])


def test_inherit_from_same_mask_keeps_warm_start() -> None:
    mask = np.array([True, True, False, False])
    dom = PopulationMember(
        id=0,
        hyperparams={"a": 1},
        booster_bytes=_tiny_booster_bytes(),
        n_rounds=3,
        feature_mask=mask,
    )
    rec = PopulationMember(id=1, hyperparams={"a": 2}, feature_mask=mask)
    child = PopulationMember(id=2, hyperparams={})
    child.inherit_from(dom, rec, {"a": 1}, feature_mask=mask.copy())
    assert child.booster_bytes == dom.booster_bytes  # warm-start preserved
    assert child.n_rounds == 3


# --------------------------- trainer.train_step with a mask ---------------------------


def test_train_step_uses_only_selected_columns(binary_data) -> None:
    mask = np.zeros(binary_data.X_train.shape[1], dtype=bool)
    mask[:3] = True  # keep 3 columns
    out = train_step(
        booster_bytes=None,
        hyperparams={"max_depth": 3},
        train=(binary_data.X_train, binary_data.y_train),
        val=(binary_data.X_val, binary_data.y_val),
        step_rounds=10,
        metric=resolve_metric("logloss"),
        base_params=BINARY_BASE,
        seed=0,
        feature_mask=mask,
    )
    booster = xgb.Booster()
    booster.load_model(bytearray(out["booster_bytes"]))
    assert booster.num_features() == 3


# --------------------------- strategy.evolve with feature selection ---------------------------


def _space() -> SearchSpace:
    return SearchSpace([Hyperparameter("max_depth", "int", low=3, high=6)])


def test_evolve_breeds_masks_and_cold_starts_changed_offspring() -> None:
    n_feat = 6
    rng = _rng(0)
    members = []
    for i in range(4):
        members.append(
            PopulationMember(
                id=i,
                hyperparams={"max_depth": 3 + i % 3},
                booster_bytes=_tiny_booster_bytes(),
                score=float(i),
                n_rounds=3,
                feature_mask=sample_mask(n_feat, 0.5, rng, 1),
            )
        )
    strat = GeneticStrategy(
        space=_space(),
        top_k=2,
        dominance_prob=0.5,
        mutation_fraction=0.5,
        mutation_intensity=0.3,
        resample_prob=0.0,
        greater_is_better=True,
        feature_selection=True,
        feature_mutation_rate=0.5,
        min_features=1,
    )
    evolved = strat.evolve(members, _rng(1))
    assert all(m.feature_mask is not None for m in evolved)
    survivors = {m.id for m in strat.select(members)}
    # Offspring whose mask changed cold-start (booster dropped).
    for m in evolved:
        if m.id not in survivors and m.booster_bytes is None:
            break
    else:
        raise AssertionError("expected at least one cold-started offspring")


# --------------------------- estimator integration ---------------------------


def _noisy_classification():
    X, y = make_classification(  # noqa: N806
        n_samples=500, n_features=20, n_informative=5, n_redundant=0, random_state=0
    )
    return X.astype(np.float32), y


def test_classifier_feature_selection_end_to_end() -> None:
    X, y = _noisy_classification()  # noqa: N806
    clf = GeneticXGBClassifier(
        feature_selection=True,
        population_size=6,
        selection_top_k=2,
        generations=4,
        step_rounds=5,
        executor="sequential",
        random_state=0,
    ).fit(X, y)
    support = clf.get_support()
    assert support.shape == (20,) and support.dtype == bool
    assert 1 <= int(support.sum()) <= 20
    assert np.array_equal(clf.get_support(indices=True), np.flatnonzero(support))
    # importances align to original features: zero on unselected columns.
    importances = clf.feature_importances_
    assert importances.shape == (20,)
    assert np.all(importances[~support] == 0.0)
    # an excluded column is provably unused: perturbing it does not change predictions.
    excluded = np.flatnonzero(~support)
    if excluded.size:
        perturbed = X.copy()
        perturbed[:, excluded[0]] += 1000.0
        assert np.array_equal(clf.predict(X), clf.predict(perturbed))


def test_regressor_feature_selection_and_refit_full() -> None:
    X, y = make_regression(n_samples=400, n_features=15, n_informative=5, random_state=0)  # noqa: N806
    X = X.astype(np.float32)  # noqa: N806
    reg = GeneticXGBRegressor(
        feature_selection=True,
        population_size=6,
        selection_top_k=2,
        generations=3,
        step_rounds=5,
        executor="sequential",
        random_state=0,
    ).fit(X, y)
    assert reg.get_support().shape == (15,)
    reg.refit_full(X, y)  # refit restricted to the selected columns
    assert reg.refit_full_ is True
    assert reg.predict(X).shape == (X.shape[0],)


def test_overfit_penalty_records_train_and_val_scores() -> None:
    X, y = _noisy_classification()  # noqa: N806
    clf = GeneticXGBClassifier(
        feature_selection=True,
        overfit_penalty=0.5,
        population_size=6,
        selection_top_k=2,
        generations=3,
        step_rounds=5,
        executor="sequential",
        random_state=0,
    ).fit(X, y)
    # With the penalty active, every member records raw train and validation scores.
    assert clf.history_["val_score"].notna().all()
    assert clf.history_["train_score"].notna().all()
    assert clf.predict(X).shape == (X.shape[0],)


def test_overfit_penalty_ignored_without_feature_selection(binary_data) -> None:
    # overfit_penalty only takes effect when feature_selection is on.
    clf = GeneticXGBClassifier(
        overfit_penalty=0.5,
        population_size=4,
        selection_top_k=2,
        generations=2,
        step_rounds=3,
        executor="sequential",
        random_state=0,
    ).fit(
        binary_data.X_train, binary_data.y_train, X_val=binary_data.X_val, y_val=binary_data.y_val
    )
    assert clf.history_["val_score"].isna().all()
    assert clf.history_["train_score"].isna().all()


def test_history_records_n_features_selected() -> None:
    X, y = _noisy_classification()  # noqa: N806
    clf = GeneticXGBClassifier(
        feature_selection=True,
        population_size=4,
        selection_top_k=2,
        generations=2,
        step_rounds=3,
        executor="sequential",
        random_state=0,
    ).fit(X, y)
    assert "n_features_selected" in clf.history_.columns
    assert clf.history_["n_features_selected"].between(1, 20).all()


def test_save_model_unsupported_with_feature_selection(tmp_path) -> None:
    X, y = _noisy_classification()  # noqa: N806
    clf = GeneticXGBClassifier(
        feature_selection=True,
        population_size=4,
        selection_top_k=2,
        generations=2,
        step_rounds=3,
        executor="sequential",
        random_state=0,
    ).fit(X, y)
    with pytest.raises(NotImplementedError, match="feature_selection"):
        clf.save_model(str(tmp_path / "m.json"))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"feature_init_prob": 0.0}, "feature_init_prob"),
        ({"feature_init_prob": 1.5}, "feature_init_prob"),
        ({"feature_mutation_rate": -0.1}, "feature_mutation_rate"),
        ({"feature_mutation_rate": 1.1}, "feature_mutation_rate"),
        ({"min_features": 0}, "min_features"),
        ({"overfit_penalty": -0.1}, "overfit_penalty"),
    ],
)
def test_invalid_feature_selection_params_raise(kwargs, match) -> None:
    X, y = _noisy_classification()  # noqa: N806
    est = GeneticXGBClassifier(
        feature_selection=True,
        population_size=4,
        selection_top_k=2,
        generations=2,
        step_rounds=3,
        executor="sequential",
        random_state=0,
        **kwargs,
    )
    with pytest.raises(ValueError, match=match):
        est.fit(X, y)


def test_feature_selection_off_keeps_all_features(binary_data) -> None:
    clf = GeneticXGBClassifier(
        population_size=4,
        selection_top_k=2,
        generations=2,
        step_rounds=3,
        executor="sequential",
        random_state=0,
    ).fit(
        binary_data.X_train, binary_data.y_train, X_val=binary_data.X_val, y_val=binary_data.y_val
    )
    assert clf.feature_mask_ is None
    assert clf.get_support().all()
    assert clf.get_support(indices=True).shape == (binary_data.X_train.shape[1],)
