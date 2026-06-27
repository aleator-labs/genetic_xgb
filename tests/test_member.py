"""Tests for the PopulationMember dataclass."""

from __future__ import annotations

import pickle

import numpy as np
import xgboost as xgb

from genetic_xgb.member import PopulationMember


def _tiny_booster(rounds: int = 5) -> xgb.Booster:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 3)).astype(np.float32)  # noqa: N806
    y = (X[:, 0] > 0).astype(int)
    dtrain = xgb.DMatrix(X, label=y)
    return xgb.train(
        {"objective": "binary:logistic", "tree_method": "hist", "verbosity": 0},
        dtrain,
        num_boost_round=rounds,
    )


def test_defaults() -> None:
    m = PopulationMember(id=7, hyperparams={"a": 1})
    assert m.id == 7
    assert m.hyperparams == {"a": 1}
    assert m.booster_bytes is None
    assert m.score is None
    assert m.n_rounds == 0
    assert m.parents is None


def test_save_and_load_roundtrip_preserves_rounds() -> None:
    booster = _tiny_booster(rounds=5)
    m = PopulationMember(id=0, hyperparams={})
    m.save_booster(booster)
    assert isinstance(m.booster_bytes, bytes)

    loaded = m.load_booster()
    assert loaded is not None
    assert loaded.num_boosted_rounds() == booster.num_boosted_rounds() == 5

    # predictions match the original booster bit-for-bit
    rng = np.random.default_rng(1)
    X = rng.standard_normal((10, 3)).astype(np.float32)  # noqa: N806
    d = xgb.DMatrix(X)
    np.testing.assert_array_equal(loaded.predict(d), booster.predict(d))


def test_load_booster_none_when_no_bytes() -> None:
    m = PopulationMember(id=1, hyperparams={})
    assert m.load_booster() is None


def test_pickle_roundtrip_preserves_bytes() -> None:
    booster = _tiny_booster(rounds=4)
    m = PopulationMember(id=3, hyperparams={"lr": 0.1}, score=0.9, n_rounds=4)
    m.save_booster(booster)

    restored = pickle.loads(pickle.dumps(m))
    assert restored.id == 3
    assert restored.hyperparams == {"lr": 0.1}
    assert restored.score == 0.9
    assert restored.n_rounds == 4
    assert restored.booster_bytes == m.booster_bytes
    assert restored.load_booster().num_boosted_rounds() == 4


def test_inherit_from_copies_dominant_and_resets_score() -> None:
    dom_booster = _tiny_booster(rounds=6)
    dominant = PopulationMember(id=2, hyperparams={"x": 1}, score=0.95, n_rounds=6)
    dominant.save_booster(dom_booster)

    recessive = PopulationMember(id=5, hyperparams={"x": 9}, score=0.5, n_rounds=3)
    recessive.save_booster(_tiny_booster(rounds=3))

    child = PopulationMember(id=8, hyperparams={"old": True}, score=0.1, n_rounds=99)
    new_params = {"x": 4}
    child.inherit_from(dominant, recessive, new_params)

    assert child.id == 8  # keeps own id
    assert child.booster_bytes == dominant.booster_bytes
    assert child.hyperparams == new_params
    assert child.n_rounds == 6
    assert child.parents == (2, 5)
    assert child.score is None
    # warm-started booster usable
    assert child.load_booster().num_boosted_rounds() == 6


def test_inherit_from_dominant_without_booster() -> None:
    dominant = PopulationMember(id=0, hyperparams={"x": 1})
    recessive = PopulationMember(id=1, hyperparams={"x": 2})
    child = PopulationMember(id=2, hyperparams={}, score=0.3)
    child.inherit_from(dominant, recessive, {"x": 3})
    assert child.booster_bytes is None
    assert child.parents == (0, 1)
    assert child.score is None
    assert child.load_booster() is None
