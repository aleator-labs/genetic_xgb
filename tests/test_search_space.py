from __future__ import annotations

import numpy as np
import pytest

from genetic_xgb.search_space import (
    Hyperparameter,
    SearchSpace,
    default_classification_space,
    default_regression_space,
)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------- #
# Hyperparameter.sample
# --------------------------------------------------------------------------- #
def test_sample_float_linear_within_bounds() -> None:
    hp = Hyperparameter("a", "float", low=-2.0, high=5.0)
    rng = _rng()
    for _ in range(500):
        v = hp.sample(rng)
        assert -2.0 <= v <= 5.0


def test_sample_float_log_within_bounds_and_spread() -> None:
    hp = Hyperparameter("lr", "float", low=1e-4, high=1.0, log=True)
    rng = _rng()
    vals = [hp.sample(rng) for _ in range(2000)]
    assert all(1e-4 <= v <= 1.0 for v in vals)
    # log sampling should reach the small-magnitude region the linear one rarely would
    assert min(vals) < 1e-2


def test_sample_int_is_integral_and_within_bounds() -> None:
    hp = Hyperparameter("d", "int", low=3, high=10)
    rng = _rng()
    for _ in range(500):
        v = hp.sample(rng)
        assert isinstance(v, int)
        assert not isinstance(v, bool)
        assert 3 <= v <= 10


def test_sample_categorical_returns_valid_python_scalar() -> None:
    choices = ("depthwise", "lossguide")
    hp = Hyperparameter("g", "categorical", choices=choices)
    rng = _rng()
    seen = set()
    for _ in range(200):
        v = hp.sample(rng)
        assert type(v) is str
        assert v in choices
        seen.add(v)
    assert seen == set(choices)  # both choices reachable


# --------------------------------------------------------------------------- #
# Hyperparameter.mutate
# --------------------------------------------------------------------------- #
def test_mutate_float_linear_respects_bounds() -> None:
    hp = Hyperparameter("a", "float", low=0.0, high=1.0)
    rng = _rng()
    v = 0.5
    for _ in range(2000):
        v = hp.mutate(v, rng, intensity=2.0)
        assert 0.0 <= v <= 1.0


def test_mutate_float_log_respects_bounds() -> None:
    hp = Hyperparameter("lr", "float", low=1e-4, high=1.0, log=True)
    rng = _rng()
    v = 1e-2
    for _ in range(2000):
        v = hp.mutate(v, rng, intensity=1.0)
        assert 1e-4 <= v <= 1.0


def test_mutate_int_stays_integral_and_bounded() -> None:
    hp = Hyperparameter("d", "int", low=3, high=10)
    rng = _rng()
    v = 6
    for _ in range(2000):
        v = hp.mutate(v, rng, intensity=1.0)
        assert isinstance(v, int)
        assert not isinstance(v, bool)
        assert 3 <= v <= 10


def test_mutate_categorical_resamples_to_valid_choice() -> None:
    choices = ("a", "b", "c")
    hp = Hyperparameter("g", "categorical", choices=choices)
    rng = _rng()
    seen = set()
    for _ in range(300):
        v = hp.mutate("a", rng, intensity=0.5)
        assert v in choices
        seen.add(v)
    assert seen == set(choices)


def test_mutate_intensity_scales_magnitude() -> None:
    hp = Hyperparameter("a", "float", low=0.0, high=100.0)
    start = 50.0

    def avg_abs_change(intensity: float) -> float:
        deltas = []
        for s in range(400):
            rng = _rng(s)
            deltas.append(abs(hp.mutate(start, rng, intensity=intensity) - start))
        return float(np.mean(deltas))

    small = avg_abs_change(0.01)
    large = avg_abs_change(0.4)
    assert large > small


# --------------------------------------------------------------------------- #
# Hyperparameter.clip
# --------------------------------------------------------------------------- #
def test_clip_float_clamps() -> None:
    hp = Hyperparameter("a", "float", low=0.0, high=1.0)
    assert hp.clip(-5.0) == 0.0
    assert hp.clip(2.0) == 1.0
    assert hp.clip(0.4) == 0.4


def test_clip_int_rounds_and_clamps() -> None:
    hp = Hyperparameter("d", "int", low=3, high=10)
    out_low = hp.clip(-2.0)
    out_high = hp.clip(99.0)
    out_mid = hp.clip(5.6)
    assert out_low == 3 and isinstance(out_low, int)
    assert out_high == 10
    assert out_mid == 6


def test_clip_categorical_valid_and_invalid() -> None:
    choices = ("x", "y", "z")
    hp = Hyperparameter("g", "categorical", choices=choices)
    assert hp.clip("y") == "y"
    assert hp.clip("nope") == "x"  # falls back to choices[0]


# --------------------------------------------------------------------------- #
# SearchSpace
# --------------------------------------------------------------------------- #
def _small_space() -> SearchSpace:
    return SearchSpace(
        [
            Hyperparameter("a", "float", low=0.0, high=10.0),
            Hyperparameter("b", "float", low=0.0, high=10.0),
            Hyperparameter("c", "float", low=0.0, high=10.0),
            Hyperparameter("d", "int", low=0, high=100),
        ]
    )


def test_searchspace_names_and_params() -> None:
    sp = _small_space()
    assert sp.names() == ["a", "b", "c", "d"]
    assert isinstance(sp.params, tuple)
    assert len(sp.params) == 4


def test_searchspace_sample_keys_and_bounds() -> None:
    sp = _small_space()
    params = sp.sample(_rng())
    assert set(params) == {"a", "b", "c", "d"}
    assert 0.0 <= params["a"] <= 10.0
    assert isinstance(params["d"], int)


def test_searchspace_clip_applies_per_gene() -> None:
    sp = _small_space()
    clipped = sp.clip({"a": -1.0, "b": 99.0, "c": 5.0, "d": 250.0})
    assert clipped["a"] == 0.0
    assert clipped["b"] == 10.0
    assert clipped["c"] == 5.0
    assert clipped["d"] == 100


def test_searchspace_mutate_changes_exactly_fraction_of_genes() -> None:
    sp = _small_space()
    params = {"a": 5.0, "b": 5.0, "c": 5.0, "d": 50}
    # n=4, fraction=0.5 -> round(2.0)=2 genes mutate; large intensity => detectable
    mutated = sp.mutate(params, _rng(7), fraction=0.5, intensity=5.0, resample_prob=0.0)
    changed = [name for name in params if mutated[name] != params[name]]
    assert len(changed) == 2
    # untouched genes are exactly preserved
    for name in params:
        if name not in changed:
            assert mutated[name] == params[name]
    # result is a NEW dict
    assert mutated is not params


def test_searchspace_mutate_tiny_positive_fraction_mutates_one_gene() -> None:
    sp = _small_space()  # 4 genes
    params = {"a": 5.0, "b": 5.0, "c": 5.0, "d": 50}
    # round(0.1 * 4) == round(0.4) == 0, but a positive fraction must mutate >= 1 gene
    mutated = sp.mutate(params, _rng(7), fraction=0.1, intensity=5.0, resample_prob=0.0)
    changed = [name for name in params if mutated[name] != params[name]]
    assert len(changed) == 1


def test_searchspace_mutate_default_fraction_count_uses_round_not_ceil() -> None:
    sp = default_classification_space()  # 11 core genes
    params = sp.sample(_rng(0))
    # round(0.3 * 11) == round(3.3) == 3 (ceil would wrongly give 4)
    mutated = sp.mutate(params, _rng(0), fraction=0.3, intensity=5.0, resample_prob=0.0)
    changed = [name for name in params if mutated[name] != params[name]]
    assert len(changed) == 3


def test_searchspace_mutate_zero_fraction_changes_nothing() -> None:
    sp = _small_space()
    params = {"a": 5.0, "b": 5.0, "c": 5.0, "d": 50}
    mutated = sp.mutate(params, _rng(1), fraction=0.0, intensity=5.0, resample_prob=0.0)
    assert mutated == params
    assert mutated is not params


def test_searchspace_mutate_resample_path_stays_in_bounds() -> None:
    sp = _small_space()
    params = {"a": 5.0, "b": 5.0, "c": 5.0, "d": 50}
    # resample_prob=1.0 forces every chosen gene through the sample() branch
    mutated = sp.mutate(params, _rng(3), fraction=1.0, intensity=5.0, resample_prob=1.0)
    assert 0.0 <= mutated["a"] <= 10.0
    assert 0.0 <= mutated["b"] <= 10.0
    assert 0.0 <= mutated["c"] <= 10.0
    assert isinstance(mutated["d"], int)
    assert 0 <= mutated["d"] <= 100


def test_searchspace_mutate_is_deterministic() -> None:
    sp = _small_space()
    params = {"a": 5.0, "b": 5.0, "c": 5.0, "d": 50}
    m1 = sp.mutate(params, _rng(99), fraction=0.5, intensity=2.0, resample_prob=0.3)
    m2 = sp.mutate(params, _rng(99), fraction=0.5, intensity=2.0, resample_prob=0.3)
    assert m1 == m2


def test_searchspace_sample_is_deterministic() -> None:
    sp = _small_space()
    assert sp.sample(_rng(5)) == sp.sample(_rng(5))


# --------------------------------------------------------------------------- #
# default_classification_space
# --------------------------------------------------------------------------- #
CORE_NAMES = {
    "learning_rate",
    "max_depth",
    "min_child_weight",
    "gamma",
    "subsample",
    "colsample_bytree",
    "colsample_bylevel",
    "colsample_bynode",
    "max_delta_step",
    "reg_alpha",
    "reg_lambda",
}


def test_default_space_core_contents() -> None:
    sp = default_classification_space()
    assert set(sp.names()) == CORE_NAMES
    by = {p.name: p for p in sp.params}
    assert by["learning_rate"].kind == "float"
    assert by["learning_rate"].log is True
    assert (by["learning_rate"].low, by["learning_rate"].high) == (1e-3, 0.3)
    assert by["max_depth"].kind == "int"
    assert (by["max_depth"].low, by["max_depth"].high) == (3, 10)
    assert by["min_child_weight"].log is True
    assert by["gamma"].log is False
    assert (by["reg_lambda"].low, by["reg_lambda"].high) == (1e-8, 10)


def test_default_space_extended_contents() -> None:
    sp = default_classification_space(extended=True)
    names = set(sp.names())
    assert names >= CORE_NAMES
    assert {"grow_policy", "max_leaves", "num_parallel_tree"} <= names
    by = {p.name: p for p in sp.params}
    assert by["grow_policy"].kind == "categorical"
    assert by["grow_policy"].choices == ("depthwise", "lossguide")
    assert by["max_leaves"].kind == "int"
    assert (by["max_leaves"].low, by["max_leaves"].high) == (0, 256)
    assert (by["num_parallel_tree"].low, by["num_parallel_tree"].high) == (1, 4)


def test_default_space_imbalance_contents() -> None:
    sp = default_classification_space(imbalance=True)
    names = set(sp.names())
    assert names >= CORE_NAMES
    assert "scale_pos_weight" in names
    by = {p.name: p for p in sp.params}
    assert by["scale_pos_weight"].log is True
    assert (by["scale_pos_weight"].low, by["scale_pos_weight"].high) == (0.1, 10)


def test_default_space_extended_and_imbalance_together() -> None:
    sp = default_classification_space(extended=True, imbalance=True)
    names = set(sp.names())
    expected = CORE_NAMES | {
        "grow_policy",
        "max_leaves",
        "num_parallel_tree",
        "scale_pos_weight",
    }
    assert names == expected


def test_default_space_sample_round_trips_through_clip() -> None:
    sp = default_classification_space(extended=True, imbalance=True)
    rng = _rng(11)
    params = sp.sample(rng)
    clipped = sp.clip(params)
    # sampling already produces in-bounds values, so clip is a no-op here
    assert clipped == params


def test_searchspace_uses_real_fixture_rng(rng: np.random.Generator) -> None:
    sp = default_classification_space()
    params = sp.sample(rng)
    assert set(params) == CORE_NAMES


def test_default_regression_space_core_contents() -> None:
    sp = default_regression_space()
    # Same tree genes as classification, but NO class-imbalance gene.
    assert set(sp.names()) == CORE_NAMES
    assert "scale_pos_weight" not in sp.names()


def test_default_regression_space_extended_contents() -> None:
    sp = default_regression_space(extended=True)
    names = set(sp.names())
    assert names == CORE_NAMES | {"grow_policy", "max_leaves", "num_parallel_tree"}
    assert "scale_pos_weight" not in names


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
