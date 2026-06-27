"""Tests for the GeneticStrategy genetic operators."""

from __future__ import annotations

import numpy as np
import xgboost as xgb

from pbt_xgb.member import PopulationMember
from pbt_xgb.search_space import Hyperparameter, SearchSpace
from pbt_xgb.strategy import GeneticStrategy

GENE_NAMES = [f"g{i}" for i in range(10)]


def _space() -> SearchSpace:
    return SearchSpace([Hyperparameter(name, "float", low=0.0, high=10.0) for name in GENE_NAMES])


def _strategy(
    *,
    top_k: int = 2,
    dominance_prob: float = 0.7,
    mutation_fraction: float = 0.3,
    mutation_intensity: float = 0.2,
    resample_prob: float = 0.1,
    greater_is_better: bool = True,
) -> GeneticStrategy:
    return GeneticStrategy(
        space=_space(),
        top_k=top_k,
        dominance_prob=dominance_prob,
        mutation_fraction=mutation_fraction,
        mutation_intensity=mutation_intensity,
        resample_prob=resample_prob,
        greater_is_better=greater_is_better,
    )


def _tiny_booster(seed: int, rounds: int = 3) -> xgb.Booster:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((40, 3)).astype(np.float32)  # noqa: N806
    y = (X[:, 0] > 0).astype(int)
    dtrain = xgb.DMatrix(X, label=y)
    return xgb.train(
        {"objective": "binary:logistic", "tree_method": "hist", "verbosity": 0},
        dtrain,
        num_boost_round=rounds,
    )


def _member(mid: int, score: float | None, seed: int | None = None) -> PopulationMember:
    params = dict.fromkeys(GENE_NAMES, 5.0)
    m = PopulationMember(id=mid, hyperparams=params, score=score, n_rounds=3)
    if seed is not None:
        m.save_booster(_tiny_booster(seed))
    return m


# --------------------------------------------------------------------------- rank / select


def test_rank_greater_is_better_orders_descending() -> None:
    strat = _strategy(greater_is_better=True)
    members = [_member(0, 0.1), _member(1, 0.9), _member(2, 0.5)]
    ranked = strat.rank(members)
    assert [m.id for m in ranked] == [1, 2, 0]


def test_rank_lower_is_better_orders_ascending() -> None:
    strat = _strategy(greater_is_better=False)
    members = [_member(0, 0.1), _member(1, 0.9), _member(2, 0.5)]
    ranked = strat.rank(members)
    assert [m.id for m in ranked] == [0, 2, 1]


def test_rank_none_scores_sort_last_both_directions() -> None:
    members = [_member(0, None), _member(1, 0.5), _member(2, None), _member(3, 0.9)]

    ranked_hi = _strategy(greater_is_better=True).rank(members)
    assert [m.id for m in ranked_hi[:2]] == [3, 1]
    assert {m.id for m in ranked_hi[2:]} == {0, 2}

    ranked_lo = _strategy(greater_is_better=False).rank(members)
    assert [m.id for m in ranked_lo[:2]] == [1, 3]
    assert {m.id for m in ranked_lo[2:]} == {0, 2}


def test_rank_does_not_mutate_input_order() -> None:
    members = [_member(0, 0.1), _member(1, 0.9)]
    _strategy().rank(members)
    assert [m.id for m in members] == [0, 1]


def test_select_returns_top_k() -> None:
    strat = _strategy(top_k=2, greater_is_better=True)
    members = [_member(0, 0.1), _member(1, 0.9), _member(2, 0.5), _member(3, 0.7)]
    chosen = strat.select(members)
    assert [m.id for m in chosen] == [1, 3]


# --------------------------------------------------------------------------- crossover


def test_crossover_dominance_one_is_all_dominant() -> None:
    strat = _strategy(dominance_prob=1.0)
    dom = PopulationMember(id=0, hyperparams=dict.fromkeys(GENE_NAMES, 2.0))
    rec = PopulationMember(id=1, hyperparams=dict.fromkeys(GENE_NAMES, 8.0))
    child = strat.crossover(dom, rec, np.random.default_rng(0))
    assert child == dict.fromkeys(GENE_NAMES, 2.0)


def test_crossover_dominance_zero_is_all_recessive() -> None:
    strat = _strategy(dominance_prob=0.0)
    dom = PopulationMember(id=0, hyperparams=dict.fromkeys(GENE_NAMES, 2.0))
    rec = PopulationMember(id=1, hyperparams=dict.fromkeys(GENE_NAMES, 8.0))
    child = strat.crossover(dom, rec, np.random.default_rng(0))
    assert child == dict.fromkeys(GENE_NAMES, 8.0)


def test_crossover_intermediate_mixes_genes() -> None:
    strat = _strategy(dominance_prob=0.5)
    dom = PopulationMember(id=0, hyperparams=dict.fromkeys(GENE_NAMES, 2.0))
    rec = PopulationMember(id=1, hyperparams=dict.fromkeys(GENE_NAMES, 8.0))
    child = strat.crossover(dom, rec, np.random.default_rng(7))
    values = set(child.values())
    assert values == {2.0, 8.0}  # both parents contributed at least one gene
    assert set(child) == set(GENE_NAMES)


# --------------------------------------------------------------------------- mutate


def test_mutate_stays_in_bounds_and_keeps_keys() -> None:
    strat = _strategy(mutation_fraction=1.0, mutation_intensity=0.5, resample_prob=0.2)
    params = dict.fromkeys(GENE_NAMES, 5.0)
    out = strat.mutate(params, np.random.default_rng(3))
    assert set(out) == set(GENE_NAMES)
    for value in out.values():
        assert 0.0 <= value <= 10.0
    assert params == dict.fromkeys(GENE_NAMES, 5.0)  # original untouched


# --------------------------------------------------------------------------- evolve


def test_evolve_preserves_size_and_ids() -> None:
    strat = _strategy(top_k=2, greater_is_better=True)
    members = [_member(i, score, seed=i) for i, score in enumerate([0.9, 0.8, 0.1, 0.2, 0.5, 0.3])]
    evolved = strat.evolve(members, np.random.default_rng(0))
    assert len(evolved) == len(members)
    assert [m.id for m in evolved] == [0, 1, 2, 3, 4, 5]


def test_evolve_keeps_survivors_unchanged() -> None:
    strat = _strategy(top_k=2, greater_is_better=True)
    members = [_member(i, score, seed=i) for i, score in enumerate([0.9, 0.8, 0.1, 0.2, 0.5, 0.3])]
    surv0_bytes = members[0].booster_bytes
    surv1_bytes = members[1].booster_bytes
    evolved = strat.evolve(members, np.random.default_rng(0))
    by_id = {m.id: m for m in evolved}

    assert by_id[0].score == 0.9
    assert by_id[1].score == 0.8
    assert by_id[0].booster_bytes == surv0_bytes
    assert by_id[1].booster_bytes == surv1_bytes
    assert by_id[0].parents is None
    assert by_id[1].parents is None


def test_evolve_offspring_inherit_dominant_and_record_parents() -> None:
    strat = _strategy(top_k=2, greater_is_better=True)
    members = [_member(i, score, seed=i) for i, score in enumerate([0.9, 0.8, 0.1, 0.2, 0.5, 0.3])]
    dominant_bytes = members[0].booster_bytes  # id 0 is the better of the two survivors
    evolved = strat.evolve(members, np.random.default_rng(0))
    by_id = {m.id: m for m in evolved}

    for off_id in (2, 3, 4, 5):
        off = by_id[off_id]
        assert off.score is None
        # only two survivors {0, 1}; dominant (better score) is always id 0
        assert off.parents == (0, 1)
        assert off.booster_bytes == dominant_bytes
        # mutated child params still valid and in bounds
        assert set(off.hyperparams) == set(GENE_NAMES)
        for value in off.hyperparams.values():
            assert 0.0 <= value <= 10.0


def test_evolve_top_k_one_uses_single_survivor_for_both_parents() -> None:
    strat = _strategy(top_k=1, greater_is_better=True)
    members = [_member(i, score, seed=i) for i, score in enumerate([0.2, 0.9, 0.5])]
    survivor_bytes = members[1].booster_bytes  # id 1 has best score
    evolved = strat.evolve(members, np.random.default_rng(1))
    by_id = {m.id: m for m in evolved}

    assert by_id[1].score == 0.9  # survivor untouched
    for off_id in (0, 2):
        off = by_id[off_id]
        assert off.parents == (1, 1)
        assert off.booster_bytes == survivor_bytes
        assert off.score is None


def test_evolve_lower_is_better_selects_smallest_scores() -> None:
    strat = _strategy(top_k=1, greater_is_better=False)
    members = [_member(i, score, seed=i) for i, score in enumerate([0.7, 0.1, 0.5])]
    survivor_bytes = members[1].booster_bytes  # id 1 has lowest (best) score
    evolved = strat.evolve(members, np.random.default_rng(2))
    by_id = {m.id: m for m in evolved}

    assert by_id[1].score == 0.1
    assert by_id[1].parents is None
    for off_id in (0, 2):
        assert by_id[off_id].parents == (1, 1)
        assert by_id[off_id].booster_bytes == survivor_bytes
