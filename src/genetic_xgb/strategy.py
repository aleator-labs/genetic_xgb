"""Genetic strategy: selection, crossover and mutation operators for PBT."""

from __future__ import annotations

import math
from typing import Any

from genetic_xgb.member import PopulationMember
from genetic_xgb.search_space import SearchSpace


class GeneticStrategy:
    """Rank/select survivors and breed offspring via crossover + mutation."""

    def __init__(
        self,
        *,
        space: SearchSpace,
        top_k: int,
        dominance_prob: float,
        mutation_fraction: float,
        mutation_intensity: float,
        resample_prob: float,
        greater_is_better: bool,
    ) -> None:
        self.space = space
        self.top_k = top_k
        self.dominance_prob = dominance_prob
        self.mutation_fraction = mutation_fraction
        self.mutation_intensity = mutation_intensity
        self.resample_prob = resample_prob
        self.greater_is_better = greater_is_better

    def rank(self, members: list[PopulationMember]) -> list[PopulationMember]:
        """Best-first ordering honoring direction.

        Members whose score is ``None`` or non-finite (``NaN`` / ``inf``) sort last
        (treated as worst), so such a member can never win selection.
        """

        def key(member: PopulationMember) -> tuple[int, float]:
            if member.score is None or not math.isfinite(member.score):
                return (1, 0.0)
            return (0, -member.score if self.greater_is_better else member.score)

        return sorted(members, key=key)

    def select(self, members: list[PopulationMember]) -> list[PopulationMember]:
        """The ``top_k`` best members."""
        return self.rank(members)[: self.top_k]

    def crossover(
        self, dominant: PopulationMember, recessive: PopulationMember, rng
    ) -> dict[str, Any]:
        """Inherit each gene from the dominant parent with ``dominance_prob``."""
        return {
            name: (
                dominant.hyperparams[name]
                if rng.random() < self.dominance_prob
                else recessive.hyperparams[name]
            )
            for name in self.space.names()
        }

    def mutate(self, params: dict[str, Any], rng) -> dict[str, Any]:
        """Delegate mutation to the search space."""
        return self.space.mutate(
            params,
            rng,
            self.mutation_fraction,
            self.mutation_intensity,
            self.resample_prob,
        )

    def evolve(self, members: list[PopulationMember], rng) -> list[PopulationMember]:
        """Elitist exploit/explore step; same length and ids as the input."""
        survivors = self.select(members)
        survivor_ids = {m.id for m in survivors}
        for slot in members:
            if slot.id in survivor_ids:
                continue
            if self.top_k == 1:
                dominant = recessive = survivors[0]
            else:
                pair = rng.choice(len(survivors), size=2, replace=False)
                dominant, recessive = self.rank([survivors[int(pair[0])], survivors[int(pair[1])]])
            child_params = self.mutate(self.crossover(dominant, recessive, rng), rng)
            slot.inherit_from(dominant, recessive, child_params)
        return sorted(members, key=lambda m: m.id)
