"""Population member dataclass for the genetic-algorithm estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import xgboost as xgb


@dataclass
class PopulationMember:
    """A single individual in the population (its genes + warm-start booster)."""

    id: int
    hyperparams: dict
    booster_bytes: bytes | None = None
    score: float | None = None
    n_rounds: int = 0
    best_iteration: int | None = None
    parents: tuple[int, int] | None = None

    def save_booster(self, booster: xgb.Booster) -> None:
        """Serialize a fitted booster into ``booster_bytes``."""
        self.booster_bytes = bytes(booster.save_raw())

    def load_booster(self) -> xgb.Booster | None:
        """Deserialize ``booster_bytes`` back into a Booster (None if empty)."""
        if self.booster_bytes is None:
            return None
        booster = xgb.Booster()
        booster.load_model(bytearray(self.booster_bytes))
        return booster

    def inherit_from(
        self,
        dominant: PopulationMember,
        recessive: PopulationMember,
        hyperparams: dict[str, Any],
    ) -> None:
        """Become offspring: warm-start from the dominant parent, reset fitness."""
        self.booster_bytes = dominant.booster_bytes
        self.hyperparams = hyperparams
        self.n_rounds = dominant.n_rounds
        self.best_iteration = dominant.best_iteration
        self.parents = (dominant.id, recessive.id)
        self.score = None
