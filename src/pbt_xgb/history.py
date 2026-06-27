"""Per-generation logging of population state into a tidy table."""

from __future__ import annotations

import pandas as pd

from pbt_xgb.member import PopulationMember


class History:
    """Collects one row per member per generation for later inspection."""

    def __init__(self) -> None:
        self._rows: list[dict] = []

    def record(self, generation: int, members: list[PopulationMember]) -> None:
        """Append one row per member capturing score, lineage and genes."""
        for member in members:
            row = {
                "generation": generation,
                "member_id": member.id,
                "score": member.score,
                "n_rounds": member.n_rounds,
                "parents": member.parents,
            }
            row.update(member.hyperparams)
            self._rows.append(row)

    def to_frame(self) -> pd.DataFrame:
        """Return the recorded rows as a :class:`pandas.DataFrame`."""
        return pd.DataFrame(self._rows)
