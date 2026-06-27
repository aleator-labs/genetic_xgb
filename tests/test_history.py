"""Integration tests for the History logger."""

from __future__ import annotations

import pandas as pd

from genetic_xgb.history import History
from genetic_xgb.member import PopulationMember


def test_record_one_row_per_member_with_lineage_and_genes() -> None:
    members = [
        PopulationMember(id=0, hyperparams={"learning_rate": 0.1, "max_depth": 4}),
        PopulationMember(
            id=1,
            hyperparams={"learning_rate": 0.2, "max_depth": 6},
            score=0.5,
            n_rounds=10,
            parents=(0, 2),
        ),
    ]
    history = History()
    history.record(0, members)
    frame = history.to_frame()

    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 2
    for column in ("generation", "member_id", "score", "n_rounds", "best_iteration", "parents"):
        assert column in frame.columns
    assert "learning_rate" in frame.columns
    assert "max_depth" in frame.columns

    row = frame[frame["member_id"] == 1].iloc[0]
    assert row["generation"] == 0
    assert row["score"] == 0.5
    assert row["n_rounds"] == 10
    assert row["parents"] == (0, 2)
    assert row["learning_rate"] == 0.2


def test_multiple_generations_accumulate() -> None:
    members = [PopulationMember(id=0, hyperparams={"gamma": 1.0})]
    history = History()
    history.record(0, members)
    history.record(1, members)
    frame = history.to_frame()
    assert len(frame) == 2
    assert sorted(frame["generation"].tolist()) == [0, 1]


def test_empty_history_yields_empty_frame() -> None:
    frame = History().to_frame()
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 0
