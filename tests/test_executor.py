"""Tests for the executor abstraction (sequential + joblib)."""

from __future__ import annotations

import numpy as np
import pytest

from pbt_xgb.executor import (
    Executor,
    JoblibExecutor,
    SequentialExecutor,
    make_executor,
)


def _weighted_sum(*, vec: np.ndarray, weight: float, offset: int) -> float:
    """Module-level so joblib can pickle it across processes."""
    return float(np.sum(vec) * weight + offset)


def _arg_dicts() -> list[dict]:
    return [
        {"vec": np.arange(5, dtype=np.float32), "weight": 2.0, "offset": 1},
        {"vec": np.ones(3, dtype=np.float32), "weight": 0.5, "offset": -2},
        {"vec": np.array([10.0, -10.0], dtype=np.float32), "weight": 3.0, "offset": 7},
    ]


def _expected(args: list[dict]) -> list[float]:
    return [_weighted_sum(**a) for a in args]


def test_sequential_preserves_order_and_values() -> None:
    args = _arg_dicts()
    result = SequentialExecutor().map(_weighted_sum, args)
    assert result == _expected(args)


def test_joblib_matches_sequential() -> None:
    args = _arg_dicts()
    seq = SequentialExecutor().map(_weighted_sum, args)
    par = JoblibExecutor(n_jobs=2).map(_weighted_sum, args)
    assert par == seq


def test_empty_arg_list_returns_empty() -> None:
    assert SequentialExecutor().map(_weighted_sum, []) == []
    assert JoblibExecutor(n_jobs=2).map(_weighted_sum, []) == []


def test_make_executor_types() -> None:
    assert isinstance(make_executor("sequential"), SequentialExecutor)
    assert isinstance(make_executor("joblib", n_jobs=2), JoblibExecutor)
    # defaults to joblib
    assert isinstance(make_executor(), JoblibExecutor)


def test_make_executor_subclasses_executor() -> None:
    assert isinstance(make_executor("sequential"), Executor)
    assert isinstance(make_executor("joblib", n_jobs=2), Executor)


def test_make_executor_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown"):
        make_executor("nope")


def test_base_executor_map_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        Executor().map(_weighted_sum, _arg_dicts())
