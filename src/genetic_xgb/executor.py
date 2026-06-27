"""Pluggable execution backends for evaluating population members in parallel."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from joblib import Parallel, delayed


class Executor:
    """Maps a function over a sequence of keyword-argument dicts, in order."""

    def map(self, fn: Callable, arg_dicts: list[dict[str, Any]]) -> list:
        raise NotImplementedError


class SequentialExecutor(Executor):
    """Runs each call in-process, one after another."""

    def map(self, fn: Callable, arg_dicts: list[dict[str, Any]]) -> list:
        return [fn(**a) for a in arg_dicts]


class JoblibExecutor(Executor):
    """Runs calls in parallel via joblib."""

    def __init__(self, n_jobs: int = -1) -> None:
        self.n_jobs = n_jobs

    def map(self, fn: Callable, arg_dicts: list[dict[str, Any]]) -> list:
        return list(Parallel(n_jobs=self.n_jobs)(delayed(fn)(**a) for a in arg_dicts))


def make_executor(kind: str = "joblib", n_jobs: int = -1) -> Executor:
    if kind == "sequential":
        return SequentialExecutor()
    if kind == "joblib":
        return JoblibExecutor(n_jobs=n_jobs)
    raise ValueError(f"unknown executor kind {kind!r}; choose 'sequential' or 'joblib'")
