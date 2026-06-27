"""Shared, seeded, real-dataset fixtures for the test suite (no mocking)."""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split

SEED = 42

# A train/validation split of real arrays. Fields are plain numpy arrays.
Split = namedtuple("Split", ["X_train", "y_train", "X_val", "y_val", "n_classes"])


def _split(X, y) -> Split:  # noqa: N803
    X_train, X_val, y_train, y_val = train_test_split(  # noqa: N806
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    return Split(X_train, y_train, X_val, y_val, int(np.unique(y).size))


@pytest.fixture
def binary_data() -> Split:
    """Real binary classification data (Wisconsin breast cancer, 569 rows)."""
    data = load_breast_cancer()
    return _split(np.asarray(data.data, dtype=np.float32), np.asarray(data.target))


@pytest.fixture
def multiclass_data() -> Split:
    """Real 3-class classification data (iris)."""
    data = load_iris()
    return _split(np.asarray(data.data, dtype=np.float32), np.asarray(data.target))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)
