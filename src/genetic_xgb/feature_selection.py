"""Genetic operators for the feature-selection mask.

A feature mask is a 1-D boolean array of length ``n_features``: ``True`` keeps the
column, ``False`` drops it from training and prediction. These helpers mirror the
hyperparameter genetic operators (gene-wise dominance crossover, per-gene mutation)
but act on bits, and always guarantee at least ``min_features`` columns are kept so a
member can never train on zero features.
"""

from __future__ import annotations

import numpy as np


def _ensure_min(mask: np.ndarray, min_features: int, rng: np.random.Generator) -> np.ndarray:
    """Turn on random off-bits until at least ``min_features`` are selected."""
    deficit = min_features - int(mask.sum())
    if deficit > 0:
        off = np.flatnonzero(~mask)
        turn_on = rng.choice(off, size=deficit, replace=False)
        mask[turn_on] = True
    return mask


def sample_mask(
    n_features: int, init_prob: float, rng: np.random.Generator, min_features: int
) -> np.ndarray:
    """Sample an initial mask: each feature included with probability ``init_prob``."""
    mask = rng.random(n_features) < init_prob
    return _ensure_min(mask, min_features, rng)


def crossover_masks(
    dominant: np.ndarray, recessive: np.ndarray, dominance_prob: float, rng: np.random.Generator
) -> np.ndarray:
    """Per-feature gene: take the dominant parent's bit with prob ``dominance_prob``."""
    take_dominant = rng.random(dominant.shape[0]) < dominance_prob
    return np.where(take_dominant, dominant, recessive)


def mutate_mask(
    mask: np.ndarray, flip_rate: float, rng: np.random.Generator, min_features: int
) -> np.ndarray:
    """Flip each bit with probability ``flip_rate``; keep at least ``min_features`` on."""
    flips = rng.random(mask.shape[0]) < flip_rate
    return _ensure_min(mask ^ flips, min_features, rng)
