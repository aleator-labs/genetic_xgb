"""Pure, picklable training step for population-based training.

This module OWNS :func:`train_step`. The step trains an XGBoost booster on the
*train* split only (warm-starting from ``booster_bytes`` when supplied) and
scores the resulting booster on the *validation* split.

HARD INVARIANT: the validation arrays are NEVER passed to ``xgb.train`` (no
``dtrain`` built from them, no ``evals``). Training uses the train split only,
so there is no information leakage from validation into the model.
"""

from __future__ import annotations

import xgboost as xgb

from .metrics import MetricSpec


def train_step(
    *,
    booster_bytes: bytes | None,
    hyperparams: dict,
    train: tuple,
    val: tuple,
    step_rounds: int,
    metric: MetricSpec,
    base_params: dict,
    seed: int,
) -> dict:
    """Train ``step_rounds`` boosting rounds and score on the validation split.

    Returns a dict with ``booster_bytes`` (raw model), ``fitness`` (the metric
    recomputed on the validation predictions), and ``n_rounds`` (total boosted
    rounds after this step).
    """
    X_train, y_train = train  # noqa: N806
    X_val, y_val = val  # noqa: N806

    params = {**base_params, **hyperparams, "seed": seed}
    dtrain = xgb.DMatrix(X_train, label=y_train)
    prev = None
    if booster_bytes is not None:
        prev = xgb.Booster()
        prev.load_model(bytearray(booster_bytes))

    booster = xgb.train(params, dtrain, num_boost_round=step_rounds, xgb_model=prev)
    proba = booster.predict(xgb.DMatrix(X_val))
    fitness = metric.score(y_val, proba)
    return {
        "booster_bytes": bytes(booster.save_raw()),
        "fitness": float(fitness),
        "n_rounds": booster.num_boosted_rounds(),
    }
