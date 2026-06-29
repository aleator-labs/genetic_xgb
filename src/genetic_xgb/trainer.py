"""Pure, picklable training step for the genetic-algorithm estimators.

This module OWNS :func:`train_step`. The step trains an XGBoost booster on the
*train* split only (warm-starting from ``booster_bytes`` when supplied) and
scores the resulting booster on the *validation* split.

INVARIANT: trees are fit on **train gradients only**, always. When
``early_stopping_rounds`` is set the validation split is additionally passed as an
``evals`` watchlist so XGBoost can decide *when to stop adding trees* — that is
model selection (round count), not gradient fitting, so the model never learns
from validation labels. When ``early_stopping_rounds is None`` no ``evals`` are
passed and the strict "validation never touches ``xgb.train``" path holds.
"""

from __future__ import annotations

import numpy as np
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
    early_stopping_rounds: int | None = None,
    eval_metric: str | None = None,
    sample_weight: np.ndarray | None = None,
) -> dict:
    """Train up to ``step_rounds`` boosting rounds and score on the validation split.

    With ``early_stopping_rounds`` set, ``step_rounds`` becomes an upper bound: the
    booster stops growing once the validation metric stops improving. The full
    (stopped) booster is kept and used for scoring/prediction.

    Returns a dict with ``booster_bytes`` (raw model), ``fitness`` (the metric
    recomputed on the validation predictions), ``n_rounds`` (total boosted rounds
    after this step), and ``best_iteration`` (where early stopping found the best
    score, or ``None`` when early stopping is disabled).

    When ``sample_weight`` is provided, the training DMatrix is built with those
    per-row weights (``xgb.DMatrix(X_train, label=y_train, weight=sample_weight)``);
    the validation DMatrix is always unweighted.
    """
    X_train, y_train = train  # noqa: N806
    X_val, y_val = val  # noqa: N806

    params = {**base_params, **hyperparams, "seed": seed}
    if eval_metric is not None:
        params["eval_metric"] = eval_metric
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
    dval = xgb.DMatrix(X_val, label=y_val)
    prev = None
    if booster_bytes is not None:
        prev = xgb.Booster()
        prev.load_model(bytearray(booster_bytes))

    kwargs = {}
    if early_stopping_rounds is not None:
        # Validation watchlist drives the STOP decision only; trees fit on train gradients.
        kwargs = {
            "evals": [(dtrain, "train"), (dval, "validation")],
            "early_stopping_rounds": early_stopping_rounds,
            "verbose_eval": False,
        }

    booster = xgb.train(params, dtrain, num_boost_round=step_rounds, xgb_model=prev, **kwargs)

    proba = booster.predict(dval)  # full (stopped) booster — no iteration_range slicing
    fitness = metric.score(y_val, proba)
    best_it = booster.best_iteration if early_stopping_rounds is not None else None
    return {
        "booster_bytes": bytes(booster.save_raw()),
        "fitness": float(fitness),
        "n_rounds": booster.num_boosted_rounds(),
        "best_iteration": best_it,
    }
