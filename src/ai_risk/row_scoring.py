"""Deterministic row-level attack-risk score combination.

Freezes the V1.1-D row scoring rule: arithmetic mean of the five frozen-seed
member ``predict_proba`` scores for the ``is_attack=1`` class, taken in the
canonical ascending seed order. Operates on plain floats only, so it can be
unit-tested with synthetic fixtures and has zero dependency on fitted models.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


class RowScoreError(ValueError):
    """Raised when a row score violates the frozen ensemble contract."""


FROZEN_MEMBER_SEEDS = (42, 43, 44, 45, 46)
FROZEN_MEMBER_COUNT = len(FROZEN_MEMBER_SEEDS)

SCORE_MIN = 0.0
SCORE_MAX = 1.0


def validate_ensemble_scores(scores: Sequence[float]) -> tuple[float, ...]:
    """Validate the five member scores: count, finiteness, and closed range.

    Each member score is an uncalibrated model-derived value in [0,1]; NaN and
    +/-Infinity are rejected because rounding/clipping are frozen off and the
    downstream digest must stay deterministic.
    """
    values = tuple(float(value) for value in scores)
    if len(values) != FROZEN_MEMBER_COUNT:
        raise RowScoreError(
            f"ensemble requires exactly {FROZEN_MEMBER_COUNT} member scores, "
            f"got {len(values)}"
        )
    for seed, value in zip(FROZEN_MEMBER_SEEDS, values):
        if math.isnan(value):
            raise RowScoreError(f"member score for seed {seed} is NaN")
        if math.isinf(value):
            raise RowScoreError(f"member score for seed {seed} is infinite")
        if value < SCORE_MIN or value > SCORE_MAX:
            raise RowScoreError(
                f"member score for seed {seed} is out of the frozen range "
                f"[{SCORE_MIN}, {SCORE_MAX}]: {value}"
            )
    return values


def combine_member_scores(
    scores: Sequence[float],
    *,
    in_frozen_seed_order: bool = True,
) -> float:
    """Return the arithmetic mean of the five member scores.

    ``in_frozen_seed_order`` documents the contract (ascending seed order); it
    defaults to True and passing False is rejected because member ordering must
    not become an accidental modeling choice.
    """
    if not in_frozen_seed_order:
        raise RowScoreError(
            "member scores must be combined in the frozen ascending seed order"
        )
    values = validate_ensemble_scores(scores)
    return sum(values) / float(len(values))