"""Frozen V1.1-A risk threshold bands for governed order risk levels.

The bands are preregistered operational validation-intensity bands (LOW,
MEDIUM, HIGH), are not calibrated probability categories, and must never be
tuned on validation or test outcomes.
"""

from __future__ import annotations

import math


class RiskLevelError(ValueError):
    """Raised when a score cannot be mapped to a governed risk level."""


RISK_LEVEL_LOW = "LOW"
RISK_LEVEL_MEDIUM = "MEDIUM"
RISK_LEVEL_HIGH = "HIGH"

#: Frozen V1.1-A boundaries: LOW < 0.3333 <= MEDIUM < 0.6667 <= HIGH.
LOW_UPPER_EXCLUSIVE = 0.3333
MEDIUM_UPPER_EXCLUSIVE = 0.6667

LEVEL_ORDER = (RISK_LEVEL_LOW, RISK_LEVEL_MEDIUM, RISK_LEVEL_HIGH)


def risk_level_for_score(score: float) -> str:
    """Map a validated [0,1] score to the frozen threshold band."""
    value = float(score)
    if math.isnan(value) or math.isinf(value):
        raise RiskLevelError(f"risk score must be finite, got {score!r}")
    if value < 0.0 or value > 1.0:
        raise RiskLevelError(f"risk score out of the frozen [0,1] range: {value!r}")
    if value < LOW_UPPER_EXCLUSIVE:
        return RISK_LEVEL_LOW
    if value < MEDIUM_UPPER_EXCLUSIVE:
        return RISK_LEVEL_MEDIUM
    return RISK_LEVEL_HIGH