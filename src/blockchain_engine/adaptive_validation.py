"""Adaptive Lightweight Validation (ALV) policy mechanics (frozen V1.1-A).

Implements the frozen band-to-checks mapping and fail-safe rule only. This is
ENGINE MECHANICS: it maps a risk level to a check set + validator count and
runs the deterministic local quorum for the HIGH path. It does not generate
governed risk score: V1.1-B uses labeled synthetic fixtures only, and nothing
here is ever presented as an AI prediction.

Frozen policy:
    LOW    -> c1..c4  , 1 validator
    MEDIUM -> c1..c6  , 1 validator
    HIGH   -> c1..c8  , 3 validators
    fail-safe: missing or malformed risk reference escalates to MEDIUM
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from .block import Block
from .errors import BlockchainValidationError
from .validation import (
    RISK_LEVELS,
    AuthorizedValidator,
    ValidationContext,
    ValidationResult,
    validate_block,
)

ALV_VALIDATOR_COUNTS: dict[str, int] = {
    "LOW": 1,
    "MEDIUM": 1,
    "HIGH": 3,
}

ALV_BAND_TO_CHECKS: dict[str, tuple[str, ...]] = {
    "LOW": ("c1", "c2", "c3", "c4"),
    "MEDIUM": ("c1", "c2", "c3", "c4", "c5", "c6"),
    "HIGH": ("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"),
}

FAIL_SAFE_LEVEL = "MEDIUM"
FAIL_SAFE_REASON = "fail_safe: missing or malformed risk reference escalates to MEDIUM"


def resolve_adaptive_validation_level(
    block: Block,
    *,
    chain: Sequence[Block] | None = None,
    order_id: str | None = None,
) -> tuple[str, str]:
    """Resolve the ALV validation level for a block under the frozen policy.

    Returns ``(level, note)``. The declared ``risk_level`` of a well-formed,
    digest-consistent ``ai_risk_reference`` selects the band. Any missing or
    malformed reference escalates to MEDIUM (fail-safe); unknown risk is never
    silently mapped to LOW.
    """
    reference = block.ai_risk_reference
    if reference is None or not isinstance(reference, dict):
        return FAIL_SAFE_LEVEL, FAIL_SAFE_REASON
    ctx = ValidationContext.for_block(block, chain=chain, order_id=order_id)
    from .validation import risk_record_digest

    required = {
        "order_id",
        "risk_score",
        "risk_level",
        "model_configuration_provenance",
        "hybrid_k13_provenance",
        "generation_stage_provenance",
        "record_digest",
    }
    if not required.issubset(reference.keys()):
        return FAIL_SAFE_LEVEL, FAIL_SAFE_REASON
    try:
        if risk_record_digest(reference) != reference["record_digest"]:
            return FAIL_SAFE_LEVEL, FAIL_SAFE_REASON
    except (KeyError, TypeError):
        return FAIL_SAFE_LEVEL, FAIL_SAFE_REASON
    if reference.get("order_id") != ctx.order_id:
        return FAIL_SAFE_LEVEL, FAIL_SAFE_REASON
    level = reference.get("risk_level")
    if level not in RISK_LEVELS:
        return FAIL_SAFE_LEVEL, FAIL_SAFE_REASON
    return level, "declared risk_level in ai_risk_reference"


def _band_config(level: str) -> tuple[tuple[str, ...], int]:
    level = level.upper()
    checks = ALV_BAND_TO_CHECKS.get(level)
    if checks is None:
        raise BlockchainValidationError(f"unknown ALV band {level!r}")
    return checks, ALV_VALIDATOR_COUNTS[level]


def validate_adaptive_alv(
    block: Block,
    *,
    chain: Sequence[Block] | None = None,
    order_id: str | None = None,
) -> ValidationResult:
    """Run ALV over a block: resolve band, apply checks, honor validator count."""
    level, note = resolve_adaptive_validation_level(block, chain=chain, order_id=order_id)
    checks, validator_count = _band_config(level)
    return validate_with_quorum(
        block,
        checks,
        validator_count=validator_count,
        chain=chain,
        order_id=order_id,
        resolved_level=level,
        note=note,
    )


def validate_with_quorum(
    block: Block,
    checks: Sequence[str],
    *,
    validator_count: int,
    chain: Sequence[Block] | None = None,
    order_id: str | None = None,
    resolved_level: str | None = None,
    note: str = "",
    contexts: Sequence[ValidationContext] | None = None,
) -> ValidationResult:
    """Deterministic local quorum over ``validator_count`` identical validators.

    For HIGH (3 validators) the simulation re-runs the identical deterministic
    validators over the same input; identical inputs therefore cannot disagree.
    If any validator's verdict differs (only possible via an injected input
    discrepancy), the block is REJECTED with the frozen discrepancy reason.
    """
    if contexts is None:
        ctx = ValidationContext.for_block(block, chain=chain, order_id=order_id)
        contexts = [ctx] * validator_count
    if not contexts:
        raise BlockchainValidationError("quorum requires at least one validator context")

    results = [
        validate_block(block, checks, chain=ctx.chain, order_id=ctx.order_id)
        for ctx in contexts
    ]

    reference_verdict = results[0]
    disagreement = [r for r in results if r != reference_verdict]

    risk_level = (
        block.ai_risk_reference["risk_level"]
        if isinstance(block.ai_risk_reference, dict)
        and block.ai_risk_reference.get("risk_level") in RISK_LEVELS
        else None
    )

    if disagreement:
        return ValidationResult(
            accepted=False,
            risk_level=risk_level,
            validation_level=resolved_level,
            applied_checks=tuple(checks),
            validator_count=validator_count,
            reason_codes=("validator_discrepancy",),
            discrepancy=True,
            note="on any input discrepancy the block is REJECTED with reason",
        )

    accepted = reference_verdict.accepted
    return ValidationResult(
        accepted=accepted,
        risk_level=risk_level,
        validation_level=resolved_level,
        applied_checks=tuple(checks),
        passed_checks=reference_verdict.passed_checks,
        failed_checks=reference_verdict.failed_checks,
        validator_count=validator_count,
        reason_codes=reference_verdict.reason_codes,
        reasons=reference_verdict.reasons,
        discrepancy=False,
        note=note,
    )


def synthetic_risk_reference(
    order_id: str,
    risk_score: float,
    *,
    model_configuration_provenance: str = "SYNTHETIC_TEST_FIXTURE",
    hybrid_k13_provenance: str = "SYNTHETIC_TEST_FIXTURE",
    generation_stage: str = "SYNTHETIC_TEST_FIXTURE",
) -> dict[str, Any]:
    """Build a labeled risk reference for unit tests only.

    ``risk_source="SYNTHETIC_TEST_FIXTURE"`` is recorded in the provenance
    fields; this record is explicitly NOT an AI prediction and is never
    consumed as governed evidence. ``risk_level`` follows the frozen
    preregistered mapping and ``generation_stage`` is labeled synthetic.
    """
    from .validation import risk_record_digest

    if isinstance(risk_score, bool) or not (0.0 <= risk_score <= 1.0):
        raise BlockchainValidationError("risk_score must be a float in [0,1]")

    level = risk_band_from_score(risk_score)
    record = {
        "order_id": order_id,
        "risk_score": float(risk_score),
        "risk_level": level,
        "model_configuration_provenance": model_configuration_provenance,
        "hybrid_k13_provenance": hybrid_k13_provenance,
        "generation_stage_provenance": generation_stage,
    }
    record["record_digest"] = risk_record_digest(record)
    return record


def risk_band_from_score(risk_score: float) -> str:
    """Frozen preregistered mapping: <0.3333 LOW; [0.3333,0.6667) MEDIUM; else HIGH."""
    score = float(risk_score)
    if score < 0.3333:
        return "LOW"
    if score < 0.6667:
        return "MEDIUM"
    return "HIGH"