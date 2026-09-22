"""Deterministic lightweight validation primitives (frozen c1-c8 contract).

Implements the frozen `AUTHORIZED_VALIDATOR_LIGHTWEIGHT` core check set and the
aggregate validator that reports machine-readable evidence (``check_id``,
``passed``, ``reason``). All checks are deterministic pure functions of the
block plus an explicit chain context; no networking, no multiprocessing, no
remote nodes, no probabilistic acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .block import (
    BLOCK_FIELDS,
    EVENT_TYPES,
    FIXED_EVENT_SEQUENCE,
    SCHEMA_VERSION,
    VALIDATION_POLICIES,
    Block,
    block_hash,
)
from .canonical import (
    canonical_order_id,
    is_canonical_timestamp,
    is_sha256_hex,
    sha256_hex,
)
from .errors import BlockchainValidationError

RISK_LEVELS = ("LOW", "MEDIUM", "HIGH")

TOTAL_CHECK_IDS = ("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8")


@dataclass(frozen=True)
class CheckResult:
    """Deterministic, machine-readable evidence for one validation check."""

    check_id: str
    passed: bool
    reason: str


@dataclass(frozen=True)
class ValidationContext:
    """Explicit context a check needs outside the block itself.

    * ``chain``: the full per-order chain up to and including ``block``.
    * ``order_id``: the expected canonical chain identity (genesis anchor).
    """

    chain: tuple[Block, ...]
    order_id: str

    @classmethod
    def for_block(
        cls, block: Block, *, chain: Sequence[Block] | None = None, order_id: str | None = None
    ) -> "ValidationContext":
        if chain is None:
            chain = (block,)
        blocks = tuple(chain)
        if not blocks or blocks[-1].block_index != block.block_index:
            blocks = tuple(blocks) + (block,)
        resolved_order_id = canonical_order_id(order_id) if order_id is not None else blocks[0].order_id
        return cls(chain=blocks, order_id=resolved_order_id)


# --------------------------------------------------------------------------- #
# Individual checks c1-c8
# --------------------------------------------------------------------------- #
def check_c1_schema(block: Block, ctx: ValidationContext) -> CheckResult:
    """c1: schema conformance."""
    problems: list[str] = []
    if block.schema_version != SCHEMA_VERSION:
        problems.append(f"schema_version {block.schema_version!r}")
    if block.order_id != canonical_order_id(block.order_id):
        problems.append("order_id not canonical")
    if not isinstance(block.block_index, int) or block.block_index < 0:
        problems.append(f"block_index {block.block_index!r}")
    if block.event_type not in EVENT_TYPES:
        problems.append(f"event_type {block.event_type!r}")
    if not is_canonical_timestamp(block.event_timestamp):
        problems.append(f"event_timestamp {block.event_timestamp!r}")
    if not is_sha256_hex(block.payload_digest):
        problems.append("payload_digest not sha256 hex")
    if block.payload_ref is not None and not isinstance(block.payload_ref, str):
        problems.append("payload_ref not string")
    if block.ai_risk_reference is not None and not isinstance(block.ai_risk_reference, dict):
        problems.append("ai_risk_reference not object")
    if block.validation_policy not in VALIDATION_POLICIES:
        problems.append(f"validation_policy {block.validation_policy!r}")
    if block.previous_hash is not None and not is_sha256_hex(block.previous_hash):
        problems.append("previous_hash not sha256 hex/null")
    if block.block_index == 0 and block.previous_hash is not None:
        problems.append("previous_hash not null at genesis")
    if block.block_index > 0 and block.previous_hash is None:
        problems.append("previous_hash null for non-genesis")
    if not is_sha256_hex(block.block_hash):
        problems.append("block_hash not sha256 hex")
    reason = "ok" if not problems else "schema violation: " + "; ".join(problems)
    return CheckResult("c1", passed=not problems, reason=reason)


def check_c2_self_hash(block: Block, ctx: ValidationContext) -> CheckResult:
    """c2: self block_hash recomputation."""
    recomputed = block_hash(block)
    if recomputed == block.block_hash:
        return CheckResult("c2", True, "ok")
    return CheckResult("c2", False, "block_hash mismatch")


def check_c3_previous_hash_link(block: Block, ctx: ValidationContext) -> CheckResult:
    """c3: previous_hash linkage (strict link to the immediate predecessor)."""
    if block.block_index == 0:
        ok = block.previous_hash is None
        return CheckResult("c3", ok, "ok" if ok else "genesis must have null previous_hash")
    previous = ctx.chain[block.block_index - 1]
    ok = block.previous_hash == previous.block_hash
    return CheckResult("c3", ok, "ok" if ok else "previous_hash link mismatch")


def check_c4_timestamp_order(block: Block, ctx: ValidationContext) -> CheckResult:
    """c4: event timestamp non-decreasing per order."""
    if block.block_index == 0:
        return CheckResult("c4", True, "ok")
    previous = ctx.chain[block.block_index - 1]
    ok = previous.event_timestamp <= block.event_timestamp
    return CheckResult("c4", ok, "ok" if ok else "timestamp not non-decreasing")


def check_c5_order_integrity(block: Block, ctx: ValidationContext) -> CheckResult:
    """c5: order integrity (canonical order_id chain-prefix match)."""
    ok = block.order_id == ctx.order_id and block.order_id == ctx.chain[0].order_id
    return CheckResult("c5", ok, "ok" if ok else "order_id chain-prefix mismatch")


def check_c6_ai_reference_present(block: Block, ctx: ValidationContext) -> CheckResult:
    """c6: ai_risk_reference present for AI-linked event blocks."""
    if block.event_type != "AI_RISK_ASSESSED":
        return CheckResult("c6", True, "not applicable for this event type")
    ok = isinstance(block.ai_risk_reference, dict) and bool(block.ai_risk_reference)
    return CheckResult("c6", ok, "ok" if ok else "missing ai_risk_reference on AI-linked block")


def risk_record_digest(record: Mapping[str, Any]) -> str:
    """SHA-256 over the canonical serialization of the six preceding fields."""
    return sha256_hex(
        {
            "order_id": record["order_id"],
            "risk_score": record["risk_score"],
            "risk_level": record["risk_level"],
            "model_configuration_provenance": record["model_configuration_provenance"],
            "hybrid_k13_provenance": record["hybrid_k13_provenance"],
            "generation_stage_provenance": record["generation_stage_provenance"],
        }
    )


def check_c7_ai_reference_digest(block: Block, ctx: ValidationContext) -> CheckResult:
    """c7: ai_risk_reference digest re-verification."""
    reference = block.ai_risk_reference
    if block.event_type != "AI_RISK_ASSESSED":
        return CheckResult("c7", True, "not applicable for this event type")
    if not isinstance(reference, dict):
        return CheckResult("c7", False, "missing ai_risk_reference object")
    required = {
        "order_id",
        "risk_score",
        "risk_level",
        "model_configuration_provenance",
        "hybrid_k13_provenance",
        "generation_stage_provenance",
        "record_digest",
    }
    missing = required.difference(reference.keys())
    if missing:
        return CheckResult("c7", False, f"risk record missing fields: {sorted(missing)}")
    try:
        recomputed = risk_record_digest(reference)
    except (KeyError, TypeError):
        return CheckResult("c7", False, "risk record fields not canonicalizable")
    if recomputed != reference["record_digest"]:
        return CheckResult("c7", False, "risk record record_digest mismatch")
    score = reference["risk_score"]
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not (0.0 <= float(score) <= 1.0):
        return CheckResult("c7", False, "risk_score out of [0,1]")
    if reference["risk_level"] not in RISK_LEVELS:
        return CheckResult("c7", False, "risk_level not in LOW/MEDIUM/HIGH")
    if reference["order_id"] != ctx.order_id:
        return CheckResult("c7", False, "risk record order_id != chain order_id")
    for key in (
        "model_configuration_provenance",
        "hybrid_k13_provenance",
        "generation_stage_provenance",
    ):
        if not isinstance(reference[key], str) or not reference[key]:
            return CheckResult("c7", False, f"{key} empty")
    return CheckResult("c7", True, "ok")


def check_c8_chain_prefix(block: Block, ctx: ValidationContext) -> CheckResult:
    """c8: chain prefix re-verification (prior blocks hash recheck, incl. links)."""
    if block.block_index != len(ctx.chain) - 1:
        return CheckResult("c8", False, "chain context incomplete")
    failures: list[str] = []
    for index in range(0, block.block_index + 1):
        candidate = ctx.chain[index]
        if candidate.block_index != index:
            failures.append(f"index gap at position {index}")
        if block_hash(candidate) != candidate.block_hash:
            failures.append(f"hash mismatch at index {index}")
        if index > 0 and candidate.previous_hash != ctx.chain[index - 1].block_hash:
            failures.append(f"broken link at index {index}")
    if failures:
        return CheckResult("c8", False, "; ".join(failures))
    return CheckResult("c8", True, "ok")


CHECK_FUNCTIONS: dict[str, Any] = {
    "c1": check_c1_schema,
    "c2": check_c2_self_hash,
    "c3": check_c3_previous_hash_link,
    "c4": check_c4_timestamp_order,
    "c5": check_c5_order_integrity,
    "c6": check_c6_ai_reference_present,
    "c7": check_c7_ai_reference_digest,
    "c8": check_c8_chain_prefix,
}


def is_valid_check_list(checks: Iterable[str] | str) -> bool:
    checks = (checks,) if isinstance(checks, str) else tuple(checks)
    return all(check in CHECK_FUNCTIONS for check in checks)


# --------------------------------------------------------------------------- #
# Aggregate validator & authorized-validator abstraction
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ValidationResult:
    """Deterministic aggregate verdict with machine-readable evidence."""

    accepted: bool
    risk_level: str | None
    validation_level: str | None
    applied_checks: tuple[str, ...]
    passed_checks: tuple[str, ...] = field(default_factory=tuple)
    failed_checks: tuple[str, ...] = field(default_factory=tuple)
    validator_count: int = 1
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    reasons: tuple[CheckResult, ...] = field(default_factory=tuple)
    discrepancy: bool = False
    note: str = ""


def apply_checks(
    block: Block,
    checks: Sequence[str],
    *,
    chain: Sequence[Block] | None = None,
    order_id: str | None = None,
) -> list[CheckResult]:
    """Run an explicit check set and return per-check machine-readable evidence."""
    ctx = ValidationContext.for_block(block, chain=chain, order_id=order_id)
    for check in checks:
        if check not in CHECK_FUNCTIONS:
            raise BlockchainValidationError(f"unknown check: {check!r}")
    return [CHECK_FUNCTIONS[check](block, ctx) for check in checks]


def validate_block(
    block: Block,
    checks: Sequence[str],
    *,
    chain: Sequence[Block] | None = None,
    order_id: str | None = None,
) -> ValidationResult:
    """Aggregate validator: deterministic verdict list, no probabilistic acceptance."""
    results = apply_checks(block, list(checks), chain=chain, order_id=order_id)
    passed = [r.check_id for r in results if r.passed]
    failed = [r.check_id for r in results if not r.passed]
    accepted = not failed
    level = (
        _declared_risk_level(block)
        if block.ai_risk_reference is not None and isinstance(block.ai_risk_reference, dict)
        else "MEDIUM"
    )
    return ValidationResult(
        accepted=accepted,
        risk_level=level,
        validation_level=None,
        applied_checks=tuple(r.check_id for r in results),
        passed_checks=tuple(passed),
        failed_checks=tuple(failed),
        reason_codes=tuple(r.check_id for r in results if not r.passed),
        reasons=tuple(results),
    )


def _declared_risk_level(block: Block) -> str | None:
    reference = block.ai_risk_reference
    if isinstance(reference, dict) and reference.get("risk_level") in RISK_LEVELS:
        return reference["risk_level"]
    return None


class AuthorizedValidator:
    """AUTHORIZED_VALIDATOR_LIGHTWEIGHT: a local deterministic validator.

    It is a deterministic re-verification abstraction, NOT a distributed
    consensus node: no network, no multiprocessing, no remote nodes.
    """

    name = "AUTHORIZED_VALIDATOR_LIGHTWEIGHT"

    def __init__(self, checks: Sequence[str]) -> None:
        if not is_valid_check_list(checks):
            raise BlockchainValidationError(f"invalid check set: {checks!r}")
        self.checks = tuple(checks)

    def validate(
        self,
        block: Block,
        *,
        chain: Sequence[Block] | None = None,
        order_id: str | None = None,
    ) -> ValidationResult:
        return validate_block(block, self.checks, chain=chain, order_id=order_id)