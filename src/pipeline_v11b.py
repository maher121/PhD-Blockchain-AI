"""V1.1-B stage pipeline/verification entry point (lightweight conformance).

This is a CONFORMANCE VERIFIER, not a scientific experiment. It constructs a
few tiny synthetic per-order chains with fixed fixtures, checks deterministic
hashing, genesis, chaining, event ordering, the frozen c1-c8 check set, and
the ALV LOW/MEDIUM/HIGH + fail-safe + quorum mechanics, then emits an
in-memory summary.

It MUST NOT and DOES NOT: read DataCo, benchmark performance, write scientific
result tables, access AI, or execute E01-E10.
"""

from __future__ import annotations

import sys
from typing import Any

from src.blockchain_engine import ALV_BAND_TO_CHECKS, ALV_VALIDATOR_COUNTS
from src.blockchain_engine.adaptive_validation import synthetic_risk_reference, validate_adaptive_alv
from src.blockchain_engine.block import FIXED_EVENT_SEQUENCE
from src.blockchain_engine.order_chain import OrderChain


def _fixed_payload() -> str:
    from src.blockchain_engine import sha256_hex

    return sha256_hex("v11b-synthetic-payload-fixture")


def run_conformance() -> dict[str, Any]:
    """Build tiny synthetic chains and verify engine mechanics deterministically.

    Returns an in-memory conformance summary dict (never persisted as a
    scientific result artifact).
    """
    summary: dict[str, Any] = {
        "stage": "V1.1-B",
        "kind": "IMPLEMENTATION_CONFORMANCE",
        "checks": {},
    }

    # --- canonical + hash determinism ----------------------------------- #
    from src.blockchain_engine import canonical_bytes, sha256_hex

    vector = {"order_id": "1001", "event_type": "ORDER_CREATED", "value": 1}
    bytes_a = canonical_bytes(vector)
    bytes_b = canonical_bytes(dict(vector))
    assert bytes_a == bytes_b
    assert sha256_hex(vector) == sha256_hex(dict(vector))
    assert len(sha256_hex(vector)) == 64
    summary["checks"]["canonical_determinism"] = sha256_hex(vector)

    # --- genesis determinism -------------------------------------------- #
    chain_a = OrderChain("1001")
    chain_b = OrderChain("1001")
    g_a = chain_a.seed_genesis("2024-01-10T08:00:00Z")
    g_b = chain_b.seed_genesis("2024-01-10T08:00:00Z")
    assert g_a.block_hash == g_b.block_hash
    assert g_a.block_index == 0
    assert g_a.event_type == "ORDER_GENESIS"
    assert g_a.previous_hash is None
    summary["checks"]["genesis"] = g_a.block_hash

    # --- build a complete synthetic order chain ------------------------- #
    chain = OrderChain("1001")
    chain.seed_genesis("2024-01-10T08:00:00Z")
    payload = _fixed_payload()
    chain.append_event("ORDER_CREATED", "2024-01-10T08:00:05Z", payload)
    chain.append_event("SHIPMENT_RECORDED", "2024-01-12T09:30:00Z", payload)
    chain.append_event("DELIVERY_STATUS_RECORDED", "2024-01-15T14:00:00Z", payload)
    risk_ref = synthetic_risk_reference("1001", 0.8)  # HIGH (0.8 >= 0.6667)
    chain.append_event(
        "AI_RISK_ASSESSED",
        "2024-01-15T14:00:05Z",
        payload,
        ai_risk_reference=risk_ref,
    )
    blocks = chain.blocks
    assert [b.event_type for b in blocks] == list(FIXED_EVENT_SEQUENCE)
    assert all(b.block_index == i for i, b in enumerate(blocks))
    assert all(b.order_id == "1001" for b in blocks)
    assert all(
        i == 0 or b.previous_hash == blocks[i - 1].block_hash
        for i, b in enumerate(blocks)
    )
    summary["checks"]["chain"] = [b.block_hash for b in blocks]

    # --- validation core checks c1-c8 ----------------------------------- #
    from src.blockchain_engine.validation import (
        check_c1_schema,
        check_c2_self_hash,
        check_c3_previous_hash_link,
        check_c4_timestamp_order,
        check_c5_order_integrity,
        check_c6_ai_reference_present,
        check_c7_ai_reference_digest,
        check_c8_chain_prefix,
        ValidationContext,
    )
    ctx = ValidationContext.for_block(blocks[-1], chain=blocks, order_id="1001")
    core = [
        check_c1_schema(blocks[-1], ctx),
        check_c2_self_hash(blocks[-1], ctx),
        check_c3_previous_hash_link(blocks[-1], ctx),
        check_c4_timestamp_order(blocks[-1], ctx),
        check_c5_order_integrity(blocks[-1], ctx),
        check_c6_ai_reference_present(blocks[-1], ctx),
        check_c7_ai_reference_digest(blocks[-1], ctx),
        check_c8_chain_prefix(blocks[-1], ctx),
    ]
    assert all(r.passed for r in core)
    summary["checks"]["c1_c8"] = {r.check_id: r.passed for r in core}

    # --- ALV paths: LOW / MEDIUM / HIGH + fail-safe ---------------------- #
    results = {}
    for band, score in (("LOW", 0.1), ("MEDIUM", 0.5), ("HIGH", 0.9)):
        order_id = {"LOW": "1101", "MEDIUM": "1102", "HIGH": "1001"}[band]
        test_chain = OrderChain(order_id)
        test_chain.seed_genesis("2024-02-01T08:00:00Z")
        test_chain.append_event("ORDER_CREATED", "2024-02-01T08:00:05Z", payload)
        test_chain.append_event("SHIPMENT_RECORDED", "2024-02-01T08:00:06Z", payload)
        test_chain.append_event("DELIVERY_STATUS_RECORDED", "2024-02-01T08:00:08Z", payload)
        ref = synthetic_risk_reference(order_id, score)
        test_chain.append_event(
            "AI_RISK_ASSESSED", "2024-02-01T08:00:10Z", payload, ai_risk_reference=ref
        )
        head = test_chain.last_block
        res = validate_adaptive_alv(head, chain=test_chain.blocks, order_id=order_id)
        assert res.accepted is True, f"{band} block not accepted"
        assert res.validation_level == band, f"{band} level mismatch"
        assert res.validator_count == ALV_VALIDATOR_COUNTS[band]
        assert tuple(res.applied_checks) == tuple(ALV_BAND_TO_CHECKS[band])
        results[band] = {
            "level": res.validation_level,
            "validator_count": res.validator_count,
            "applied_checks": list(res.applied_checks),
        }

    fail_safe_chain = OrderChain("1201")
    fail_safe_chain.seed_genesis("2024-03-01T08:00:00Z")
    fail_safe_chain.append_event("ORDER_CREATED", "2024-03-01T08:00:05Z", payload)
    fail_safe_chain.append_event("SHIPMENT_RECORDED", "2024-03-01T08:00:06Z", payload)
    fail_safe_chain.append_event("DELIVERY_STATUS_RECORDED", "2024-03-01T08:00:08Z", payload)
    fail_safe_chain.append_event("AI_RISK_ASSESSED", "2024-03-01T08:00:10Z", payload, ai_risk_reference=None)
    fail_safe_res = validate_adaptive_alv(
        fail_safe_chain.last_block, chain=fail_safe_chain.blocks, order_id="1201"
    )
    assert fail_safe_res.validation_level == "MEDIUM", "fail-safe must escalate to MEDIUM"
    assert fail_safe_res.accepted is False, "malformed AI block must be rejected"
    results["FAIL_SAFE"] = {
        "level": fail_safe_res.validation_level,
        "accepted": fail_safe_res.accepted,
        "reason": fail_safe_res.note,
    }
    summary["checks"]["alv"] = results

    return summary


def main() -> None:
    summary = run_conformance()
    print(f"V1.1-B conformance OK for stage {summary['stage']}")
    for name, value in summary["checks"].items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()