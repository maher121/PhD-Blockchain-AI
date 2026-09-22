"""V1.1-B core lightweight blockchain engine tests (synthetic fixtures only).

Covers the frozen engine mechanics: canonical serialization, SHA-256 hashing,
deterministic blocks, genesis, per-order chains, event ordering, c1-c8
validation primitives, ALV LOW/MEDIUM/HIGH + fail-safe + quorum mechanics,
mutation/integrity behavior, protocol conformance vs. the frozen V1.1-A
config, and governance (no PoW/mining, no DataCo, no AI, no experiments).

Synthetic risk fixtures use ``risk_source="SYNTHETIC_TEST_FIXTURE"`` and are
never labeled as AI-generated.
"""

from __future__ import annotations

import dataclasses
import json
import hashlib
from pathlib import Path

import pytest
import yaml

from src.blockchain_engine import (
    ALV_BAND_TO_CHECKS,
    ALV_VALIDATOR_COUNTS,
    AuthorizedValidator,
    Block,
    InvalidBlockError,
    InvalidEventTransitionError,
    OrderChain,
    canonical_bytes,
    canonical_json,
    canonical_order_id,
    construct_block,
    construct_genesis_block,
    is_sha256_hex,
    normalize_timestamp_utc,
    resolve_adaptive_validation_level,
    risk_band_from_score,
    risk_record_digest,
    sha256_bytes,
    sha256_hex,
    synthetic_risk_reference,
    timestamps_non_decreasing,
    validate_adaptive_alv,
    validate_block,
    validate_with_quorum,
)
from src.blockchain_engine.block import (
    DEFAULT_VALIDATION_POLICY,
    EVENT_TYPES,
    FIXED_EVENT_SEQUENCE,
    VALIDATION_POLICY_PREFIX,
    VALIDATION_POLICIES,
    genesis_payload_digest,
)
from src.blockchain_engine.validation import (
    CheckResult,
    ValidationContext,
    apply_checks,
    check_c1_schema,
    check_c2_self_hash,
    check_c3_previous_hash_link,
    check_c4_timestamp_order,
    check_c5_order_integrity,
    check_c6_ai_reference_present,
    check_c7_ai_reference_digest,
    check_c8_chain_prefix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "blockchain_v11.yaml"
ENGINE_DIR = PROJECT_ROOT / "src" / "blockchain_engine"


@pytest.fixture(scope="module")
def cfg() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _dig(value: str) -> str:
    return hashlib.sha256(f"v11b:{value}".encode("utf-8")).hexdigest()


def _complete_chain(order_id: str = "1001", score: float = 0.8) -> OrderChain:
    """A complete frozen-sequence chain with a labeled synthetic risk fixture."""
    chain = OrderChain(order_id)
    chain.seed_genesis("2024-01-10T08:00:00Z")
    chain.append_event("ORDER_CREATED", "2024-01-10T08:00:05Z", _dig("created"))
    chain.append_event("SHIPMENT_RECORDED", "2024-01-12T09:30:00Z", _dig("ship"))
    chain.append_event(
        "DELIVERY_STATUS_RECORDED", "2024-01-15T14:00:00Z", _dig("deliv")
    )
    ref = synthetic_risk_reference(order_id, score)
    chain.append_event(
        "AI_RISK_ASSESSED",
        "2024-01-15T14:00:05Z",
        _dig("ai"),
        ai_risk_reference=ref,
    )
    return chain


# ========================================================================== #
# A. Canonicalization
# ========================================================================== #
def test_canonical_sorts_keys():
    assert canonical_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}'


def test_canonical_compact_separators_no_trailing_newline():
    raw = canonical_json({"order_id": "1001", "value": 1})
    assert " " not in raw.replace('":"', "\ufffd").replace(":{", "\ufffd")
    assert raw == '{"order_id":"1001","value":1}'
    assert not raw.endswith("\n")


def test_canonical_utf8_ascii_escaping():
    raw = canonical_bytes({"note": "café"})
    assert raw == b'{"note":"caf\\u00e9"}'


def test_canonical_nan_forbidden():
    with pytest.raises(ValueError):
        canonical_json({"risk_score": float("nan")})


def test_canonical_infinity_forbidden():
    with pytest.raises(ValueError):
        canonical_json({"v": float("inf")})


def test_canonical_equivalent_dicts_stable_bytes():
    a = {"event_type": "ORDER_CREATED", "order_id": "1001", "value": 1}
    b = {"value": 1, "order_id": "1001", "event_type": "ORDER_CREATED"}
    assert canonical_bytes(a) == canonical_bytes(b)


def test_canonical_null_sentinel():
    assert canonical_json({"previous_hash": None}) == '{"previous_hash":null}'


def test_canonical_float_roundtrip():
    assert canonical_json({"risk_score": 0.5}) == '{"risk_score":0.5}'


def test_timestamp_fixed_roundtrip():
    assert normalize_timestamp_utc("2024-01-10T08:00:00Z") == "2024-01-10T08:00:00Z"


def test_timestamp_rejects_naive_datetime():
    import datetime as dt

    with pytest.raises(Exception):
        normalize_timestamp_utc(dt.datetime(2024, 1, 10, 8, 0, 0))


def test_timestamp_rejects_wall_clock_and_offsets():
    for bad in ["2024-01-10T08:00:00+00:00", "2024-01-10 08:00:00Z", "2024-13-10T08:00:00Z"]:
        with pytest.raises(Exception):
            normalize_timestamp_utc(bad)


def test_timestamp_rejects_microseconds():
    import datetime as dt

    aware = dt.datetime(2024, 1, 10, 8, 0, 0, 250000, tzinfo=dt.timezone.utc)
    with pytest.raises(Exception):
        normalize_timestamp_utc(aware)


def test_timestamps_non_decreasing():
    assert timestamps_non_decreasing("2024-01-10T08:00:00Z", "2024-01-10T08:00:05Z")
    assert not timestamps_non_decreasing("2024-01-10T08:00:05Z", "2024-01-10T08:00:00Z")


def test_order_id_canonicalization():
    assert canonical_order_id("1001") == "1001"
    assert canonical_order_id(1001) == "1001"
    with pytest.raises(Exception):
        canonical_order_id("01001")
    with pytest.raises(Exception):
        canonical_order_id("-1001")


# ========================================================================== #
# Deterministic test vectors (reproducibility anchors)
# ========================================================================== #
TEST_VECTOR_BYTES = b'{"event_type":"ORDER_CREATED","order_id":"1001","value":1}'
TEST_VECTOR_SHA256 = "f3a60396d0dc033b2e3f68b6c80b24ab0b86f22ae7d2f7bf3327e730546880fc"
GENESIS_VECTOR_SHA256 = "a393bff2d61008fe9f586db0c6506cf7b0d2c45683b3d17f63af4cfaea3ffecf"


def test_fixed_canonical_vector_bytes():
    vector = {"order_id": "1001", "event_type": "ORDER_CREATED", "value": 1}
    assert canonical_bytes(vector) == TEST_VECTOR_BYTES


def test_fixed_canonical_vector_hash():
    vector = {"order_id": "1001", "event_type": "ORDER_CREATED", "value": 1}
    assert sha256_hex(vector) == TEST_VECTOR_SHA256


def test_fixed_genesis_hash_reproducible():
    chain = OrderChain("1001")
    chain.seed_genesis("2024-01-10T08:00:00Z")
    assert chain.last_block.block_hash == GENESIS_VECTOR_SHA256
    assert chain.last_block.block_hash == construct_genesis_block(
        "1001", "2024-01-10T08:00:00Z"
    ).block_hash


# ========================================================================== #
# B. Hashing
# ========================================================================== #
def test_sha256_hex_lowercase_64():
    digest = sha256_hex({"a": 1})
    assert len(digest) == 64
    assert digest == digest.lower()
    assert is_sha256_hex(digest)


def test_sha256_bytes_primitive():
    assert sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()


def test_hash_changes_on_field_change():
    assert sha256_hex({"a": 1}) != sha256_hex({"a": 2})


def test_block_hash_excludes_block_hash_field():
    block = construct_block(
        order_id="1001",
        block_index=1,
        event_type="ORDER_CREATED",
        event_timestamp="2024-01-10T08:00:05Z",
        payload_digest=_dig("created"),
        previous_hash="0" * 64,
    )
    with_block_hash = sha256_hex({**block.preimage(), "block_hash": block.block_hash})
    assert with_block_hash != block.block_hash


def test_block_hash_equals_hash_of_preimage():
    block = construct_block(
        order_id="1001",
        block_index=1,
        event_type="ORDER_CREATED",
        event_timestamp="2024-01-10T08:00:05Z",
        payload_digest=_dig("created"),
        previous_hash="0" * 64,
    )
    assert sha256_hex(block.preimage()) == block.block_hash


# ========================================================================== #
# C. Genesis
# ========================================================================== #
def test_genesis_deterministic():
    a = construct_genesis_block("1001", "2024-01-10T08:00:00Z")
    b = construct_genesis_block("1001", "2024-01-10T08:00:00Z")
    assert a == b
    assert a.block_hash == b.block_hash


def test_genesis_structural_properties():
    g = construct_genesis_block("1001", "2024-01-10T08:00:00Z")
    assert g.block_index == 0
    assert g.event_type == "ORDER_GENESIS"
    assert g.previous_hash is None
    assert g.event_timestamp == "2024-01-10T08:00:00Z"


def test_genesis_payload_digest_from_content():
    g = construct_genesis_block("1001", "2024-01-10T08:00:00Z")
    expected = sha256_hex(
        {"order_id": "1001", "order_creation_canonical": "2024-01-10T08:00:00Z"}
    )
    assert g.payload_digest == expected
    assert genesis_payload_digest("1001", "2024-01-10T08:00:00Z") == expected


def test_genesis_chain_header():
    chain = OrderChain("1001")
    g = chain.seed_genesis("2024-01-10T08:00:00Z")
    assert chain.order_id == "1001"
    assert chain.length == 1
    assert g.order_id == "1001"


# ========================================================================== #
# D. Block
# ========================================================================== #
def test_block_schema_completeness():
    block = construct_block(
        order_id="1001",
        block_index=1,
        event_type="ORDER_CREATED",
        event_timestamp="2024-01-10T08:00:05Z",
        payload_digest=_dig("created"),
        previous_hash="0" * 64,
    )
    fields = {
        "schema_version",
        "order_id",
        "block_index",
        "event_type",
        "event_timestamp",
        "payload_digest",
        "payload_ref",
        "ai_risk_reference",
        "validation_policy",
        "previous_hash",
        "block_hash",
    }
    assert set(block.as_dict().keys()) == fields


def test_block_is_frozen_dataclass():
    block = construct_block(
        order_id="1001",
        block_index=1,
        event_type="ORDER_CREATED",
        event_timestamp="2024-01-10T08:00:05Z",
        payload_digest=_dig("created"),
        previous_hash="0" * 64,
    )
    assert dataclasses.is_dataclass(block)
    with pytest.raises(dataclasses.FrozenInstanceError):
        block.payload_digest = "x" * 64  # type: ignore[misc]


def test_genesis_previous_hash_required_null():
    with pytest.raises(InvalidBlockError):
        construct_block(
            order_id="1001",
            block_index=0,
            event_type="ORDER_GENESIS",
            event_timestamp="2024-01-10T08:00:00Z",
            payload_digest=_dig("g"),
            previous_hash="0" * 64,
        )


def test_non_genesis_previous_hash_required():
    with pytest.raises(InvalidBlockError):
        construct_block(
            order_id="1001",
            block_index=1,
            event_type="ORDER_CREATED",
            event_timestamp="2024-01-10T08:00:05Z",
            payload_digest=_dig("created"),
            previous_hash=None,
        )


def test_block_rejects_bad_payload_digest():
    with pytest.raises(InvalidBlockError):
        construct_block(
            order_id="1001",
            block_index=1,
            event_type="ORDER_CREATED",
            event_timestamp="2024-01-10T08:00:05Z",
            payload_digest="not-a-hash",
            previous_hash="0" * 64,
        )


def test_block_rejects_bad_event_type():
    with pytest.raises(InvalidBlockError):
        construct_block(
            order_id="1001",
            block_index=1,
            event_type="NOT_AN_EVENT",
            event_timestamp="2024-01-10T08:00:05Z",
            payload_digest=_dig("x"),
            previous_hash="0" * 64,
        )


def test_block_rejects_negative_index():
    with pytest.raises(InvalidBlockError):
        construct_block(
            order_id="1001",
            block_index=-1,
            event_type="ORDER_CREATED",
            event_timestamp="2024-01-10T08:00:05Z",
            payload_digest=_dig("x"),
            previous_hash="0" * 64,
        )


# ========================================================================== #
# E. OrderChain
# ========================================================================== #
def test_chain_events_follow_frozen_sequence(_complete_chain_e=None):
    chain = _complete_chain("1001")
    assert [b.event_type for b in chain.blocks] == list(FIXED_EVENT_SEQUENCE)


def test_chain_index_increment_and_linking():
    chain = OrderChain("1001")
    chain.seed_genesis("2024-01-10T08:00:00Z")
    for i, event in enumerate(FIXED_EVENT_SEQUENCE[1:], start=1):
        block = chain.append_event(event, f"2024-01-1{i}T08:00:00Z", _dig(event))
        assert block.block_index == i
        assert block.previous_hash == chain.blocks[i - 1].block_hash


def test_multiple_independent_order_chains():
    a = _complete_chain("1001")
    b = _complete_chain("1002")
    assert a.order_id != b.order_id
    assert a.blocks[0].block_hash != b.blocks[0].block_hash
    assert all(blk.order_id == "1001" for blk in a.blocks)
    assert all(blk.order_id == "1002" for blk in b.blocks)


def test_chain_blocks_readonly():
    chain = _complete_chain("1001")
    blocks = chain.blocks
    assert isinstance(blocks, tuple)
    with pytest.raises(AttributeError):
        blocks.append(blocks[0])  # type: ignore[attr-defined]


def test_timestamps_non_decreasing_enforced():
    chain = OrderChain("1001")
    chain.seed_genesis("2024-01-10T08:00:00Z")
    chain.append_event("ORDER_CREATED", "2024-01-10T08:00:05Z", _dig("created"))
    with pytest.raises(InvalidEventTransitionError):
        chain.append_event("SHIPMENT_RECORDED", "2024-01-10T08:00:01Z", _dig("ship"))


# ========================================================================== #
# F. Invalid behavior
# ========================================================================== #
def test_cross_order_append_rejected():
    chain = _complete_chain("1001")
    foreign = construct_block(
        order_id="200",
        block_index=chain.length,
        event_type="ORDER_CREATED",
        event_timestamp="2024-02-01T08:00:05Z",
        payload_digest=_dig("foreign"),
        previous_hash=chain.last_block.block_hash,
    )
    with pytest.raises(InvalidBlockError) as exc:
        chain.append_block(foreign)
    assert "cross-order" in str(exc.value)


def test_wrong_previous_hash_rejected():
    chain = OrderChain("1001")
    chain.seed_genesis("2024-01-10T08:00:00Z")
    chain.append_event("ORDER_CREATED", "2024-01-10T08:00:05Z", _dig("created"))
    wrong_prev = construct_block(
        order_id="1001",
        block_index=chain.length,
        event_type="SHIPMENT_RECORDED",
        event_timestamp="2024-01-12T09:30:00Z",
        payload_digest=_dig("ship"),
        previous_hash="0" * 64,
    )
    with pytest.raises(InvalidBlockError):
        chain.append_block(wrong_prev)


def test_wrong_index_rejected():
    chain = _complete_chain("1001")
    bad_index = construct_block(
        order_id="1001",
        block_index=99,
        event_type="ORDER_CREATED",
        event_timestamp="2024-02-01T08:00:05Z",
        payload_digest=_dig("x"),
        previous_hash=chain.last_block.block_hash,
    )
    with pytest.raises(InvalidBlockError) as exc:
        chain.append_block(bad_index)
    assert "index gap" in str(exc.value)


def test_invalid_event_transition_rejected():
    chain = OrderChain("1001")
    chain.seed_genesis("2024-01-10T08:00:00Z")
    with pytest.raises(InvalidEventTransitionError):
        chain.append_event("AI_RISK_ASSESSED", "2024-01-10T08:00:05Z", _dig("skip"))


def test_event_duplication_rejected():
    chain = OrderChain("1001")
    chain.seed_genesis("2024-01-10T08:00:00Z")
    chain.append_event("ORDER_CREATED", "2024-01-10T08:00:05Z", _dig("created"))
    with pytest.raises(InvalidEventTransitionError):
        chain.append_event("ORDER_CREATED", "2024-01-10T08:00:06Z", _dig("again"))


def test_malformed_hash_rejected_at_construction():
    with pytest.raises(InvalidBlockError):
        construct_block(
            order_id="1001",
            block_index=1,
            event_type="ORDER_CREATED",
            event_timestamp="2024-01-10T08:00:05Z",
            payload_digest="g" * 64,  # not lowercase hex digest form
            previous_hash="0" * 64,
        )


# ========================================================================== #
# G. Validation checks c1-c8
# ========================================================================== #
def test_all_checks_pass_on_intact_chain():
    chain = _complete_chain("1001")
    ctx = ValidationContext.for_block(chain.last_block, chain=chain.blocks, order_id="1001")
    for check in (
        check_c1_schema,
        check_c2_self_hash,
        check_c3_previous_hash_link,
        check_c4_timestamp_order,
        check_c5_order_integrity,
        check_c6_ai_reference_present,
        check_c7_ai_reference_digest,
        check_c8_chain_prefix,
    ):
        result = check(chain.last_block, ctx)
        assert result.passed, f"{result.check_id}: {result.reason}"


def test_check_results_have_structured_evidence():
    chain = _complete_chain("1001")
    ctx = ValidationContext.for_block(chain.last_block, chain=chain.blocks, order_id="1001")
    result = check_c1_schema(chain.last_block, ctx)
    assert isinstance(result, CheckResult)
    assert result.check_id == "c1"
    assert result.passed in (True, False)
    assert isinstance(result.reason, str)


def test_c2_detects_stale_hash():
    chain = _complete_chain("1001")
    tampered = dataclasses.replace(chain.blocks[2], payload_digest=_dig("changed"))
    result = check_c2_self_hash(tampered, ValidationContext.for_block(
        tampered, chain=(*chain.blocks[:2], tampered, *chain.blocks[3:]), order_id="1001"))
    assert not result.passed
    assert "mismatch" in result.reason


def test_c3_detects_broken_link():
    chain = _complete_chain("1001")
    tampered = dataclasses.replace(chain.blocks[2], previous_hash="0" * 64)
    ctx = ValidationContext.for_block(
        tampered, chain=(*chain.blocks[:2], tampered, *chain.blocks[3:]), order_id="1001"
    )
    assert not check_c3_previous_hash_link(tampered, ctx).passed


def test_c4_detects_timestamp_regression():
    chain = _complete_chain("1001")
    tampered = dataclasses.replace(
        chain.blocks[2], event_timestamp="2024-01-10T07:00:00Z"
    )
    ctx = ValidationContext.for_block(
        tampered, chain=(*chain.blocks[:2], tampered, *chain.blocks[3:]), order_id="1001"
    )
    assert not check_c4_timestamp_order(tampered, ctx).passed


def test_c5_detects_order_integrity_violation():
    chain = _complete_chain("1001")
    ctx = ValidationContext.for_block(chain.last_block, chain=chain.blocks, order_id="9999")
    assert not check_c5_order_integrity(chain.last_block, ctx).passed


def test_c6_requires_reference_on_ai_linked_block():
    chain = _complete_chain("1001")
    no_ref = dataclasses.replace(chain.last_block, ai_risk_reference=None)
    ctx = ValidationContext.for_block(no_ref, chain=chain.blocks, order_id="1001")
    assert not check_c6_ai_reference_present(no_ref, ctx).passed
    assert check_c6_ai_reference_present(chain.blocks[0], ctx).passed


def test_c7_detects_tampered_reference_digest():
    chain = _complete_chain("1001")
    tampered_ref = dict(chain.last_block.ai_risk_reference)
    tampered_ref["risk_score"] = 0.5
    tampered = dataclasses.replace(chain.last_block, ai_risk_reference=tampered_ref)
    ctx = ValidationContext.for_block(tampered, chain=chain.blocks, order_id="1001")
    assert not check_c7_ai_reference_digest(tampered, ctx).passed


def test_c8_detects_prior_block_hash_change():
    chain = _complete_chain("1001")
    tampered = dataclasses.replace(chain.blocks[1], payload_digest=_dig("changed"))
    blocks = (chain.blocks[0], tampered, *chain.blocks[2:])
    ctx = ValidationContext.for_block(blocks[-1], chain=blocks, order_id="1001")
    assert not check_c8_chain_prefix(blocks[-1], ctx).passed


def test_apply_checks_deterministic():
    chain = _complete_chain("1001")
    checks = ("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8")
    first = apply_checks(chain.last_block, checks, chain=chain.blocks, order_id="1001")
    second = apply_checks(chain.last_block, checks, chain=chain.blocks, order_id="1001")
    assert [(r.check_id, r.passed, r.reason) for r in first] == [
        (r.check_id, r.passed, r.reason) for r in second
    ]


def test_unknown_check_raises():
    chain = _complete_chain("1001")
    with pytest.raises(Exception):
        apply_checks(chain.last_block, ["c99"], chain=chain.blocks, order_id="1001")


# ========================================================================== #
# H. ALV mechanics
# ========================================================================== #
def test_band_mapping_matches_frozen_policy(cfg):
    config_bands = cfg["adaptive_lightweight_validation"]["band_to_checks"]
    for band in ("LOW", "MEDIUM", "HIGH"):
        assert ALV_BAND_TO_CHECKS[band] == tuple(config_bands[band]["checks"])
        assert ALV_VALIDATOR_COUNTS[band] == config_bands[band]["validator_count"]


def test_low_path():
    chain = _complete_chain("1001", score=0.1)
    res = validate_adaptive_alv(chain.last_block, chain=chain.blocks, order_id="1001")
    assert res.accepted is True
    assert res.validation_level == "LOW"
    assert res.validator_count == 1
    assert res.applied_checks == ("c1", "c2", "c3", "c4")


def test_medium_path():
    chain = _complete_chain("1001", score=0.5)
    res = validate_adaptive_alv(chain.last_block, chain=chain.blocks, order_id="1001")
    assert res.accepted is True
    assert res.validation_level == "MEDIUM"
    assert res.validator_count == 1
    assert res.applied_checks == ("c1", "c2", "c3", "c4", "c5", "c6")


def test_high_path_three_validators():
    chain = _complete_chain("1001", score=0.8)
    res = validate_adaptive_alv(chain.last_block, chain=chain.blocks, order_id="1001")
    assert res.accepted is True
    assert res.validation_level == "HIGH"
    assert res.validator_count == 3
    assert res.applied_checks == ("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8")


def test_high_deterministic_quorum_agrees():
    chain = _complete_chain("1001", score=0.8)
    res = validate_adaptive_alv(chain.last_block, chain=chain.blocks, order_id="1001")
    assert res.accepted is True
    assert res.discrepancy is False
    assert "validator_discrepancy" not in res.reason_codes


def test_quorum_rejects_on_input_discrepancy():
    chain = _complete_chain("1001", score=0.8)
    ctx_a = ValidationContext.for_block(chain.last_block, chain=chain.blocks, order_id="1001")
    ctx_b = ValidationContext.for_block(chain.last_block, chain=chain.blocks, order_id="9999")
    res = validate_with_quorum(
        chain.last_block,
        ("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"),
        validator_count=3,
        contexts=[ctx_a, ctx_a, ctx_b],
    )
    assert res.accepted is False
    assert res.discrepancy is True
    assert "validator_discrepancy" in res.reason_codes


def test_fail_safe_missing_reference_escalates_to_medium():
    chain = OrderChain("1001")
    chain.seed_genesis("2024-01-10T08:00:00Z")
    chain.append_event("ORDER_CREATED", "2024-01-10T08:00:05Z", _dig("created"))
    chain.append_event("SHIPMENT_RECORDED", "2024-01-12T09:30:00Z", _dig("ship"))
    chain.append_event("DELIVERY_STATUS_RECORDED", "2024-01-15T14:00:00Z", _dig("deliv"))
    chain.append_event("AI_RISK_ASSESSED", "2024-01-15T14:00:05Z", _dig("ai"), ai_risk_reference=None)
    level, note = resolve_adaptive_validation_level(
        chain.last_block, chain=chain.blocks, order_id="1001"
    )
    assert level == "MEDIUM"
    assert "fail_safe" in note


def test_fail_safe_malformed_reference_escalates_to_medium():
    chain = OrderChain("1001")
    chain.seed_genesis("2024-01-10T08:00:00Z")
    chain.append_event("ORDER_CREATED", "2024-01-10T08:00:05Z", _dig("created"))
    chain.append_event("SHIPMENT_RECORDED", "2024-01-12T09:30:00Z", _dig("ship"))
    chain.append_event("DELIVERY_STATUS_RECORDED", "2024-01-15T14:00:00Z", _dig("deliv"))
    bad_ref = synthetic_risk_reference("1001", 0.8)
    bad_ref["record_digest"] = "0" * 64
    chain.append_event(
        "AI_RISK_ASSESSED", "2024-01-15T14:00:05Z", _dig("ai"), ai_risk_reference=bad_ref
    )
    level, note = resolve_adaptive_validation_level(
        chain.last_block, chain=chain.blocks, order_id="1001"
    )
    assert level == "MEDIUM"
    assert "fail_safe" in note


def test_unknown_risk_level_not_mapped_to_low():
    ref = synthetic_risk_reference("1001", 0.1)
    ref["risk_level"] = "CRITICAL"
    ref["record_digest"] = risk_record_digest({
        "order_id": ref["order_id"],
        "risk_score": ref["risk_score"],
        "risk_level": ref["risk_level"],
        "model_configuration_provenance": ref["model_configuration_provenance"],
        "hybrid_k13_provenance": ref["hybrid_k13_provenance"],
        "generation_stage_provenance": ref["generation_stage_provenance"],
    })
    chain = OrderChain("1001")
    chain.seed_genesis("2024-01-10T08:00:00Z")
    chain.append_event("ORDER_CREATED", "2024-01-10T08:00:05Z", _dig("created"))
    chain.append_event("SHIPMENT_RECORDED", "2024-01-12T09:30:00Z", _dig("ship"))
    chain.append_event("DELIVERY_STATUS_RECORDED", "2024-01-15T14:00:00Z", _dig("deliv"))
    chain.append_event("AI_RISK_ASSESSED", "2024-01-15T14:00:05Z", _dig("ai"), ai_risk_reference=ref)
    level, _ = resolve_adaptive_validation_level(
        chain.last_block, chain=chain.blocks, order_id="1001"
    )
    assert level == "MEDIUM"


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.0, "LOW"), (0.3332, "LOW"), (0.3333, "MEDIUM"), (0.6666, "MEDIUM"), (0.6667, "HIGH"), (1.0, "HIGH")],
)
def test_risk_band_from_score_pre_registered(score, expected):
    assert risk_band_from_score(score) == expected


def test_synthetic_reference_never_ai_labeled():
    ref = synthetic_risk_reference("1001", 0.8)
    assert ref["generation_stage_provenance"] == "SYNTHETIC_TEST_FIXTURE"
    assert ref["model_configuration_provenance"] == "SYNTHETIC_TEST_FIXTURE"
    assert "AI" not in ref["generation_stage_provenance"]


def test_authorized_validator_local_deterministic():
    chain = _complete_chain("1001", score=0.5)
    validator = AuthorizedValidator(("c1", "c2", "c3", "c4", "c5", "c6"))
    res = validator.validate(chain.last_block, chain=chain.blocks, order_id="1001")
    assert res.accepted is True
    assert res.applied_checks == ("c1", "c2", "c3", "c4", "c5", "c6")


def test_validation_result_schema():
    chain = _complete_chain("1001", score=0.5)
    res = validate_adaptive_alv(chain.last_block, chain=chain.blocks, order_id="1001")
    assert hasattr(res, "accepted")
    assert hasattr(res, "risk_level")
    assert hasattr(res, "validation_level")
    assert hasattr(res, "applied_checks")
    assert hasattr(res, "passed_checks")
    assert hasattr(res, "failed_checks")
    assert hasattr(res, "validator_count")
    assert hasattr(res, "reason_codes")


# ========================================================================== #
# Protocol conformance vs frozen config
# ========================================================================== #
def test_architecture_conformance(cfg):
    assert cfg["architecture"]["primary_classification"] == "ORDER_CENTRIC_PER_ORDER_LOGICAL_BLOCKCHAIN"
    assert cfg["block_schema"]["schema_version"] == "1"


def test_event_catalog_matches_frozen_sequence(cfg):
    types = list(cfg["events"]["types"].keys())
    assert types == list(FIXED_EVENT_SEQUENCE)
    assert set(types) == set(EVENT_TYPES)


def test_canonical_serialization_conformance(cfg):
    rule = cfg["canonical_serialization"]
    assert rule["sort_keys"] is True
    assert rule["separators"] == [",", ":"]
    assert rule["ensure_ascii"] is True
    assert rule["allow_nan"] is False
    assert rule["trailing_newline"] is False
    assert rule["encoding"] == "utf-8"


def test_hash_policy_conformance(cfg):
    assert cfg["hash"]["algorithm"] == "sha256"
    assert cfg["hash"]["preimage_excludes"] == ["block_hash"]


def test_genesis_conformance(cfg):
    assert cfg["genesis"]["index"] == 0
    assert cfg["genesis"]["event_type"] == "ORDER_GENESIS"
    assert cfg["genesis"]["previous_hash"] == "null"


def test_validation_core_checks_conformance(cfg):
    checks = set(cfg["validation"]["core_checks"].keys())
    assert checks == {
        "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8",
    }
    assert cfg["validation"]["mode"] == "AUTHORIZED_VALIDATOR_LIGHTWEIGHT"


def test_fail_safe_conformance(cfg):
    fs = cfg["adaptive_lightweight_validation"]["fail_safe"]
    assert "MEDIUM" in fs
    assert "escalates" in fs


def test_tie_handling_conformance(cfg):
    tie = cfg["adaptive_lightweight_validation"]["tie_handling"]
    assert "REJECTED" in tie
    assert "deterministic" in tie


def test_no_pow_no_mining(cfg):
    assert cfg["scope"]["pow_required_allowed"] is False
    assert cfg["scope"]["mining_allowed"] is False


def test_block_schema_conformance(cfg):
    fields = set(cfg["block_schema"]["fields"].keys())
    assert fields == {
        "schema_version",
        "order_id",
        "block_index",
        "event_type",
        "event_timestamp",
        "payload_digest",
        "payload_ref",
        "ai_risk_reference",
        "validation_policy",
        "previous_hash",
        "block_hash",
    }


# ========================================================================== #
# I. Governance / no scientific leakage
# ========================================================================== #
@pytest.mark.parametrize(
    "forbidden",
    [
        "read_csv",
        "pandas",
        ".fit(",
        ".predict",
        ".predict_proba",
        "bpso",
        "bgwo",
        "hybrid_optimizer",
        "joblib",
        "checkpoint",
        "datetime.now",
        "time.time",
        "random.",
        "numpy",
    ],
)
def test_engine_source_has_no_forbidden_leakage(forbidden):
    source = "\n".join(p.read_text(encoding="utf-8") for p in ENGINE_DIR.glob("*.py"))
    assert forbidden not in source, f"forbidden token {forbidden!r} found in engine source"


def test_engine_has_no_import_of_blockchain_legacy_or_ai():
    source = "\n".join(p.read_text(encoding="utf-8") for p in ENGINE_DIR.glob("*.py"))
    assert "src.blockchain." not in source
    assert "src.pipeline_v" not in source
    assert "src.ai" not in source


def test_engine_does_not_read_dataco_files():
    import os

    engine_files = list(ENGINE_DIR.glob("*.py"))
    source = "\n".join(p.read_text(encoding="utf-8") for p in engine_files)
    assert "data/raw" not in source
    assert "data/processed" not in source
    assert "test_dataco" not in source


def test_governance_constants_frozen():
    assert all(level in VALIDATION_POLICIES for level in (
        "AUTHORIZED_VALIDATOR_LOW",
        "AUTHORIZED_VALIDATOR_MEDIUM",
        "AUTHORIZED_VALIDATOR_HIGH",
    ))
    assert VALIDATION_POLICY_PREFIX == "AUTHORIZED_VALIDATOR_"
    assert DEFAULT_VALIDATION_POLICY == "AUTHORIZED_VALIDATOR_MEDIUM"


def test_no_random_nonce_field():
    block = construct_block(
        order_id="1001",
        block_index=1,
        event_type="ORDER_CREATED",
        event_timestamp="2024-01-10T08:00:05Z",
        payload_digest=_dig("created"),
        previous_hash="0" * 64,
    )
    assert "nonce" not in block.as_dict()
    assert "pow" not in block.as_dict().keys() and "proof" not in block.as_dict().keys()


def test_v11b_boundary_no_risk_generation(cfg):
    v11b = cfg["future_boundaries"]["v11b"]
    assert "GOERVNED" not in v11b  # governance typo guard
    assert "GOVERNED_AI_RISK" not in v11b