"""Frozen V1.1-A block schema, deterministic block construction, and genesis.

The block schema mirrors `config.block_schema` exactly (11 fields). Every block
is an immutable (frozen dataclass) object; ``block_hash`` is ``H(canonical(
block_without_block_hash))`` and is computed from the canonical serialization of
all fields except ``block_hash`` itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .canonical import (
    canonical_json,
    canonical_order_id,
    is_sha256_hex,
    normalize_timestamp_utc,
    sha256_hex,
)
from .errors import InvalidBlockError

SCHEMA_VERSION = "1"

GENESIS_EVENT_TYPE = "ORDER_GENESIS"
GENESIS_BLOCK_INDEX = 0

EVENT_TYPES = frozenset(
    {
        "ORDER_GENESIS",
        "ORDER_CREATED",
        "SHIPMENT_RECORDED",
        "DELIVERY_STATUS_RECORDED",
        "AI_RISK_ASSESSED",
    }
)

FIXED_EVENT_SEQUENCE: tuple[str, ...] = (
    "ORDER_GENESIS",
    "ORDER_CREATED",
    "SHIPMENT_RECORDED",
    "DELIVERY_STATUS_RECORDED",
    "AI_RISK_ASSESSED",
)

VALIDATION_POLICY_PREFIX = "AUTHORIZED_VALIDATOR_"
VALIDATION_LEVELS = ("LOW", "MEDIUM", "HIGH")
VALIDATION_POLICIES = frozenset(
    f"{VALIDATION_POLICY_PREFIX}{level}" for level in VALIDATION_LEVELS
)
DEFAULT_VALIDATION_POLICY = f"{VALIDATION_POLICY_PREFIX}MEDIUM"

BLOCK_FIELDS = (
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
)


@dataclass(frozen=True)
class Block:
    """A single block of a per-order logical chain (exact frozen schema)."""

    schema_version: str
    order_id: str
    block_index: int
    event_type: str
    event_timestamp: str
    payload_digest: str
    payload_ref: str | None
    ai_risk_reference: dict[str, Any] | None
    validation_policy: str
    previous_hash: str | None
    block_hash: str

    def preimage(self) -> dict[str, Any]:
        """The block with ``block_hash`` excluded — the exact hash preimage.

        ``block_hash`` never participates in its own computation (frozen rule).
        All other fields (including ``None`` optional fields, serialized as
        ``null``) are present; canonical serialization sorts keys, so ordering
        is deterministic regardless of dict insertion order.
        """
        return {
            "schema_version": self.schema_version,
            "order_id": self.order_id,
            "block_index": self.block_index,
            "event_type": self.event_type,
            "event_timestamp": self.event_timestamp,
            "payload_digest": self.payload_digest,
            "payload_ref": self.payload_ref,
            "ai_risk_reference": self.ai_risk_reference,
            "validation_policy": self.validation_policy,
            "previous_hash": self.previous_hash,
        }

    def as_dict(self) -> dict[str, Any]:
        """The full block as a plain dict (all 11 fields, block_hash included)."""
        return {
            "schema_version": self.schema_version,
            "order_id": self.order_id,
            "block_index": self.block_index,
            "event_type": self.event_type,
            "event_timestamp": self.event_timestamp,
            "payload_digest": self.payload_digest,
            "payload_ref": self.payload_ref,
            "ai_risk_reference": self.ai_risk_reference,
            "validation_policy": self.validation_policy,
            "previous_hash": self.previous_hash,
            "block_hash": self.block_hash,
        }


def block_hash(block: Block) -> str:
    """Return ``sha256(canonical_bytes(block_without_block_hash))``."""
    return sha256_hex(block.preimage())


def genesis_payload_digest(order_id: str, order_creation_canonical: str) -> str:
    """Genesis ``payload_digest``: H(canonical({order_id, order creation})).

    Frozen genesis rule: content is the canonical JSON of ``{"order_id": ...,
    "order_creation_canonical": ...}`` (order creation derived from the
    governed order date). The digest is the SHA-256 of those canonical bytes.
    """
    return sha256_hex(
        {"order_id": order_id, "order_creation_canonical": order_creation_canonical}
    )


def construct_block(
    *,
    order_id: Any,
    block_index: int,
    event_type: str,
    event_timestamp: Any,
    payload_digest: str,
    validation_policy: str = DEFAULT_VALIDATION_POLICY,
    payload_ref: str | None = None,
    ai_risk_reference: Mapping[str, Any] | None = None,
    previous_hash: str | None = None,
    schema_version: str = SCHEMA_VERSION,
) -> Block:
    """Deterministically construct a validated, hashed block.

    Enforces the frozen schema constraints: non-negative block index, known
    event type (catalog and fixed-sequence membership), lowercase 64-hex
    payload digest, canonical timestamp, valid policy label, and 64-hex or
    null previous hash (null only at index 0).
    """
    canonical_oid = canonical_order_id(order_id)
    if not isinstance(block_index, int) or isinstance(block_index, bool):
        raise InvalidBlockError(
            f"block_index must be an integer, got {type(block_index).__name__}"
        )
    if block_index < 0:
        raise InvalidBlockError(f"block_index must be non-negative, got {block_index}")
    if event_type not in EVENT_TYPES:
        raise InvalidBlockError(f"Unknown event_type {event_type!r}")
    canonical_ts = normalize_timestamp_utc(event_timestamp)
    if not is_sha256_hex(payload_digest):
        raise InvalidBlockError("payload_digest must be a lowercase 64-char hex digest")
    if validation_policy not in VALIDATION_POLICIES:
        raise InvalidBlockError(f"Unknown validation_policy {validation_policy!r}")
    if payload_ref is not None and (not isinstance(payload_ref, str) or not payload_ref):
        raise InvalidBlockError("payload_ref must be a non-empty string or null")
    if block_index == 0:
        if previous_hash is not None:
            raise InvalidBlockError("genesis (index 0) must have previous_hash = null")
    else:
        if not is_sha256_hex(previous_hash):
            raise InvalidBlockError("non-genesis previous_hash must be a 64-char hex digest")

    block = Block(
        schema_version=schema_version,
        order_id=canonical_oid,
        block_index=block_index,
        event_type=event_type,
        event_timestamp=canonical_ts,
        payload_digest=payload_digest,
        payload_ref=payload_ref,
        ai_risk_reference=dict(ai_risk_reference) if ai_risk_reference is not None else None,
        validation_policy=validation_policy,
        previous_hash=previous_hash,
        block_hash="",
    )
    return Block(**{**block.as_dict(), "block_hash": block_hash(block)})


def construct_genesis_block(
    order_id: Any,
    order_creation_canonical: Any,
    *,
    validation_policy: str = DEFAULT_VALIDATION_POLICY,
) -> Block:
    """Construct the deterministic genesis block (index 0, ORDER_GENESIS).

    Reproducible from the canonical ``order_id`` and its canonical order
    creation time alone; no nonce, no random input.
    """
    canonical_oid = canonical_order_id(order_id)
    canonical_ts = normalize_timestamp_utc(order_creation_canonical)
    return construct_block(
        order_id=canonical_oid,
        block_index=GENESIS_BLOCK_INDEX,
        event_type=GENESIS_EVENT_TYPE,
        event_timestamp=canonical_ts,
        payload_digest=genesis_payload_digest(canonical_oid, canonical_ts),
        validation_policy=validation_policy,
        previous_hash=None,
    )


def block_without_hash(block: Block) -> dict[str, Any]:
    """Alias of :meth:`Block.preimage` (block content excluding block_hash)."""
    return block.preimage()


def canonical_block_string(block: Block) -> str:
    """Canonical JSON string of the full block (used for interop/fingerprints)."""
    return canonical_json(block.as_dict())