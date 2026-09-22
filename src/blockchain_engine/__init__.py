"""V1.1-B core lightweight blockchain engine.

Implements exactly the frozen V1.1-A protocol (order-centric per-order logical
blockchain): canonical serialization, SHA-256 hashing, deterministic blocks,
per-order chains, deterministic integrity validation, frozen c1-c8 lightweight
validation primitives, and ALV policy mechanics — ENGINE MECHANICS ONLY.

No DataCo access, no AI access, no optimizer access, no experiment execution.
"""

from __future__ import annotations

from . import adaptive_validation, block, canonical, order_chain, validation
from .adaptive_validation import (
    ALV_BAND_TO_CHECKS,
    ALV_VALIDATOR_COUNTS,
    resolve_adaptive_validation_level,
    risk_band_from_score,
    synthetic_risk_reference,
    validate_adaptive_alv,
    validate_with_quorum,
)
from .block import (
    BLOCK_FIELDS,
    DEFAULT_VALIDATION_POLICY,
    EVENT_TYPES,
    FIXED_EVENT_SEQUENCE,
    SCHEMA_VERSION,
    VALIDATION_POLICIES,
    Block,
    construct_block,
    construct_genesis_block,
    genesis_payload_digest,
)
from .canonical import (
    canonical_bytes,
    canonical_json,
    canonical_order_id,
    is_canonical_timestamp,
    is_sha256_hex,
    normalize_timestamp_utc,
    sha256_bytes,
    sha256_hex,
    timestamps_non_decreasing,
)
from .errors import (
    BlockchainEngineError,
    BlockchainValidationError,
    InvalidBlockError,
    InvalidEventTransitionError,
    InvalidTimestampError,
)
from .order_chain import OrderChain
from .validation import (
    AuthorizedValidator,
    CheckResult,
    ValidationContext,
    ValidationResult,
    apply_checks,
    risk_record_digest,
    validate_block,
)

__all__ = [
    "adaptive_validation",
    "block",
    "canonical",
    "order_chain",
    "validation",
    "ALV_BAND_TO_CHECKS",
    "ALV_VALIDATOR_COUNTS",
    "AuthorizedValidator",
    "BLOCK_FIELDS",
    "Block",
    "BlockchainEngineError",
    "BlockchainValidationError",
    "CheckResult",
    "DEFAULT_VALIDATION_POLICY",
    "EVENT_TYPES",
    "FIXED_EVENT_SEQUENCE",
    "InvalidBlockError",
    "InvalidEventTransitionError",
    "InvalidTimestampError",
    "OrderChain",
    "SCHEMA_VERSION",
    "VALIDATION_POLICIES",
    "ValidationContext",
    "ValidationResult",
    "apply_checks",
    "canonical_bytes",
    "canonical_json",
    "canonical_order_id",
    "construct_block",
    "construct_genesis_block",
    "genesis_payload_digest",
    "is_canonical_timestamp",
    "is_sha256_hex",
    "normalize_timestamp_utc",
    "resolve_adaptive_validation_level",
    "risk_band_from_score",
    "risk_record_digest",
    "sha256_bytes",
    "sha256_hex",
    "synthetic_risk_reference",
    "timestamps_non_decreasing",
    "validate_adaptive_alv",
    "validate_block",
    "validate_with_quorum",
]