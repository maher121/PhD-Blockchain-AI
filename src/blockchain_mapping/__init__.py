"""V1.1-C DataCo order/event mapping package.

Consumes the frozen V1.1-A protocol + V1.1-B engine contracts and maps the
governed DataCo development scope (non-TEST split metadata) into per-order
hash-linked chains (genesis + ORDER_CREATED + SHIPMENT_RECORDED +
DELIVERY_STATUS_RECORDED). Partial pre-AI chains are the authorized V1.1-C
representation; AI_RISK_ASSESSED is reserved for V1.1-D and never generated
here.

Data honesty: DataCo is a historical tabular dataset, not a blockchain event
log. Every payload field carries exactly one provenance class
(OBSERVED_ATTRIBUTE / DETERMINISTICALLY_DERIVED / RESEARCH_GENERATED) and
derived representations are never presented as observed events.
"""

from __future__ import annotations

from . import chain_builder, dataco_schema, event_mapping, order_aggregation
from .chain_builder import (
    ORDER_LEVEL_CONFLICT_POLICY,
    OrderLevelConflictError,
    build_order_chain,
    check_order_level_consistency,
    validate_constructed_chain,
)
from .dataco_schema import (
    DATACO_DATE_FORMAT,
    DATACO_TIMESTAMP_FORMAT,
    SCHEMA_IDENTITY,
    SCHEMA_VERSION,
    EVENT_PROVENANCE_TABLE,
    PII_EXCLUDED_COLUMNS,
    REQUIRED_COLUMNS,
    DataCoSchemaError,
)
from .event_mapping import build_mapping_events, event_payload_digests
from .order_aggregation import (
    aggregate_order_lines,
    canonical_line_payload,
    line_payload_digest,
    order_aggregate_digest,
)

__all__ = [
    "DATACO_DATE_FORMAT",
    "DATACO_TIMESTAMP_FORMAT",
    "SCHEMA_IDENTITY",
    "SCHEMA_VERSION",
    "EVENT_PROVENANCE_TABLE",
    "PII_EXCLUDED_COLUMNS",
    "REQUIRED_COLUMNS",
    "ORDER_LEVEL_CONFLICT_POLICY",
    "DataCoSchemaError",
    "OrderLevelConflictError",
    "aggregate_order_lines",
    "build_mapping_events",
    "build_order_chain",
    "canonical_line_payload",
    "chain_builder",
    "check_order_level_consistency",
    "dataco_schema",
    "event_mapping",
    "event_payload_digests",
    "line_payload_digest",
    "order_aggregate_digest",
    "order_aggregation",
    "validate_constructed_chain",
]