"""Per-order chain construction over the frozen V1.1-B engine.

V1.1-C builds *partial pre-AI* chains: genesis + ORDER_CREATED +
SHIPMENT_RECORDED + DELIVERY_STATUS_RECORDED. AI_RISK_ASSESSED is reserved for
V1.1-D and never appended here. Construction is deterministic and uses only the
frozen engine API (``OrderChain.seed_genesis`` / ``append_event`` + the frozen
c1-c8 validators).
"""

from __future__ import annotations

from typing import Any, Mapping

from src.blockchain_engine.order_chain import OrderChain
from src.blockchain_engine.validation import validate_block

from .dataco_schema import (
    DELIVERY_STATUS_COLUMN,
    ORDER_CREATED_TS_COLUMN,
    ORDER_ID_COLUMN,
    ORDER_ITEM_ID_COLUMN,
    ORDER_STATUS_COLUMN,
    SHIPMENT_TS_COLUMN,
    DataCoSchemaError,
    canonical_timestamp,
    schema_anchor_sha256,
)
from .event_mapping import build_mapping_events
from .order_aggregation import aggregate_order_lines

VALIDATION_CHECKS_V11C = ("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8")

#: Frozen V1.1-C conflict policy for per-order order-level source values.
#: FAIL_CLOSED: rows of one canonical order must agree on every order-level
#: source field; disagreement raises a deterministic error (never silently
#: chooses first/last, votes, averages, or aggregates).
ORDER_LEVEL_CONFLICT_POLICY = "FAIL_CLOSED"

#: Order-level source fields consumed by the three V1.1-C logical events.
#: Rows of one order must agree on each of these before event construction.
ORDER_LEVEL_FIELD_SOURCES: tuple[str, ...] = (
    ORDER_CREATED_TS_COLUMN,   # ORDER_CREATED event timestamp
    SHIPMENT_TS_COLUMN,        # SHIPMENT_RECORDED / DELIVERY_STATUS_RECORDED timestamp
    DELIVERY_STATUS_COLUMN,    # DELIVERY_STATUS_RECORDED payload field
    ORDER_STATUS_COLUMN,       # DELIVERY_STATUS_RECORDED payload field (when present)
)


class OrderMappingError(ValueError):
    """An order could not be deterministically constructed into a chain."""


class OrderLevelConflictError(OrderMappingError):
    """Rows of one canonical order disagree on an order-level source value.

    Raised under the frozen ``ORDER_LEVEL_CONFLICT_POLICY = FAIL_CLOSED``
    policy; carries the canonical order id and the conflicting source field
    but never any unnecessary PII.
    """


def _as_row_mapping(row: Any) -> Mapping[str, Any]:
    """Normalize a pandas Series / row into a plain mapping of column->value."""
    if isinstance(row, Mapping):
        return row
    try:  # pandas row (Series): expose .to_dict() when available
        to_dict = getattr(row, "to_dict", None)
        if to_dict is not None:
            return dict(to_dict())
    except Exception:  # pragma: no cover - defensive
        pass
    raise OrderMappingError(f"unsupported row type {type(row).__name__}")


def validate_governed_row(row: Mapping[str, Any]) -> None:
    """Fail-closed schema validation of one governed DataCo row for V1.1-C."""
    for required in (ORDER_ID_COLUMN, ORDER_ITEM_ID_COLUMN, ORDER_CREATED_TS_COLUMN, SHIPMENT_TS_COLUMN):
        if required not in row:
            raise DataCoSchemaError(f"missing required column {required!r}")
    if DELIVERY_STATUS_COLUMN not in row:
        raise DataCoSchemaError(f"missing required column {DELIVERY_STATUS_COLUMN!r}")
    # canonical forms validate the values themselves (raises when malformed)
    canonical_timestamp(row[ORDER_CREATED_TS_COLUMN])
    canonical_timestamp(row[SHIPMENT_TS_COLUMN])


def _canonical_order_level_value(row: Mapping[str, Any], source: str) -> str | None:
    """Deterministic canonical form of one order-level source value.

    Timestamps are canonicalized via the frozen canonical form (the exact value
    an event would carry); status/state values are stripped strings. A blank
    ``Order Status`` (absent observation, not mapped into the payload) yields
    ``None``.
    """
    if source in (ORDER_CREATED_TS_COLUMN, SHIPMENT_TS_COLUMN):
        return canonical_timestamp(row[source])
    value = row.get(source)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return str(value).strip()


def check_order_level_consistency(order_id: Any, rows: list[Mapping[str, Any]]) -> list[str]:
    """Fail-closed check that all of an order's rows agree on order-level fields.

    Returns the list of ``ORDER_LEVEL_FIELD_SOURCES`` consumed (one logical
    event per type is authorized). If any order-level source value differs
    across rows, raises ``OrderLevelConflictError`` (FAIL_CLOSED). No first /
    last / vote / average / silent-aggregation resolution is ever performed.
    """
    consumed: list[str] = []
    for source in ORDER_LEVEL_FIELD_SOURCES:
        values = [_canonical_order_level_value(row, source) for row in rows]
        # A blank (absent) Order Status contributes no payload field, so it is
        # not a conflict when every row is blank; otherwise any disagreement --
        # including blank-vs-present -- fails closed.
        if source == ORDER_STATUS_COLUMN and all(value is None for value in values):
            continue
        distinct = set(values)
        if len(distinct) > 1:
            raise OrderLevelConflictError(
                f"order {order_id!r}: conflicting {source!r} across governing rows "
                f"({len(distinct)} distinct values) under policy "
                f"{ORDER_LEVEL_CONFLICT_POLICY!r}"
            )
        consumed.append(source)
    return consumed


def build_order_chain(
    order_id: Any,
    rows: list[Any],
) -> dict[str, Any]:
    """Deterministically construct one order's partial pre-AI V1.1-C chain.

    Returns a descriptor dict:
    ``{order_id, aggregate, events, chain_length, blocks, last_block_hash,
    genesis_block_hash, schema_anchor, mapping_status}`` — or raises
    ``OrderMappingError`` on a fail-closed mapping failure (reported, never
    silently skipped).
    """
    if not rows:
        raise OrderMappingError(f"order {order_id!r} has zero governed rows")

    row_mappings = [_as_row_mapping(row) for row in rows]
    for row in row_mappings:
        var = row.get(ORDER_ID_COLUMN)
        validate_governed_row(row)
        if var is None:
            raise OrderMappingError(f"order {order_id!r}: null Order Id")

    # FAIL_CLOSED: all rows of one canonical order must agree on every
    # order-level source value before any order-level event is constructed.
    # The governed dataset is conflict-free today; this guards future inputs
    # from silent first/last/vote/average resolution.
    check_order_level_consistency(order_id, row_mappings)

    first = row_mappings[0]
    creation_canonical = canonical_timestamp(first[ORDER_CREATED_TS_COLUMN])
    shipment_canonical = canonical_timestamp(first[SHIPMENT_TS_COLUMN])

    aggregate = aggregate_order_lines(row_mappings)
    actual_order_id = aggregate["order_id"]
    if str(order_id).strip() != actual_order_id:
        raise OrderMappingError(
            f"order identity mismatch: requested {order_id!r}, rows define {actual_order_id!r}"
        )

    events = build_mapping_events(
        row_mappings,
        aggregate=aggregate,
        order_creation_canonical=creation_canonical,
        shipment_canonical=shipment_canonical,
    )

    chain = OrderChain(actual_order_id)
    chain.seed_genesis(creation_canonical)
    for event in events:
        try:
            chain.append_event(
                event["event_type"],
                event["event_timestamp"],
                event["payload_digest"],
                payload_ref=None,
            )
        except Exception as exc:  # engine-level deterministic rejection
            raise OrderMappingError(
                f"order {actual_order_id!r} append rejected at "
                f"{event['event_type']!r}: {exc}"
            ) from exc

    return {
        "order_id": actual_order_id,
        "aggregate": aggregate,
        "events": events,
        "chain_length": chain.length,
        "blocks": chain.blocks,
        "last_block_hash": chain.last_block.block_hash if chain.last_block else None,
        "genesis_block_hash": chain.blocks[0].block_hash if chain.blocks else None,
        "schema_anchor": schema_anchor_sha256(),
        "mapping_status": "CONSTRUCTED",
    }


def validate_constructed_chain(
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-verify a constructed chain's head via the frozen c1-c8 set.

    Returns ``{accepted, check_results}``. The head is the last block; c6/c7
    remain "not applicable" for the non-AI event types V1.1-C produces.
    """
    blocks = tuple(descriptor["blocks"])
    head = blocks[-1]
    result = validate_block(
        head,
        VALIDATION_CHECKS_V11C,
        chain=blocks,
        order_id=descriptor["order_id"],
    )
    return {
        "accepted": result.accepted,
        "applied_checks": list(result.applied_checks),
        "passed_checks": list(result.passed_checks),
        "failed_checks": list(result.failed_checks),
        "reason_codes": list(result.reason_codes),
        "check_results": {
            cr.check_id: {"passed": cr.passed, "reason": cr.reason}
            for cr in result.reasons
        },
    }