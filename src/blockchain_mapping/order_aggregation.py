"""Deterministic per-order line aggregation for V1.1-C payload digests.

Frozen V1.1-A §6 policy: order payloads aggregate that order's item lines over
canonical columns only — ``item_count`` plus per-line canonical payload digests
sorted by canonical order item id (transaction identity). The order-level
aggregate digest is the nested hash over the ordered line digests.

All rules are pure and deterministic; no DataCo file access happens here.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from src.blockchain_engine.canonical import canonical_json, canonical_order_id, sha256_hex

from .dataco_schema import (
    CANONICAL_LINE_COLUMNS,
    MISSING_OBSERVED_SENTINEL,
    ORDER_ID_COLUMN,
    ORDER_ITEM_ID_COLUMN,
    ORDER_ITEM_QUANTITY_COLUMN,
    ORDER_ITEM_TOTAL_COLUMN,
    DataCoSchemaError,
)


def canonical_numeric(value: Any, *, column: str) -> float | None:
    """Deterministic numeric canonicalization; NaN/Infinity become the sentinel.

    ``float(value)`` round-trips through canonical JSON (frozen float rule).
    NaN/Infinity are never emitted as numbers — they map to the explicit
    OBSERVED-missing sentinel so the canonical encoding stays finite (matching
    ``json.dumps(..., allow_nan=False)`` determinism).
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "inf", "+inf", "-inf"}:
            return None
        value = text
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise DataCoSchemaError(f"non-numeric value for {column!r}: {value!r}")
    if not math.isfinite(number):
        return None
    return number


def canonical_line_payload(
    row: Mapping[str, Any],
    *,
    order_id_canonical: str,
    transaction_id_canonical: str,
) -> dict[str, Any]:
    """Build the canonical per-line payload for one governed DataCo row.

    Minimal metadata: order identity (derived), transaction identity (derived),
    plus the canonical numeric columns. PII/identifier fields never appear.
    """
    return {
        "order_id": order_id_canonical,
        "transaction_id": transaction_id_canonical,
        **{
            column: canonical_numeric(row.get(column), column=column)
            for column in CANONICAL_LINE_COLUMNS
        },
    }


def line_payload_digest(payload: Mapping[str, Any]) -> str:
    """SHA-256 over the canonical serialization of a canonical line payload."""
    return sha256_hex(dict(payload))


def _transaction_sort_key(transaction_id_canonical: str) -> int:
    return int(transaction_id_canonical)


def aggregate_order_lines(
    rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate an order's rows into a deterministic order payload.

    Returns ``{"order_id", "item_count", "line_digests", "transaction_ids"}``.
    Line digests are computed over canonical per-line payloads and sorted by
    canonical order item id (transaction identity). Identical duplicate lines
    (same transaction_id AND same canonical payload) collapse deterministically;
    conflicting duplicates (same transaction_id, different payload) raise
    ``DataCoSchemaError`` (fail closed at order level).
    """
    if not rows:
        raise DataCoSchemaError("cannot aggregate an order with zero lines")

    order_id_canonical: str | None = None
    seen: dict[str, tuple[str, str]] = {}  # transaction_id -> (digest, line_payload_json)
    duplicates_collapsed = 0
    for row in rows:
        row_order_id = canonical_order_id(row[ORDER_ID_COLUMN])
        if order_id_canonical is None:
            order_id_canonical = row_order_id
        elif row_order_id != order_id_canonical:
            raise DataCoSchemaError(
                f"mixed order identities in one line group: "
                f"{order_id_canonical!r} vs {row_order_id!r}"
            )
        transaction_id = canonical_order_id(row[ORDER_ITEM_ID_COLUMN])
        payload = canonical_line_payload(
            row,
            order_id_canonical=order_id_canonical,
            transaction_id_canonical=transaction_id,
        )
        digest = line_payload_digest(payload)
        payload_json = canonical_json(payload)
        prior = seen.get(transaction_id)
        if prior is None:
            seen[transaction_id] = (digest, payload_json)
        elif prior[1] == payload_json:
            duplicates_collapsed += 1
        else:
            raise DataCoSchemaError(
                f"conflicting duplicate transaction {transaction_id!r} in order "
                f"{order_id_canonical!r}"
            )

    ordered = sorted(seen.items(), key=lambda kv: _transaction_sort_key(kv[0]))
    line_digests = [digest for _, (digest, _) in ordered]
    transaction_ids = [tid for tid, _ in ordered]
    return {
        "order_id": order_id_canonical,
        "item_count": len(line_digests),
        "line_digests": line_digests,
        "transaction_ids": transaction_ids,
        "duplicates_collapsed": duplicates_collapsed,
    }


def order_aggregate_digest(aggregate: Mapping[str, Any]) -> str:
    """Nested order-level digest over the canonical aggregate of its lines.

    Digest preimage is the canonical serialization of
    ``{"order_id", "item_count", "line_digests"}`` (sorted keys, sorted line
    digests) — the nested hash mandated by the frozen aggregation rule.
    """
    preimage = {
        "order_id": aggregate["order_id"],
        "item_count": aggregate["item_count"],
        "line_digests": aggregate["line_digests"],
    }
    return sha256_hex(preimage)


def payload_for_missing_observed() -> dict[str, Any]:
    """Explicit, deterministic representation of a missing observed attribute.

    Used when a governed observed column is absent on an order row so canonical
    JSON never carries NaN/Infinity (frozen float rule) yet absence remains an
    explicit, deterministic, documented marker.
    """
    return {"value": MISSING_OBSERVED_SENTINEL}