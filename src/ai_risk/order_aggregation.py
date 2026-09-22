"""Deterministic MAX row-to-order aggregation for governed risk records.

V1.1-D freezes MAX as the order-level rule: an order's risk score is the
maximum of the five-seed mean scores over every eligible validation row of
that order. Aggregation must be row-order-invariant and fail closed on any
missing or invalid input; the contributing row identifiers are aggregated
into a stable digest rather than being exposed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from collections.abc import Iterable, Sequence

from ..blockchain_engine.canonical import canonical_order_id


class OrderAggregationError(ValueError):
    """Raised when an order aggregation invariant is violated."""


@dataclass(frozen=True)
class RowScore:
    """One validated five-seed-mean row score resolved to an order."""

    row_id: str
    order_id: str
    score: float


@dataclass(frozen=True)
class OrderAggregate:
    """The frozen MAX aggregation result for a single canonical order."""

    order_id: str
    risk_score: float
    source_row_count: int
    source_rows_digest: str


def _canonical_row_id(value: object) -> str:
    if isinstance(value, bool) or value is None:
        raise OrderAggregationError(f"invalid stable row identifier: {value!r}")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        digits = value.strip()
        if digits.isdigit():
            return str(int(digits))
    raise OrderAggregationError(f"non-canonical stable row identifier: {value!r}")


def source_rows_digest(row_ids: Iterable[object]) -> str:
    """SHA-256 over canonical JSON of the sorted stable row_id strings."""
    canonical = sorted(_canonical_row_id(value) for value in row_ids)
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validated_score(score: float, row_id: str) -> float:
    value = float(score)
    if math.isnan(value) or math.isinf(value):
        raise OrderAggregationError(f"non-finite row score for row {row_id!r}")
    if value < 0.0 or value > 1.0:
        raise OrderAggregationError(
            f"row score out of the frozen [0,1] range for row {row_id!r}: {value}"
        )
    return value


def aggregate_rows(
    rows: Sequence[RowScore],
) -> OrderedDict[str, OrderAggregate]:
    """Aggregate validated rows by canonical order using the frozen MAX rule.

    The result is keyed by canonical order id and the value contains the max
    row score, the contributing row count, and a stable digest over the
    sorted contributing stable row identifiers. Malformed or out-of-range
    input fails closed rather than being dropped.
    """
    buckets: dict[str, list[RowScore]] = {}
    for row in rows:
        if not isinstance(row, RowScore):
            raise OrderAggregationError(
                f"aggregation input must be RowScore, got {type(row).__name__}"
            )
        order_id = canonical_order_id(row.order_id)
        row_id = _canonical_row_id(row.row_id)
        score = _validated_score(row.score, row_id)
        buckets.setdefault(order_id, []).append(
            RowScore(row_id=row_id, order_id=order_id, score=score)
        )

    result: OrderedDict[str, OrderAggregate] = OrderedDict()
    for order_id in sorted(buckets):
        rows_for_order = buckets[order_id]
        if not rows_for_order:
            raise OrderAggregationError(f"order {order_id!r} has no contributing rows")
        max_score = max(row.score for row in rows_for_order)
        digest = source_rows_digest(row.row_id for row in rows_for_order)
        result[order_id] = OrderAggregate(
            order_id=order_id,
            risk_score=max_score,
            source_row_count=len(rows_for_order),
            source_rows_digest=digest,
        )
    return result