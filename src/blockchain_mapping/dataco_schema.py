"""Frozen V1.1-C DataCo mapping contract: schema, columns, provenance.

This module is a dependency leaf of the mapping package: it only declares the
governed column catalogue, the provenance classification table (A/B/C), the
timestamp formats, and the schema identity anchor. It imports only the V1.1-A
config constants (aim educational: keep rules close to the frozen protocol).

DataCo is a historical tabular dataset; it is NOT a native blockchain event
log and has no provenance DAG. Every payload field therefore carries exactly
one provenance class.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Any

from src.blockchain_engine.canonical import canonical_json

SCHEMA_VERSION = "1"

DATACO_DATE_FORMAT = "%m/%d/%Y %H:%M"
DATACO_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# --------------------------------------------------------------------------- #
# Governed DataCo (SMART Supply Chain) raw column names
# --------------------------------------------------------------------------- #
ORDER_ID_COLUMN = "Order Id"
ORDER_ITEM_ID_COLUMN = "Order Item Id"
ORDER_CREATED_TS_COLUMN = "order date (DateOrders)"
SHIPMENT_TS_COLUMN = "shipping date (DateOrders)"
DELIVERY_STATUS_COLUMN = "Delivery Status"
ORDER_STATUS_COLUMN = "Order Status"
ORDER_ITEM_QUANTITY_COLUMN = "Order Item Quantity"
ORDER_ITEM_TOTAL_COLUMN = "Order Item Total"

# The canonical per-line payload carries ONLY these columns (minimal metadata).
CANONICAL_LINE_COLUMNS: tuple[str, ...] = (
    ORDER_ITEM_QUANTITY_COLUMN,
    ORDER_ITEM_TOTAL_COLUMN,
)

# Columns required to be present in the governed development scope source.
REQUIRED_COLUMNS: tuple[str, ...] = (
    ORDER_ID_COLUMN,
    ORDER_ITEM_ID_COLUMN,
    ORDER_CREATED_TS_COLUMN,
    SHIPMENT_TS_COLUMN,
    DELIVERY_STATUS_COLUMN,
)

# PII (and other high-cardinality identifier/outcome) fields that must never
# enter any canonical payload, digest, or artifact of V1.1-C.
PII_EXCLUDED_COLUMNS: tuple[str, ...] = (
    "row_id",
    "Customer Email",
    "Customer Fname",
    "Customer Lname",
    "Customer Password",
    "Customer Street",
    "Customer City",
    "Customer State",
    "Customer Zipcode",
    "Customer Country",
    "Latitude",
    "Longitude",
    "Product Name",
    "Product Description",
    "Product Image",
    "Product Card Id",
    "Category Name",
    "Customer Id",
    "Order Customer Id",
    "Order Region",
    "Order State",
    "Order Country",
    "Order City",
    "Order Zipcode",
    "Days for shipping (real)",
)

# --------------------------------------------------------------------------- #
# Provenance classification (V1.1-A §5): A observed / B derived / C generated
# --------------------------------------------------------------------------- #
OBSERVED_ATTRIBUTE = "OBSERVED_ATTRIBUTE"
DETERMINISTICALLY_DERIVED = "DETERMINISTICALLY_DERIVED"
RESEARCH_GENERATED = "RESEARCH_GENERATED"

EVENT_PROVENANCE_TABLE: dict[str, str] = {
    "ORDER_GENESIS": DETERMINISTICALLY_DERIVED,
    "ORDER_CREATED": DETERMINISTICALLY_DERIVED,
    "SHIPMENT_RECORDED": DETERMINISTICALLY_DERIVED,
    "DELIVERY_STATUS_RECORDED": OBSERVED_ATTRIBUTE,
    "AI_RISK_ASSESSED": RESEARCH_GENERATED,  # reserved for V1.1-D; never V1.1-C
}

PAYLOAD_FIELD_PROVENANCE: dict[str, str] = {
    "order_id": DETERMINISTICALLY_DERIVED,
    "order_creation_canonical": DETERMINISTICALLY_DERIVED,
    "shipment_canonical": DETERMINISTICALLY_DERIVED,
    "delivery_status": OBSERVED_ATTRIBUTE,
    "order_status": OBSERVED_ATTRIBUTE,
    "item_count": DETERMINISTICALLY_DERIVED,
    "line_digests": DETERMINISTICALLY_DERIVED,
    "order_item_quantity": OBSERVED_ATTRIBUTE,
    "order_item_total": OBSERVED_ATTRIBUTE,
}

# --------------------------------------------------------------------------- #
# Schema identity anchor
# --------------------------------------------------------------------------- #
SCHEMA_IDENTITY: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "stage": "V1.1-C",
    "architecture": "ORDER_CENTRIC_PER_ORDER_LOGICAL_BLOCKCHAIN",
    "dataset": "DataCo SMART Supply Chain",
    "event_catalog_frozen": True,
    "event_sequence": [
        "ORDER_GENESIS",
        "ORDER_CREATED",
        "SHIPMENT_RECORDED",
        "DELIVERY_STATUS_RECORDED",
        "AI_RISK_ASSESSED",
    ],
    "ai_risk_assessed_generated_in_v11c": False,
    "payload_strategy": "DIGEST_MINIMAL_METADATA",
    "order_identity_rule": "canonical decimal integer string (no sign, no leading zeros)",
    "order_creation_rule": "ORDER_CREATED event timestamp := canonical UTC form of 'order date (DateOrders)'",
    "shipment_rule": "SHIPMENT_RECORDED event timestamp := canonical UTC form of 'shipping date (DateOrders)'",
    "delivery_status_rule": "DELIVERY_STATUS_RECORDED := observed 'Delivery Status' value within payload digest",
    "aggregation_rule": "per-line canonical payload digests sorted by canonical order item id; nested order-level digest",
    "timestamp_policy": "only governed dataset timestamps; canonical UTC ISO-8601; equal timestamps ordered by frozen event sequence",
    "missing_value_policy": "NaN/Infinity never emitted; absent observed attributes serialize deterministically as an explicit OBSERVED-missing marker in canonical JSON",
    "duplicate_policy": "duplicate order-item lines within one order are deterministic-broken: identical lines collapse; conflicting lines fail closed at order level",
}

# Sentinel used in canonical payloads for a missing OBSERVED attribute.
MISSING_OBSERVED_SENTINEL = "OBSERVED_MISSING"


class DataCoSchemaError(ValueError):
    """A governed DataCo row violates the frozen V1.1-C mapping contract."""


def schema_anchor_sha256() -> str:
    """Deterministic anchor over the frozen schema identity (rule anchor)."""
    return hashlib.sha256(canonical_json(_schema_anchor_payload()).encode("utf-8")).hexdigest()


def _schema_anchor_payload() -> dict[str, Any]:
    """Canonical, sorted sub-set of SCHEMA_IDENTITY used for the rule anchor.

    Freeze ordering: keys are sorted by canonical_json; the value is a curated
    sub-set so that the anchor depends only on rule text, not on incidental
    prose or list order of unchanged siblings.
    """
    keep = (
        "schema_version",
        "stage",
        "architecture",
        "dataset",
        "event_catalog_frozen",
        "event_sequence",
        "ai_risk_assessed_generated_in_v11c",
        "payload_strategy",
        "order_identity_rule",
        "order_creation_rule",
        "shipment_rule",
        "delivery_status_rule",
        "aggregation_rule",
        "timestamp_policy",
        "missing_value_policy",
        "duplicate_policy",
    )
    return {key: SCHEMA_IDENTITY[key] for key in keep if key in SCHEMA_IDENTITY}


def canonical_timestamp(value: Any) -> str:
    """Deterministically normalize a governed DataCo timestamp to canonical UTC.

    Source format is ``%m/%d/%Y %H:%M`` (no zone). The governed dataset has no
    timezone; we normalize by reading the wall clock as UTC (documented choice,
    deterministic, no wall-clock call). Raised on any non-conforming value.
    """
    if not isinstance(value, str):
        raise DataCoSchemaError(f"timestamp must be a string, got {type(value).__name__}")
    try:
        parsed = datetime.strptime(value, DATACO_DATE_FORMAT)
    except ValueError as exc:
        raise DataCoSchemaError(
            f"unparseable DataCo timestamp {value!r} (expected {DATACO_DATE_FORMAT!r})"
        ) from exc
    return parsed.strftime(DATACO_TIMESTAMP_FORMAT)


_CANONICAL_TS_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")


def is_canonical_v11c_timestamp(value: Any) -> bool:
    return isinstance(value, str) and bool(_CANONICAL_TS_RE.fullmatch(value))