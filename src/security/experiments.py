"""Blockchain and combined security experiments for V0.4."""

from __future__ import annotations

import time
from typing import Any, Mapping

from src.blockchain.core import create_block, create_transaction
from src.blockchain.validation import verify_chain
from src.security.integrity import modify_transaction_after_sealing


FIXED_GENESIS_TIME = "2020-01-01T00:00:00+00:00"
FIXED_BLOCK_TIME = "2020-01-01T00:00:01+00:00"


def run_blockchain_integrity_experiment(
    record: Mapping[str, Any],
    *,
    tamper_source_field: str = "Order Item Quantity",
    modified_value: Any | None = None,
) -> dict[str, Any]:
    """Seal one clean transaction, edit it without rehashing, and verify twice."""
    transaction = dataco_record_to_transaction(record)
    genesis = create_block([], block_id=0, previous_hash="0" * 64, timestamp=FIXED_GENESIS_TIME)
    data_block = create_block(
        [transaction],
        block_id=1,
        previous_hash=genesis["current_block_hash"],
        timestamp=FIXED_BLOCK_TIME,
    )
    chain = [genesis, data_block]

    clean_started = time.perf_counter()
    clean_valid = verify_chain(chain)
    clean_seconds = time.perf_counter() - clean_started

    transaction_field = _blockchain_field(tamper_source_field)
    original_value = transaction[transaction_field]
    if modified_value is None:
        if not isinstance(original_value, (int, float)):
            raise ValueError("A modified value is required for non-numeric tampering.")
        modified_value = max(1.0, float(original_value) * 2.0 + 1.0)
    tampered_chain = modify_transaction_after_sealing(
        chain,
        block_id=1,
        transaction_id=transaction["transaction_id"],
        field=transaction_field,
        new_value=modified_value,
    )
    tampered_started = time.perf_counter()
    tampered_valid = verify_chain(tampered_chain)
    tampered_seconds = time.perf_counter() - tampered_started
    return {
        "clean_chain_verification": bool(clean_valid),
        "tampered_chain_verification": bool(tampered_valid),
        "integrity_failure_detected": bool(clean_valid and not tampered_valid),
        "tampered_transaction_id": transaction["transaction_id"],
        "tampered_source_field": tamper_source_field,
        "tampered_transaction_field": transaction_field,
        "original_value": original_value,
        "modified_value": modified_value,
        "clean_verification_time_seconds": clean_seconds,
        "tampered_verification_time_seconds": tampered_seconds,
        "verification_time_seconds": clean_seconds + tampered_seconds,
        "chain": chain,
        "tampered_chain": tampered_chain,
    }


def combined_security_result(
    *, ai_detected_anomaly: bool, blockchain_integrity_valid: bool
) -> dict[str, Any]:
    """Report complementary AI and ledger signals without a decision engine."""
    return {
        "ai_detected_anomaly": bool(ai_detected_anomaly),
        "blockchain_integrity_valid": bool(blockchain_integrity_valid),
    }


def dataco_record_to_transaction(record: Mapping[str, Any]) -> dict[str, Any]:
    """Map only existing DataCo trace fields into the V0.1 transaction schema."""
    extra_fields = {}
    for source, target in (
        ("Order Item Total", "order_item_total"),
        ("Days for shipping (real)", "shipping_days"),
        ("Order Region", "order_region"),
    ):
        if source in record:
            extra_fields[target] = record[source]
    return create_transaction(
        transaction_id=str(_get(record, "Order Item Id", "row_id")),
        participant_id=str(_get(record, "Order Customer Id")),
        product_id=str(_get(record, "Product Card Id")),
        order_id=str(_get(record, "Order Id")),
        timestamp=_get(record, "order date (DateOrders)"),
        quantity=float(_get(record, "Order Item Quantity")),
        transaction_status=str(_get(record, "Order Status")),
        extra_fields=extra_fields,
    )


def _get(record: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    raise ValueError(f"DataCo record lacks required blockchain field alternatives: {names}")


def _blockchain_field(source_field: str) -> str:
    mapping = {
        "Order Item Quantity": "quantity",
        "quantity": "quantity",
        "Order Item Total": "order_item_total",
        "total_amount": "order_item_total",
        "Days for shipping (real)": "shipping_days",
        "shipping_days": "shipping_days",
        "Order Status": "transaction_status",
        "transaction_status": "transaction_status",
        "order date (DateOrders)": "timestamp",
        "timestamp": "timestamp",
        "Order Region": "order_region",
        "location_region": "order_region",
    }
    if source_field not in mapping:
        raise ValueError(f"Source field {source_field!r} is not represented in the transaction.")
    return mapping[source_field]
