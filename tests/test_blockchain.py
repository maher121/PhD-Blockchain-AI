"""Unit tests for the simulated blockchain (Prototype V0.1)."""

from __future__ import annotations

import pytest

from src.blockchain.core import (
    build_chain_from_transactions,
    create_block,
    create_genesis_block,
    create_transaction,
)
from src.blockchain.crypto import calculate_block_hash, calculate_transaction_hash
from src.blockchain.validation import (
    detect_tampering,
    validate_block,
    validate_transaction,
    verify_chain,
)

SAMPLE_TIMESTAMP = "2024-06-01T10:00:00+00:00"


@pytest.fixture
def sample_transaction() -> dict:
    """A canonical, valid transaction record."""
    return create_transaction(
        transaction_id="TXN-000001",
        participant_id="supplier_a",
        product_id="PROD-001",
        order_id="ORD-000001",
        timestamp=SAMPLE_TIMESTAMP,
        quantity=120.0,
        transaction_status="completed",
    )


@pytest.fixture
def sample_chain() -> list[dict]:
    """A small intact chain: genesis + two data blocks with three transactions."""
    transactions = [
        create_transaction(
            transaction_id=f"TXN-{i:06d}",
            participant_id="supplier_b",
            product_id="PROD-002",
            order_id=f"ORD-{i:06d}",
            timestamp=SAMPLE_TIMESTAMP,
            quantity=float(i * 10),
            transaction_status="completed",
        )
        for i in range(1, 4)
    ]
    return build_chain_from_transactions(transactions, max_transactions=2)


# --- transaction hashing -------------------------------------------------


def test_transaction_hash_is_deterministic(sample_transaction: dict) -> None:
    hash_a = calculate_transaction_hash(sample_transaction)
    hash_b = calculate_transaction_hash(sample_transaction)
    assert hash_a == hash_b
    assert len(hash_a) == 64


def test_transaction_hash_changes_when_fields_change(sample_transaction: dict) -> None:
    original_hash = calculate_transaction_hash(sample_transaction)
    modified = dict(sample_transaction)
    modified["quantity"] = modified["quantity"] + 1.0
    assert calculate_transaction_hash(modified) != original_hash


def test_transaction_hash_excludes_stored_data_hash(sample_transaction: dict) -> None:
    hashed = calculate_transaction_hash(sample_transaction)
    with_manual_hash = dict(sample_transaction)
    with_manual_hash["data_hash"] = "x" * 64
    assert calculate_transaction_hash(with_manual_hash) == hashed


# --- transaction validation ---------------------------------------------


def test_valid_transaction_passes_validation(sample_transaction: dict) -> None:
    assert validate_transaction(sample_transaction) is True


def test_missing_field_fails_validation(sample_transaction: dict) -> None:
    incomplete = {key: value for key, value in sample_transaction.items() if key != "order_id"}
    assert validate_transaction(incomplete) is False


def test_stale_hash_fails_validation(sample_transaction: dict) -> None:
    tampered = dict(sample_transaction)
    tampered["quantity"] = 0.5
    assert validate_transaction(tampered) is False


@pytest.mark.parametrize("bad_quantity", [0.0, -5.0, "not-a-number"])
def test_invalid_quantity_fails_validation(sample_transaction: dict, bad_quantity: object) -> None:
    invalid = dict(sample_transaction)
    invalid["quantity"] = bad_quantity
    assert validate_transaction(invalid) is False


# --- block hashing -------------------------------------------------------


def test_block_hash_deterministic() -> None:
    block = create_block(transactions=[], block_id=1, previous_hash="0" * 64)
    assert calculate_block_hash(block) == calculate_block_hash(block)
    assert len(block["current_block_hash"]) == 64


def test_block_hash_changes_when_transactions_change(sample_transaction: dict) -> None:
    block_a = create_block(transactions=[sample_transaction], block_id=1, previous_hash="0" * 64)
    modified = dict(sample_transaction)
    modified["quantity"] = modified["quantity"] + 10.0
    block_b = create_block(transactions=[modified], block_id=1, previous_hash="0" * 64)
    assert block_a["current_block_hash"] != block_b["current_block_hash"]


def test_block_hash_changes_when_previous_hash_changes(sample_transaction: dict) -> None:
    block_a = create_block(transactions=[sample_transaction], block_id=2, previous_hash="a" * 64)
    block_b = create_block(transactions=[sample_transaction], block_id=2, previous_hash=block_a["current_block_hash"])
    assert block_a["current_block_hash"] != block_b["current_block_hash"]


# --- block validation ----------------------------------------------------


def test_valid_block_passes_validation(sample_transaction: dict) -> None:
    block = create_block(transactions=[sample_transaction], block_id=1, previous_hash="0" * 64)
    assert validate_block(block) is True


def test_block_fails_when_hash_stale(sample_transaction: dict) -> None:
    block = create_block(transactions=[sample_transaction], block_id=1, previous_hash="0" * 64)
    block["transactions"][0]["quantity"] = 9999.0
    assert validate_block(block) is False


def test_genesis_block_is_valid() -> None:
    genesis = create_genesis_block()
    assert validate_block(genesis) is True


# --- chain verification --------------------------------------------------


def test_intact_chain_verifies(sample_chain: list[dict]) -> None:
    assert verify_chain(sample_chain) is True


def test_broken_link_fails_verification(sample_chain: list[dict]) -> None:
    sample_chain[2]["previous_block_hash"] = "f" * 64
    assert verify_chain(sample_chain) is False


def test_head_tampering_fails_verification(sample_chain: list[dict]) -> None:
    sample_chain[-1]["current_block_hash"] = "f" * 64
    assert verify_chain(sample_chain) is False


def test_empty_chain_fails_verification() -> None:
    assert verify_chain([]) is False


# --- tamper detection ----------------------------------------------------


def test_detect_tampering_clean_chain(sample_chain: list[dict]) -> None:
    report = detect_tampering(sample_chain)
    assert report["tampered"] is False
    assert report["tampered_blocks"] == []
    assert report["tampered_transactions"] == []


def test_detect_tampering_catches_transaction_edit(sample_chain: list[dict]) -> None:
    target_block = sample_chain[1]
    target_tx = target_block["transactions"][0]
    target_tx["quantity"] = -999.0
    report = detect_tampering(sample_chain)
    assert report["tampered"] is True
    assert target_tx["transaction_id"] in report["tampered_transactions"]


def test_detect_tampering_catches_block_hash_edit(sample_chain: list[dict]) -> None:
    sample_chain[1]["current_block_hash"] = "c" * 64
    report = detect_tampering(sample_chain)
    assert report["tampered"] is True
    assert 1 in report["tampered_blocks"]


def test_detect_tampering_catches_broken_link(sample_chain: list[dict]) -> None:
    sample_chain[1]["previous_block_hash"] = "b" * 64
    report = detect_tampering(sample_chain)
    assert report["tampered"] is True
    assert (0, 1) in report["broken_links"]


def test_tampering_after_block_creation_fails_integrity() -> None:
    """End-to-end: editing a sealed transaction must be detected."""
    transaction = create_transaction(
        transaction_id="TXN-77",
        participant_id="wholesaler_1",
        product_id="PROD-010",
        order_id="ORD-77",
        timestamp=SAMPLE_TIMESTAMP,
        quantity=500.0,
        transaction_status="in_transit",
    )
    chain = build_chain_from_transactions([transaction], max_transactions=8)

    assert verify_chain(chain) is True

    chain[1]["transactions"][0]["quantity"] = 5000.0

    assert verify_chain(chain) is False
    report = detect_tampering(chain)
    assert report["tampered"] is True
    assert "TXN-77" in report["tampered_transactions"]